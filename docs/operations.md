# Operations

## Toolchain

| tool | needed for | install |
|---|---|---|
| Python ≥ 3.13 + `uv` | everything | `curl -sSfL https://astral.sh/uv/install.sh \| sh` |
| `elan` + Lean 4.35 | proofs (`VERIFIED`) | `curl -sSfL https://elan.lean-lang.org/elan-init.sh \| sh -s -- -y` |
| `bun` + Node | building the UI | `curl -fsSL https://bun.sh/install \| bash` |
| Modal account | real GPU profiling | `uv run modal setup` |

Without Lean the harness still runs end to end: proofs report `UNSUPPORTED` with
`Lean toolchain not found`, and the SLA's `verified` check therefore fails closed.
Without Modal, `--backend sim` exercises the whole loop on a cost model.

## GPU / backend matrix

| backend | numbers | `ncu` | speedup baseline | SLA can pass |
|---|---|---|---|---|
| `modal` + `--gpu H100`/`H200`/`B200`/`A100`/`L4` | measured | probed once, usually unavailable in sandboxes | torch, same device | yes |
| `sim` | cost model (`simulated: true`) | unavailable | modelled 2.4× | never |

`probe` writes `.harness/caps-<GPU>.json` (device name, arch, SM count, peak
bandwidth/TFLOPs, `ncu: true|false`). Delete that file to re-probe after an image
change.

## Modal setup

```bash
uv run modal deploy -m harness.backends.modal_gpu      # one-time (re-run after edits)
uv run modal run -m harness.backends.modal_gpu         # prints the probe result
```

The image is `nvidia/cuda:12.9.1-devel-ubuntu22.04` + `torch`/`numpy`/`pydantic`, with
a best-effort `nsight-compute` install from NVIDIA's apt repo (failures are tolerated ,
counters are optional). The harness calls the deployed functions by name
(`run_job`, `run_job_b200`, …), so deploying once is enough for many runs.

To profile a payload by hand:

```bash
uv run python - <<'PY'            # writes /tmp/job.json using the real driver + header
import json, sys; sys.path.insert(0, '.')
from harness.model import Variant
from harness.registry import get, geometry
from harness.runner.gen import bench_driver, harness_header
k = get("elementwise_affine"); env = k.shape_env(); g = geometry(k); b, f = k.sol_numbers(env)
v = Variant.build(variant_id="manual", run_id="manual", round_n=0, kernel=k, level=0,
                  transform_id=None, params={})
json.dump({"variant_id": v.id, "source": v.source,
           "harness_header": harness_header(k, env, k.buffers),
           "driver": bench_driver(v.source, v.id), "reps": 10, "warmup": 5,
           "baseline": k.baseline, "dtype": k.dtype, "bytes_moved": b, "flops": f,
           "sol_kind": k.sol.kind, "grid": g["grid"], "block": g["block"],
           "out_index": k.out_index, "buffers": k.buffers, "shape": {"N": env["N"]},
           "tol": k.tol, "ncu": False, "tensor_cores": False},
          open("/tmp/job.json", "w"))
PY
uv run modal run -m harness.backends.modal_gpu --payload-file /tmp/job.json
```

## Environment

The full list is in the README's Configuration table: `HARNESS_LLM_BASE_URL`,
`HARNESS_LLM_MODEL`, `HARNESS_STATE_DIR`, `HARNESS_LEDGER_DIR`, `HARNESS_GIT_NAME`,
`HARNESS_GIT_EMAIL`. The two LLM variables are the only ones that change what the
research stage proposes; the rest change where files land or who authors the commits.

## Tuning a run

| knob | flag / field | effect |
|---|---|---|
| SOL target | `--sol 55` | the roofline gate (percent of peak) |
| speedup target | `--speedup 1.5` | gate against the torch baseline |
| rounds | `--max-rounds 4` | hard cap |
| stall limit | `Sla(stall_rounds=2)` | consecutive non-improvements before `STALLED` |
| noise tolerance | `Sla(tolerance_pct=3.0)` | a candidate is kept if it beats the parent's `p50` **or** SOL by this much, **or** if it satisfies the SLA gate |
| timing | `Sla(reps=30, warmup=10)` | `cudaEvent` reps per profile |
| required proof | `--no-require-verified` | only for experiments: lets the gate pass without a proof |

Kernel manifests (`harness/kernels/<id>/kernel.toml`) control shape, dtype, SOL formula,
buffers, tolerance, baseline and the starting ladder level.

## Performance expectations

Measured on H100 (80 GB HBM3), 4 M `s32` elements (33.5 MB of traffic), `out = 10·x + 3`,
50 reps after 25 warmup reps, two independent runs:

| ladder rung | SOL % | p50 us |
|---|---|---|
| L0 scalar | 63.6 / 64.8 | 15.7 / 15.5 |
| L1 `int4` | 82.8 / 75.6 | 12.1 / 13.2 |
| L3 grid-stride + `__stcs` | 94.3 / 93.7 | 10.6 / 10.7 |

At 1 M elements the same kernel is launch bound and both the absolute numbers and the
spread get worse (roughly 25 to 40 % SOL between sandboxes), so prefer a problem size
that takes at least a few microseconds per launch. Each Modal job runs in a fresh
container, so the first kernels of a job see idle GPU clocks: that is why the default
warmup is 25 reps. Keep the stall limit at 2 and the keep tolerance at 3 %.

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `App 'kernel-harness' not found` | functions not deployed / redeployed elsewhere | `uv run modal deploy -m harness.backends.modal_gpu` |
| `run failed (exit -11)` | a kernel reading device memory from host code, or an out-of-range write | check the variant's staging; the driver copies inputs through host buffers |
| `numeric: FAIL` with `max_abs_err = inf` | output buffer size/shape mismatch | compare `harness_gen.h` element counts with the reference |
| `UNSUPPORTED: … dtype f32` | float kernel | expected: the model is integer-exact; use an `s32` kernel to verify |
| `UNSUPPORTED: … Lean toolchain not found` | no `lake` on `PATH` | install elan, or accept `PARTIAL`-grade runs |
| `counters: unavailable` | `ncu` missing in the image | diagnosis uses roofline + static PTX rules |
| `no applicable transform` | every catalogue rung applied or reverted | add a ladder rung or a catalogue entry |
| `git commit failed` | ledger repo has no git identity | `git config user.name/user.email`, or `HARNESS_GIT_NAME`/`HARNESS_GIT_EMAIL` (the harness never overrides an existing identity) |

## Housekeeping

```bash
rm -rf .harness/caps-H100.json                 # re-probe the device
uv run harness report --run <run_id>           # machine-readable round summary
git log --oneline -- ledgers                   # the track record
```

`.harness/` (state, logs, generated Lean) and `lean/.lake/` are ignored; `ledgers/`,
`lean/Harness/Generated` are artifacts, and only `ledgers/` is committed.
