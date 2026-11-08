# Reference

Internals for people extending or debugging KernelHarness. Start with the
[README](../README.md) if you have not run it yet.

- [Architecture](#architecture)
- [Statuses and codes](#statuses-and-codes)
- [The Lean model](#the-lean-model)
- [Operations](#operations)

## Architecture

Dependencies point one way: the **orchestrator** drives the **stages**; stages use the
**backends** and the **translator**; the translator emits Lean. Nothing depends on the UI.

```text
web/ (React) ──HTTP+SSE── harness/server.py
                               │
harness/orchestrator.py   measure → diagnose → research → optimize → verify → ledger
   │            │                    │                        │
stages/      backends/            translator/              runner/
 diagnose     sim.py (cost model)  ptx_forms.toml (81)      gen.py (bench driver, header)
 research     modal_gpu.py         parse.py   PTX → model   reference.py (numpy refs)
 optimize     peaks.py             lower.py   template → model
 verify                            interp.py  reference executor
 ledger                            emit.py    model → Lean ──► lake env lean
```

| path | role |
|---|---|
| `harness/kernels/<id>/kernel.toml` | kernel manifest: family, shape, dtype, SOL formula, buffers, tolerance, baseline |
| `harness/templates/<family>/L<n>/kernel.cu` | one CUDA template per ladder rung (`@@NAME@@` placeholders) |
| `harness/stages/catalog.toml` | bottleneck → fix catalogue with rationale and references |
| `harness/stages/optimize.py:TRANSFORMS` | fix → target rung and template parameters |

### Data

Two stores, with different jobs:

- **SQLite** (`$HARNESS_STATE_DIR/harness.db`): one row per fact, which the UI and API read.
  Tables: `runs`, `rounds`, `variants`, `events`, `profiles`, `bottlenecks`,
  `hypotheses`, `proofs`, `commits`. The profile schema is `backends/base.py:PROFILE_KEYS`.
- **Ledger** (`$HARNESS_LEDGER_DIR`): a standalone git repository, ignored by this one.
  Each round is one commit of `runs/<run>/round-NN/`; each finished run adds a commit
  updating `REPORT.md` and `index.json`. Push it anywhere to share results.

### The target check

`orchestrator.sla_met` is a pure function over the latest profile and proof. All must hold:

| check | condition |
|---|---|
| `sol` | `sol_pct >= --sol` |
| `speedup` | torch baseline exists and `speedup >= --speedup` |
| `latency` | no latency target, or `p50_ms <= latency_ms` |
| `verified` | proof is `VERIFIED` (unless `--no-require-verified`) |
| `numeric` | GPU output matched the reference |
| `measured` | numbers are not simulated |

## Statuses and codes

### Run status and stop reason

| status | stop reason | meaning |
|---|---|---|
| `running` | | loop active |
| `sla_met` | `SLA_MET` | every check passed |
| `max_rounds` | `MAX_ROUNDS` | round cap reached |
| `stalled` | `STALLED` | `stall_rounds` (default 2) rounds without improvement |
| `blocked` | `BLOCKED` | compile/launch failure, `numeric: FAIL`, no applicable fix, or a stage error |
| `stopped` | `STOPPED_BY_USER` | stopped from the UI or API |

### Round verdict

| verdict | meaning |
|---|---|
| `baseline` | round 0, the unmodified kernel |
| `improved` | beat the parent by `tolerance_pct` (3%) on p50 or SOL, or met the target; becomes the new parent |
| `reverted` | committed, but the next round starts from the parent |
| `blocked` | no usable variant |

### Proof status

| status | meaning |
|---|---|
| `VERIFIED` | Lean build clean, no `sorry`, axiom audit printed `AUDIT_OK` |
| `PARTIAL` | goals or `sorry` remain (listed) |
| `FAILED` | Lean rejected the module (errors listed) |
| `UNSUPPORTED` | outside the model: float dtype, transcendental, no lowering, or no Lean toolchain |

### Bottleneck codes

| code | fires when | typical fix |
|---|---|---|
| `LOW_OCCUPANCY` | `occupancy_pct < 50` | `launch_bound`, `reduce_smem` |
| `MEM_BW_BOUND` | memory bound; high severity if `sol_pct < 70` | `vectorize_x4`, `streaming_stores`, `grid_stride` |
| `COMPUTE_BOUND` | compute bound; high severity if `sol_pct < 70` | tensor-core rungs |
| `UNCOALESCED` | `ncu` sectors/request > 1.5, or static stride > 1 | `smem_tile`, `swizzle_128b` |
| `NO_TENSOR_CORES` | GEMM/attention without `mma`/`wgmma`/`tcgen05` | `mma_sync`, `wgmma` |
| `NO_ASYNC_COPY` | shared memory used without `cp.async` | `cp_async_double_buffer` |
| `SMALL_PROBLEM` | `grid < 2 × sm_count` | `persistent_cta` |
| `HIGH_REGISTERS` | `regs_per_thread > 96` | `launch_bound` |
| `SYNC_OVERHEAD` | `ncu` barrier-stall share > 10% | `mbarrier_pipeline` |
| `UNVERIFIED` | latest proof not `VERIFIED` | close the proof first |
| `BASELINE_UNAVAILABLE` | no torch baseline | speedup check fails closed |

## The Lean model

`lean/Harness/` is plain Lean 4 (`v4.35.0`, no Mathlib). Memory is bytes and specs are
predicates over bytes, so proofs check offline in seconds.

| module | contents |
|---|---|
| `PTX.Basic`, `PTX.Instr`, `PTX.Program` | addresses, byte maps, values; the 81 `Op` forms; `Launch` |
| `PTX.Thread` | the small-step machine `step`/`run`/`traceOf` |
| `PTX.Event`, `PTX.Execution` | events, relations (`po`, `co`, `rf`, `sw`, …) and the eight consistency axioms |
| `PTX.Observer` | `observe`, `Correct`, `nonvacuous`, `correct_of_canonical` |
| `PTX.Tcgen05`, `PTX.Cluster` | tensor memory, mbarrier completion, DSMEM and cluster barriers |
| `Lib.Core`, `Tactic` | lemmas and the `harness_auto` tactic used by generated proofs |
| `Audit` | fails the build on any axiom beyond `propext`, `Classical.choice`, `Quot.sound` |
| `Tests.*` | sanity proofs (stale read rejected, observing read accepted) |

**Executions are derived, not supplied.** A `Candidate` is a per-thread step count plus an
oracle; its events come from `step`, so the axioms constrain the program's own executions.
The axioms (`valueAgreement`, `coherenceAxiom`, `atomicityAxiom`, `scPerLocation`,
`causalityAxiom`, `noMixedSizeRace`, `constructRules`, `proxyFenceRule`) are executable
`Bool` checkers, so `decide` can check a concrete trace and a failure names the axiom.

**Top-level statement.**

```lean
def Correct (l : Launch) (post : Post) : Prop :=
  ∀ e : Candidate l, e.oracle = (Candidate.canonical l).oracle →
    Consistent e → e.Complete → post (observe e)
```

A generated module (`lean/Harness/Generated/<Variant>.lean`, checked with
`lake env lean`, never part of the library build) proves `canonConsistent` and
`canonPost` by `decide`, then `correct` via `correct_of_canonical`, then runs the audit.

**Limits.** The step machine is thread local, so cross-thread visibility comes only from
the axioms and intra-launch producer/consumer kernels fail `coherenceAxiom`. IEEE
rounding is not modelled. Proofs are about a fixed small instance, recorded in the
proof's `instantiation` field. `Complete` fixes step counts, so preempted or partial
executions are out of scope. `lean/lakefile.mathlib.toml` optionally adds Mathlib for
anyone who wants to restate specs over `ℝ`; the harness does not use it.

## Operations

### Toolchain

| tool | needed for |
|---|---|
| Python 3.13 + `uv` | everything |
| `elan` + Lean 4.35 | `VERIFIED` proofs |
| `bun` | building the UI (`cd web && bun install && bun run build`) |
| Modal account | real GPU measurements (`H100`, `H200`, `B200`, `A100`, `L4`) |

The Modal image is `nvidia/cuda:12.9.1-devel-ubuntu22.04` with torch and numpy, plus a
best-effort `nsight-compute`. Deploy once with `uv run modal deploy -m harness.backends.modal_gpu`;
`uv run modal run -m harness.backends.modal_gpu` prints the device probe. The probe is
cached in `.harness/caps-<GPU>.json`; delete it to re-probe.

### Tuning

| knob | where | effect |
|---|---|---|
| SOL / speedup target | `--sol`, `--speedup` | the target check |
| round cap | `--max-rounds` | hard stop |
| stall limit | `Sla.stall_rounds` (2) | rounds without improvement before `STALLED` |
| noise tolerance | `Sla.tolerance_pct` (3.0) | margin a candidate must beat its parent by |
| timing | `Sla.reps` (50), `Sla.warmup` (25) | `cudaEvent` repetitions |

Prefer problem sizes that take at least a few microseconds per launch. At 1M elements
the elementwise kernel is launch bound and SOL varies 25–40% between containers. Each
Modal job starts on idle clocks, which is why warmup defaults to 25.

### Troubleshooting

| symptom | fix |
|---|---|
| `App 'kernel-harness' not found` | `uv run modal deploy -m harness.backends.modal_gpu` |
| `run failed (exit -11)` | the kernel touched device memory from host code or wrote out of range |
| `numeric: FAIL`, `max_abs_err = inf` | output buffer size mismatch between manifest and reference |
| `UNSUPPORTED: … dtype f32` | expected: proofs are integer exact |
| `UNSUPPORTED: … Lean toolchain not found` | install elan, or accept unverified runs |
| `counters: unavailable` | `ncu` missing in the image; diagnosis falls back to roofline and static PTX rules |
| `no applicable transform` | every fix for the finding is applied; add a ladder rung or catalogue entry |
| `git commit failed` | the ledger repo has no identity: set `HARNESS_GIT_NAME`/`HARNESS_GIT_EMAIL` |
