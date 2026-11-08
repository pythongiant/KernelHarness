# Architecture

The harness is four layers with one direction of dependency: **stages** talk to
**backends** and the **translator**; the **translator** emits Lean; nothing depends on
the UI.

```
                 ┌──────────────────────────── web/ (Vite + React) ─────────────────────────────┐
                 │ run control · round narrative · proof viewer · provenance graph · event log  │
                 └───────────────────────────────────▲──────────────────────────────────────────┘
                                                     │ HTTP + SSE (harness/server.py)
┌────────────────────────────────────────────────────┴──────────────────────────────────────────┐
│ harness/orchestrator.py    stages: PROFILE → DIAGNOSE → RESEARCH → OPTIMIZE → VERIFY → LEDGER │
│   SLA gate · revert-on-regression · stop reasons · retries (Modal, 2×, 30 s)                  │
├───────────────┬─────────────────────────────────┬──────────────────────────┬──────────────────┤
│ stages/       │ backends/                       │ translator/              │ runner/          │
│ diagnose      │ sim.py (cost model)             │ ptx_forms.toml (81)      │ bench_main.cu    │
│ research      │ modal_gpu.py (image, probe,     │ parse.py  PTX → model    │ harness_gen.h    │
│ optimize      │   ncu, torch baseline)          │ lower.py  template → model│ reference.py    │
│ verify        │ peaks.py (device table)         │ interp.py reference exec │ (fill + refs)    │
│ ledger        │                                 │ emit.py   model → Lean   │                  │
└───────────────┴─────────────────────────────────┴──────────────────────────┴──────────────────┘
                                                     │ subprocess: lake env lean
                                                     ▼
                                       lean/Harness/… (model + generated modules)
                                                     │
                                                     ▼
                                    ledgers/runs/<run>/round-NN/ (git commit)
```

## Data model (`harness/db.py`)

SQLite at `.harness/harness.db`, WAL, one row per fact so the UI can render a round
without re-deriving anything:

| table | holds |
|---|---|
| `runs` | config JSON, status, stop reason |
| `rounds` | per-round verdict (`baseline`, `improved`, `reverted`, `blocked`) |
| `variants` | family, transform, params, rendered `.cu`, `.ptx`, sha256 |
| `events` | stage transitions and reasoning (`body_md` is what the UI shows) |
| `profiles` | the profile JSON (fixed schema, see `backends/base.py:PROFILE_KEYS`) |
| `bottlenecks` | code, severity, evidence text |
| `hypotheses` | source (`catalog` / `llm:<model>`), transform, rationale, refs |
| `proofs` | status, model, open goals, modules, axiom audit, log path |
| `commits` | run, round, sha, message |

## A round, step by step

1. `PROFILE` (baseline or candidate) builds a `ProfileJob` (source, generated header,
   shape, tolerance, reps/warmup, bytes/flops from the manifest) and hands it to the
   backend. Simulated results carry `simulated: true` and `numeric: "SKIPPED"`.
2. `DIAGNOSE` runs `stages/diagnose.py` over the profile plus static PTX features
   (`stages/features.py`), producing findings in fixed rule order.
3. `RESEARCH` maps findings to catalogue rows, filters already-applied/already-reverted
   transforms, and ranks by bottleneck class then severity.
4. `OPTIMIZE` asks `stages/optimize.py` for the child variant: the transform selects a
   ladder rung, the kernel source is re-rendered from the template, and an identical
   sha256 raises `NoOpVariant` so the loop never logs a fake improvement.
5. `VERIFY` lowers the variant (real PTX when the GPU job returned it, otherwise the
   template lowering), computes the reference bytes, emits the Lean module and runs
   Lean. `emit.check` classifies the result; `VERIFIED` needs a clean build, no
   `sorry`, and an axiom audit that printed `AUDIT_OK`.
6. `LEDGER` writes the round directory, commits, then regenerates the run's
   `REPORT.md`/`index.json`/`summary.json` and the ledger-wide index.

The SLA gate is a pure function (`orchestrator.sla_met`) over the latest profile and
proof, so it is testable without a GPU:

```python
checks = {
    "sol":      p["sol_pct"] >= cfg.sla.sol_pct,
    "speedup":  p["baseline"]["speedup"] is not None and p["baseline"]["speedup"] >= cfg.sla.speedup,
    "latency":  cfg.sla.latency_ms is None or p["p50_ms"] <= cfg.sla.latency_ms,
    "verified": (not cfg.sla.require_verified) or proof["status"] == "VERIFIED",
    "numeric":  p["numeric"] == "PASS",
    "measured": not p["simulated"],
}
```

## Instruction model (`harness/translator/`)

`ptx_forms.toml` is the contract between the parser, the Lean `Op` type and the
emitter: 81 rows, each with `op`, `quals`, `operands`, `semantics` and `lean_ctor`.
Parsing matches the longest dotted prefix (`ld.global.v4.f32` → form `ld`, qualifiers
`global v4 f32`); an unmatched opcode raises `UnsupportedInstruction` naming the
opcode and line, and never skips it silently.

Programs are held as a small instruction dict (`op`, `dst`, `src`, `pred`) with memory
operands as `base + tid·stride + (tid/rowSize)·rowStride + (tid%rowSize)·colStride + off`.
That single affine form covers 1-D elementwise addressing, row/column 2-D tiles, and
per-thread slices, which is what lets the model treat addresses symbolically.

`interp.py` is a reference interpreter of the same instruction model; the harness uses
it to compute expected bytes and static features. It is deliberately *not* the oracle
for correctness: Lean recomputes the program, and the numpy reference is compared
against the GPU output.

## Failure handling

| failure | behaviour |
|---|---|
| launch failure / compile error | round blocked, error event, no commit |
| `numeric: FAIL` | run blocked, no commit |
| no applicable transform | round reverted, stall counter increments |
| Lean build error | proof `FAILED`, open goals recorded, loop continues |
| `sorry` remaining | proof `PARTIAL`, open goals recorded, gate fails |
| unsupported opcode / dtype | proof `UNSUPPORTED` with the reason, gate fails |
| Modal infra error | retried twice, 30 s apart, then the run blocks |
| user stop | `STOPPED_BY_USER`, artifacts of finished rounds are kept |
