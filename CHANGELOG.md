# Changelog

## 0.1.0 (2026-11-07)

First working closed loop.

**Harness**

- Loop `PROFILE → DIAGNOSE → RESEARCH → OPTIMIZE → VERIFY → LEDGER` with an explicit
  SLA gate (SOL, speedup, latency, verified proof, numeric, measured) and stop reasons
  `SLA_MET`, `MAX_ROUNDS`, `STALLED`, `BLOCKED`, `STOPPED_BY_USER`.
- Per-round git ledger: profile, diagnosis, hypotheses, kernel source + PTX, diff,
  proof status/module/log; browsable run index (`ledgers/REPORT.md`).
- Deterministic simulator backend plus a Modal backend (H100/H200/B200/A100/L4) with
  device probe, `ncu` detection, torch baseline, and numeric comparison against a
  bit-identical numpy reference.
- Kernel ladder: elementwise L0–L3 (`s32`), gemm L0–L3 (`s32` and `f32`/`wmma`),
  attention L0–L1 (`f32`), plus user-supplied kernels.
- Research catalogue with cited rationales and an optional OpenAI-compatible LLM that
  can only *add* proposals.

**Model (`lean/Harness`)**

- 81 PTX forms; small-step machine with derived event traces; the eight consistency
  axioms as executable checkers; observer reads, `apiSync`, multi-kernel composition.
- `tcgen05` (tensor memory, mbarrier completion) and cluster/DSMEM (`mapa`, cluster
  barriers, launch-control cancellation) extensions.
- `nonvacuous` and `correct` proved; `correct_of_canonical` is the per-variant bridge;
  axiom audit restricted to `propext`, `Classical.choice`, `Quot.sound`.
- Sanity proofs for the paper's barrier example and the initialization-sourced
  observer read, both rejected/accepted by evaluation.

**Surfaces**

- FastAPI API + SSE stream with backlog replay; Vite/React UI with run control, round
  narrative, proof viewer, provenance graph, event log.
- CLI: `init`, `registry`, `run`, `verify`, `profile`, `serve`, `report`.
- 22 tests covering the expression whitelist, PTX parser, diagnosis rules, research
  ranking, the simulated loop (commits, monotonicity, blocking), and the HTTP/SSE API.

**Measured (H100 80 GB HBM3)**

- `elementwise_affine` 1 M `s32`: 46.8 % → 56.4 % of peak bandwidth (0.0082 → 0.0069 ms),
  2.4× the torch baseline, proofs `VERIFIED` for every round, `SLA_MET` at a 55 % gate.
- `matmul_tiled` (f32 tensor cores) and `attention_scalar` (softmax) profile with
  `numeric: PASS` and correctly report `UNSUPPORTED` proofs.

**Known limits (documented, enforced)**

- The step machine is thread-local; intra-launch producer/consumer kernels are rejected
  rather than accepted unsoundly.
- Numerical scope is exact integer arithmetic; IEEE rounding is not modelled.
- Proofs are about a concrete small launch instance (`instantiation` records it).
