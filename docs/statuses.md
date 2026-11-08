# Statuses, stop reasons and verdicts

The harness uses a closed vocabulary everywhere (DB, API, UI, tests).

## Run status (`runs.status`)

| status | meaning |
|---|---|
| `running` | the loop is active |
| `sla_met` | every gate passed: SOL, speedup, latency (if set), `VERIFIED` proof, `numeric: PASS`, measured numbers |
| `max_rounds` | round cap reached with the gate still failing |
| `stalled` | `stall_rounds` consecutive rounds did not improve `p50` |
| `blocked` | a hard stop: launch/compile failure, `numeric: FAIL`, no applicable transform, or an uncaught stage error |
| `stopped` | the operator stopped the run |

## Stop reasons (`runs.stop_reason`)

`SLA_MET` · `MAX_ROUNDS` · `STALLED` · `BLOCKED` · `STOPPED_BY_USER`, the
machine-readable counterpart of the status above.

## Round verdicts (`rounds.status`)

| verdict | meaning |
|---|---|
| `baseline` | round 0, the unmodified kernel, committed as the starting point |
| `improved` | the candidate beat the parent's `p50` or SOL by ≥ `tolerance_pct`, **or** satisfied the SLA gate, and became the new parent |
| `reverted` | the candidate was committed but did not beat the parent; the next round starts from the parent |
| `blocked` | the round produced no usable variant |

## Proof status (`proofs.status`)

| status | checks that passed | gate effect |
|---|---|---|
| `VERIFIED` | Lean build clean **and** no `sorry` **and** axiom audit `AUDIT_OK` | satisfies `verified` |
| `PARTIAL` | nothing failed outright, but goals/`sorry` remain | fails `verified`; open goals listed |
| `FAILED` | Lean rejected the module | fails `verified`; errors listed |
| `UNSUPPORTED` | the variant is outside the model's scope (dtype, transcendental, missing toolchain) | fails `verified`; reason listed |

Open goals, the module path and the audit (`{"ok": true, "allowed": [...], "checked": [...]}`)
are stored with the proof and shown on `#/proof/<variant>`.

## Bottleneck codes (`bottlenecks.code`)

| code | rule (in evaluation order) | typical fix |
|---|---|---|
| `LOW_OCCUPANCY` | `occupancy_pct < 50` | `launch_bound`, `reduce_smem` |
| `MEM_BW_BOUND` | roofline says memory-bound; high if `sol_pct < 70` | `vectorize_x4`, `streaming_stores`, `grid_stride` |
| `COMPUTE_BOUND` | roofline says compute-bound; high if `sol_pct < 70` | tensor-core rungs |
| `UNCOALESCED` | `ncu.sectors_per_request > 1.5`, or static stride > 1 | `smem_tile`, `swizzle_128b` |
| `NO_TENSOR_CORES` | gemm/attention with no `mma`/`wgmma`/`tcgen05` | `mma_sync`, `wgmma` |
| `NO_ASYNC_COPY` | `smem_bytes > 0` and no `cp.async*` | `cp_async_double_buffer` |
| `SMALL_PROBLEM` | `grid < 2 × sm_count` | `persistent_cta` |
| `HIGH_REGISTERS` | `regs_per_thread > 96` | `launch_bound` |
| `SYNC_OVERHEAD` | `ncu` barrier-stall share > 10 % | `mbarrier_pipeline` |
| `UNVERIFIED` | latest proof is not `VERIFIED` | close the obligations first |
| `BASELINE_UNAVAILABLE` | no torch baseline / no speedup | speedup gate fails closed |

## Event kinds (`events.kind`)

`stage_start` · `stage_end` · `reasoning` · `measurement` · `decision` · `artifact` ·
`error` · `log`. The SSE feed maps these to `stage`, `round`, `log`, `error` frames and
replays the backlog on connect.

## Profile fields worth reading first

`sol_pct` (roofline), `bound` (`memory`/`compute`), `p50_ms`, `eff_bw_gbps`/`tflops`,
`numeric` (`PASS`/`FAIL`/`SKIPPED`), `simulated`, `regs_per_thread`, `smem_bytes`,
`occupancy_pct` + `occupancy_limiter`, `counters`/`ncu`, `baseline.speedup`.
