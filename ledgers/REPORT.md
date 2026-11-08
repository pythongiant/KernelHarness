# Kernel discovery ledger

Every round of every harness run is committed here: the measured profile, the diagnosis, the ranked hypotheses, the kernel source and PTX, the diff, and the Lean verification status with its module. Numbers come from a real GPU (`backend: modal`) or from the deterministic simulator (`backend: sim`, which can never satisfy the SLA).

| run | kernel | backend | status | rounds | verified | best SOL % | report |
|---|---|---|---|---|---|---|---|
| `run_1181f4db4eb2` | elementwise_affine | sim | max_rounds / MAX_ROUNDS | 3 | 3 | 71.0 | [run_1181f4db4eb2](runs/run_1181f4db4eb2/REPORT.md) |
| `run_50f5cafca7ed` | custom_affine_fast | modal | stalled / STALLED | 1 | 0 | 59.6 | [run_50f5cafca7ed](runs/run_50f5cafca7ed/REPORT.md) |
| `run_57be3f4ba2da` | attention_scalar | modal | max_rounds / MAX_ROUNDS | 2 | 0 | 0.1 | [run_57be3f4ba2da](runs/run_57be3f4ba2da/REPORT.md) |
| `run_69cbe7246706` | elementwise_affine | modal | sla_met / SLA_MET | 3 | 3 | 94.3 | [run_69cbe7246706](runs/run_69cbe7246706/REPORT.md) |
| `run_87196cd133e6` | elementwise_affine | modal | sla_met / SLA_MET | 3 | 3 | 93.7 | [run_87196cd133e6](runs/run_87196cd133e6/REPORT.md) |
| `run_e18357827eae` | matmul_tiled | modal | stalled / STALLED | 1 | 0 | 9.1 | [run_e18357827eae](runs/run_e18357827eae/REPORT.md) |
