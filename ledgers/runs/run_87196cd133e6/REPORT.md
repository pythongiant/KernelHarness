# Run `run_87196cd133e6`

Kernel `elementwise_affine` · GPU `H100` · backend `modal` · status **sla_met** (SLA_MET)

Each row is one committed round; `proof` is the Lean verification status and `commit` its git sha.

| round | transform | sol % | p50 ms | proof | commit |
|---|---|---|---|---|---|
| 0 | `baseline` | 64.8 | 0.015 | VERIFIED | `c01cf27dc561` |
| 1 | `vectorize_x4` | 75.6 | 0.013 | VERIFIED | `5f35488d1d62` |
| 2 | `streaming_stores` | 93.7 | 0.011 | VERIFIED | `fbbfb6c1f7bd` |
