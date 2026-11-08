# Run `run_69cbe7246706`

Kernel `elementwise_affine` · GPU `H100` · backend `modal` · status **sla_met** (SLA_MET)

Each row is one committed round; `proof` is the Lean verification status and `commit` its git sha.

| round | transform | sol % | p50 ms | proof | commit |
|---|---|---|---|---|---|
| 0 | `baseline` | 63.6 | 0.016 | VERIFIED | `9fe3c5a937e8` |
| 1 | `vectorize_x4` | 82.8 | 0.012 | VERIFIED | `a38b60db1a99` |
| 2 | `streaming_stores` | 94.3 | 0.011 | VERIFIED | `8cc786565bee` |
