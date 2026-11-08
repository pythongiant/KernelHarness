# Run `run_1181f4db4eb2`

Kernel `elementwise_affine` · GPU `H100` · backend `sim` · status **max_rounds** (MAX_ROUNDS)

Each row is one committed round; `proof` is the Lean verification status and `commit` its git sha.

| round | transform | sol % | p50 ms | proof | commit |
|---|---|---|---|---|---|
| 0 | `baseline` | 10.0 | 0.100 | VERIFIED | `9289af2b0335` |
| 1 | `vectorize_x4` | 34.0 | 0.029 | VERIFIED | `1b5d168f1142` |
| 2 | `streaming_stores` | 71.0 | 0.014 | VERIFIED | `d5bd025a170d` |
