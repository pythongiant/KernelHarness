# Examples

| example | shows |
|---|---|
| [`quickstart.sh`](quickstart.sh) | the whole loop offline (sim backend), the ledger layout, a proof check, and how to start the UI |
| [`transcript.md`](transcript.md) | one real H100 round, captured verbatim: every stage's reasoning and the gate output |
| [`custom_kernel/`](custom_kernel/) | registering a kernel you did not write (`family = "user"`), including the manifest fields and what the harness does (and does not) promise for it |
| [`../ledgers/`](../ledgers/) | the committed output of real runs: `run_69cbe7246706` and its repeat `run_87196cd133e6` (full ladder, `SLA_MET`), `run_e18357827eae` (`matmul_tiled`, `UNSUPPORTED` proof), `run_57be3f4ba2da` (`attention_scalar`, `UNSUPPORTED` proof), `run_50f5cafca7ed` (`custom_affine_fast`, user kernel), `run_1181f4db4eb2` (`--backend sim`, gate refuses) |

```bash
bash examples/quickstart.sh
```

Outputs are small: a round directory is a few kB (profile JSON, Markdown, source, PTX,
diff, Lean module, build log).
