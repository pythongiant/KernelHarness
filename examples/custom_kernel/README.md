# Custom kernel example

Register a kernel you did not write in the harness's vocabulary.

```bash
cp -r examples/custom_kernel harness/kernels/custom_affine_fast
uv run harness registry | grep -A2 custom_affine_fast
uv run harness run --kernel custom_affine_fast --gpu H100 --max-rounds 2
```

What happens:

1. `PROFILE` compiles `kernel.cu` inside the Modal image, times it, and compares the
   output buffer against the `elementwise` reference (declared in `kernel.toml` with
   `reference = "elementwise"`, because `family = "user"` has no built-in reference).
2. `DIAGNOSE` reports the roofline findings like any other kernel, because the bottleneck
   rules only read the profile, not the family.
3. `RESEARCH`/`OPTIMIZE` find no ladder for `family = "user"` and emit
   `NO_TRANSFORM_AVAILABLE`; the round and the baseline are still committed, so the
   measurement is part of the track record.
4. `VERIFY` reports `UNSUPPORTED` with a reason: the model needs a lowering, and this
   family has none. Supply `--kernel` a kernel from a laddered family (or add a
   lowering in `harness/translator/lower.py`) if you need a machine-checked proof.

Manifest fields worth knowing: `shape` (evaluated by `harness/launch.py`'s whitelisted
expression evaluator), `sol` (the roofline numerator: `bytes` and `flops`), `buffers`
(order must match `p.buf[i]`), `out_index`, `tol` (exact for `s32`, `rtol`/`atol`
otherwise), `baseline` (`torch` or `none`).
