# Contributing

## Setup and checks

```bash
uv sync --extra dev
uv run ruff check . && uv run ruff format --check .   # lint + format
uv run pytest                                          # no GPU or Lean needed
(cd lean && lake build && lake build Harness.Audit)    # prints AUDIT_OK
(cd web && bun install && bun run build)
```

CI runs all of the above. If you touch the Lean model or the emitter, also check one
end-to-end proof: `uv run harness run --kernel elementwise_affine --backend sim --max-rounds 1`
should report `proof VERIFIED` for every round.

## Ground rules

1. **Never fake a number.** Anything not measured carries `simulated: true`; anything
   not proved is `PARTIAL`, `FAILED` or `UNSUPPORTED` with a reason. The target check
   must not pass on either.
2. **No new axioms, no `sorry`, no `native_decide`.** `Harness.Audit` fails the build on
   unexpected axioms. If a statement is too strong, weaken it and say why.
3. **Evidence first.** A bottleneck rule must quote the metric that fired it, and a
   catalogue entry must cite a reference.
4. **One instruction model.** `ptx_forms.toml` and `lean/Harness/PTX/Instr.lean` must agree.
5. **No new runtime dependency** without a reason in the PR.

## Adding a kernel

1. `harness/kernels/<id>/kernel.toml` (see `examples/custom_kernel/kernel.toml` for the fields).
2. For a new family, one `harness/templates/<family>/L<n>/kernel.cu` per ladder rung,
   using `@@PLACEHOLDER@@` tokens.
3. For proofs, a lowering in `harness/translator/lower.py` and a reference in
   `harness/runner/reference.py`.
4. Check that `uv run harness registry` resolves the shape and SOL.

## Adding a fix

1. A row in `harness/stages/catalog.toml` (bottleneck, transform id, rationale, refs).
2. An entry in `TRANSFORMS` in `harness/stages/optimize.py` (target rung, parameters).
3. A template rung that implements it. A fix with no rung produces identical source,
   and the optimiser refuses it.

## Adding a PTX form

1. A row in `harness/translator/ptx_forms.toml`.
2. The matching constructor in `lean/Harness/PTX/Instr.lean` and its semantics in
   `lean/Harness/PTX/Thread.lean`.
3. Semantics in `harness/translator/interp.py`.
4. A fixture in `tests/fixtures/` and a parse test.
