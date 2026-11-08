# Contributing

## Setup

```bash
uv sync --extra dev
cd lean && lake build && cd ..      # needs elan + Lean 4.35
cd web && bun install && bun run build && cd ..
```

## Before you send a change

```bash
uv run pytest                                   # 22 tests, no GPU/Lean needed
(cd lean && lake build && lake build Harness.Audit)   # prints AUDIT_OK
(cd web && bun run build)
```

If you touch the model or the emitter, also run one end-to-end proof:

```bash
uv run harness run --kernel elementwise_affine --backend sim --max-rounds 1
uv run harness verify --variant <variant_id>    # must print status VERIFIED
```

## Rules of the project

1. **Never fake a number.** Anything not measured carries `simulated: true`; anything
   not proved carries `PARTIAL`/`FAILED`/`UNSUPPORTED` with a reason. The SLA gate is
   not allowed to pass on either.
2. **Never add an axiom or a `sorry`.** `Harness.Audit` fails the build on unexpected
   axioms; `emit.check` downgrades a module containing `sorry` to `PARTIAL`. If a
   statement is too strong, weaken it and say why in the commit message.
3. **Evidence-first diagnostics.** A bottleneck rule must quote the metric that fired
   it (`evidence_md`), and a hypothesis must cite a reference.
4. **One instruction model.** New PTX forms go into `harness/translator/ptx_forms.toml`
   *and* `lean/Harness/PTX/Instr.lean` (same `lean_ctor`), with a parser fixture.
5. **Disclose scope.** Anything the model cannot express (floats, intra-launch
   producer/consumer, unbounded shapes) is reported as a limit, not worked around.

## Adding a kernel

1. `harness/kernels/<id>/kernel.toml`, see `examples/custom_kernel/kernel.toml`.
2. `harness/templates/<family>/L<n>/kernel.cu`, one file per ladder rung, using
   `@@PLACEHOLDER@@` tokens only (CUDA braces are preserved; no `str.format`).
3. If the family needs verification, add a lowering in
   `harness/translator/lower.py` and a reference in `harness/runner/reference.py`.
4. `uv run harness registry` must show the new shape/SOL resolved.

## Adding a transform

1. A catalogue row in `harness/stages/catalog.toml` (bottleneck, transform id, effect,
   rationale, refs, target rung).
2. A `TRANSFORMS` entry in `harness/stages/optimize.py` (target ladder level, params).
3. A template rung that actually implements it, because the optimizer refuses byte-identical
   output, so a transform without a rung shows up as a no-op immediately.

## Adding a PTX form

1. Row in `ptx_forms.toml` (`op`, `quals`, `operands`, `semantics`, `lean_ctor`).
2. Matching constructor in `lean/Harness/PTX/Instr.lean`.
3. Semantics in `lean/Harness/PTX/Thread.lean` (which event(s) it emits) and in
   `harness/translator/interp.py` (for features and reference bytes).
4. Fixture under `tests/fixtures/` plus a parse test.

## Style

- Python: stdlib + the declared dependencies only; no new runtime dependency without
  a reason in the PR description.
- Lean: byte-level statements, `decide` on concrete traces, no `native_decide` (it adds
  an axiom).
- TypeScript: no state library; the UI renders DB rows and invents nothing.
