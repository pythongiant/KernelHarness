"""VERIFY stage: emit Lean for a variant and classify the proof.

Flow: lower the template to a concrete instruction program over a small launch
instance → build the initial byte map and the reference output bytes (numpy) →
emit `lean/Harness/Generated/<Variant>.lean` → run Lean → classify.

The three checks are independent and all required for `VERIFIED`:
1. `lake env lean` accepts the module (no errors, no `sorry`),
2. the axiom audit ran and reported `AUDIT_OK`,
3. the module's obligations (`canonConsistent`, `canonPost`) were discharged —
   otherwise the status stays `PARTIAL` and the open goals are reported.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import LEAN_DIR
from ..model import Variant
from ..runner import reference
from ..translator import emit, lower

LEAN_N = 8
LEAN_GEMM = 4


class VerifyError(Exception):
    pass


def _s32_bytes(values) -> bytes:
    return np.asarray(values, dtype="<i4").tobytes()


def build_instance(variant: Variant) -> dict:
    """Return program, geometry, initial bytes and reference output for the
    verification instance of a variant."""
    kernel = variant.kernel
    family = variant.family
    if family == "elementwise":
        n = LEAN_N
        program, geom = lower.lower_elementwise(
            {**variant.params, "lean_n": n}, lean_n=n,
            a=int(kernel.shape_env().get("a", 10)), b=int(kernel.shape_env().get("b", 3)))
        a = int(kernel.shape_env().get("a", 10))
        b = int(kernel.shape_env().get("b", 3))
        x = reference.fill_s32(n)
        raw = _s32_bytes(x)
        init = {lower.IN_BASE + i: raw[i] for i in range(len(raw))}
        out = (np.int32(a) * x + np.int32(b)).astype(np.int32)
        return {
            "program": program, "geom": geom, "init": init,
            "out_base": lower.OUT_BASE, "out_bytes": list(_s32_bytes(out)),
            "instance": f"elementwise N={n}, a={a}, b={b}",
        }
    if family == "gemm":
        m = n = k = LEAN_GEMM
        program, geom = lower.lower_matmul_scalar(variant.params, m=m, n=n, k=k)
        A = reference.fill_s32(m * k).reshape(m, k).astype(np.int32)
        B = reference.fill_s32(k * n, seed=reference.SEED ^ 0x5A5A).reshape(k, n).astype(np.int32)
        rawA, rawB = _s32_bytes(A), _s32_bytes(B)
        init = {lower.A_BASE + i: rawA[i] for i in range(len(rawA))}
        init.update({lower.B_BASE + i: rawB[i] for i in range(len(rawB))})
        C = (A @ B).astype(np.int32)
        return {
            "program": program, "geom": geom, "init": init,
            "out_base": lower.C_BASE, "out_bytes": list(_s32_bytes(C)),
            "instance": f"gemm M=K=N={m}",
        }
    raise NotImplementedError(f"no verification instance for family {family}")


def verify(variant: Variant, *, log_dir: Path | None = None) -> emit.ProofResult:
    """Verify a variant against the Lean model."""
    if variant.kernel.dtype not in ("s32", "i32"):
        return emit.ProofResult(
            status="UNSUPPORTED", model="paper-full",
            open_goals=[
                "UNSUPPORTED: the model reasons about bit patterns and exact integer "
                f"arithmetic; dtype `{variant.kernel.dtype}` needs IEEE rounding, which is "
                "outside its numerical scope (see numerical_scope)."],
            axiom_audit={"ok": False, "run": False}, modules={}, instantiation="none")
    if variant.family in ("attention",):
        prelude = (
            "softmax uses expf, which the model does not provide: the PTX semantics "
            "covers bit patterns and exact integer arithmetic, not transcendentals"
        )
        return emit.ProofResult(
            status="UNSUPPORTED", model="paper-full",
            open_goals=[f"UNSUPPORTED: {prelude}"],
            axiom_audit={"ok": False, "run": False},
            modules={}, instantiation="none",
        )
    try:
        inst = build_instance(variant)
    except NotImplementedError as exc:
        return emit.ProofResult(
            status="UNSUPPORTED", model="paper-full",
            open_goals=[f"UNSUPPORTED: {exc}"],
            axiom_audit={"ok": False, "run": False}, modules={}, instantiation="none")

    name, source = emit.emit_lean(
        variant_id=variant.id,
        family=variant.family,
        program=inst["program"],
        geom=inst["geom"],
        init=inst["init"],
        out_base=inst["out_base"],
        out_bytes=inst["out_bytes"],
        meta={"level": variant.level, "transform": variant.transform_id},
    )
    result = emit.check(name, source, log_dir=log_dir or (LEAN_DIR / ".proof-logs"))
    result.instantiation = inst["instance"]
    return result
