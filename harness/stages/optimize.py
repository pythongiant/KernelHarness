"""OPTIMIZE stage: materialize the next variant and refuse no-op edits."""

from __future__ import annotations

import copy

from ..model import Variant

#: transform id -> (minimum ladder level, parameter updates). The ladder index is
#: the source of truth: a transform never moves a variant below its current level, so
#: two transforms that land on the same level are one edit of the kernel.
TRANSFORMS: dict[str, tuple[int, dict]] = {
    "vectorize_x4": (1, {"vector_width": 4}),
    "smem_tile": (1, {}),
    "reduce_smem": (2, {}),
    "launch_bound": (2, {"vector_width": 4, "launch_bound": True}),
    "mma_sync": (2, {}),
    "cp_async_double_buffer": (2, {}),
    "streaming_stores": (3, {"vector_width": 4, "streaming": True}),
    "grid_stride": (3, {"vector_width": 4, "streaming": True, "grid_stride": True}),
    "swizzle_128b": (3, {}),
    "persistent_cta": (3, {}),
    "mbarrier_pipeline": (3, {}),
    "wgmma": (3, {}),
    "none": (0, {}),
}


class NoOpVariant(Exception):
    """Raised when the optimized source is byte-identical to its parent."""


def apply_transform(variant: Variant, hypothesis: dict, round_n: int | None = None) -> Variant:
    """Build the child variant for a hypothesis, or raise `NoOpVariant`."""
    tid = hypothesis.get("transform_id")
    if tid not in TRANSFORMS:
        raise NoOpVariant(f"unknown transform: {tid}")
    target_level, params = TRANSFORMS[tid]
    levels = variant.kernel.levels()
    level = max(variant.level, target_level)
    if levels:
        level = min(level, max(levels))
    new_params = {**copy.deepcopy(variant.params), **params}
    if (
        level >= 2
        and variant.kernel.family == "gemm"
        and variant.kernel.dtype not in ("f16", "bf16")
    ):
        raise NoOpVariant(
            f"level {level} for `{variant.kernel.id}` needs half-precision operands for the "
            f"tensor-core/cp.async staging; kernel dtype is `{variant.kernel.dtype}`"
        )
    child = Variant.build(
        variant_id=f"{variant.id}-{tid}-r{round_n if round_n is not None else variant.round_n}",
        run_id=variant.run_id,
        round_n=variant.round_n,
        kernel=variant.kernel,
        level=level,
        transform_id=tid,
        params=new_params,
        parent_id=variant.id,
    )
    if child.sha256 == variant.sha256:
        raise NoOpVariant(
            f"transform {tid} produced an identical source (level {level}, params {new_params})"
        )
    child.meta["hypothesis"] = hypothesis
    return child
