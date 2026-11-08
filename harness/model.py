"""Variant model: a kernel implementation at a level with applied transforms."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .registry import Kernel, geometry, render_source


@dataclass
class Variant:
    id: str
    run_id: str
    round_n: int
    family: str
    level: int
    transform_id: str | None
    params: dict
    kernel: Kernel
    source: str
    sha256: str
    parent_id: str | None = None
    ptx: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def geom(self) -> dict[str, int]:
        return geometry(self.kernel)

    @classmethod
    def build(cls, *, variant_id: str, run_id: str, round_n: int, kernel: Kernel,
              level: int, transform_id: str | None, params: dict,
              parent_id: str | None = None) -> "Variant":
        source = render_source(kernel, level, params)
        sha = hashlib.sha256(source.encode()).hexdigest()
        return cls(id=variant_id, run_id=run_id, round_n=round_n, family=kernel.family,
                   level=level, transform_id=transform_id, params=params, kernel=kernel,
                   source=source, sha256=sha, parent_id=parent_id)


def digest(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()
