"""Static instruction-mix features extracted from a variant's PTX."""

from __future__ import annotations

import re

_STRIDE_RE = re.compile(r"stride[=: ]\s*(\d+)")


def features(ptx: str) -> dict:
    """Summarize an instruction listing."""
    mix: dict[str, int] = {}
    vector_widths: set[int] = set()
    barriers = 0
    smem = 0
    max_stride = 1

    for raw in ptx.splitlines():
        line = raw.split("//")[0].strip().rstrip(";")
        if not line or line.startswith((".", "{", "}")):
            continue
        opcode = line.split()[0].lstrip("@").split(".")[0]
        mix[opcode] = mix.get(opcode, 0) + 1
        if opcode in ("ld", "st", "ldmatrix"):
            if ".v4" in line:
                vector_widths.add(4)
            elif ".v2" in line:
                vector_widths.add(2)
            else:
                vector_widths.add(1)
            m = _STRIDE_RE.search(line)
            if m:
                max_stride = max(max_stride, int(m.group(1)))
        if opcode in ("bar", "membar", "fence"):
            barriers += 1
        if ".shared" in line and opcode in ("ld", "st", "cp"):
            smem += 4

    has = lambda pred: any(pred(k) for k in mix)  # noqa: E731
    return {
        "instr_mix": mix,
        "vector_widths": sorted(vector_widths) or [1],
        "barriers": barriers,
        "smem_bytes": smem,
        "has_async": has(lambda k: k == "cp"),
        "has_mma": "mma" in mix,
        "has_wgmma": "wgmma" in mix,
        "has_tcgen05": "tcgen05" in mix,
        "max_stride": max_stride,
        "instruction_count": sum(mix.values()),
    }
