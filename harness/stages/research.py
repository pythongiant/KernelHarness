"""RESEARCH stage: turn diagnosed bottlenecks into ranked, cited hypotheses.

Catalog-only by default. When `HARNESS_LLM_BASE_URL` is set and answers, an
LLM proposal is *added* (never substituted) and constrained to known transform ids.
LLM output can never bypass profiling or verification.
"""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path

CATALOG_PATH = Path(__file__).with_name("catalog.toml")

KNOWN_TRANSFORMS = {
    "vectorize_x4", "streaming_stores", "grid_stride", "launch_bound", "reduce_smem",
    "mma_sync", "wgmma", "cp_async_double_buffer", "smem_tile", "swizzle_128b",
    "persistent_cta", "mbarrier_pipeline", "none",
}


def load_catalog(path: Path | None = None) -> list[dict]:
    data = tomllib.loads((path or CATALOG_PATH).read_text())
    return data.get("entry", [])


def _applied(variant) -> set[str]:
    applied = set()
    p = variant.params or {}
    if int(p.get("vector_width") or 1) >= 4:
        applied.add("vectorize_x4")
    if p.get("streaming"):
        applied.add("streaming_stores")
    if p.get("grid_stride"):
        applied.add("grid_stride")
    if p.get("launch_bound"):
        applied.add("launch_bound")
    if variant.level >= 2 and variant.family == "gemm":
        applied.add("mma_sync")
        applied.add("cp_async_double_buffer")
    if variant.level >= 3 and variant.family == "gemm":
        applied.update({"wgmma", "swizzle_128b"})
    return applied


def propose(bottlenecks: list[dict], profile: dict, kernel, variant,
            *, llm: bool | None = None, exclude_transforms: set[str] | None = None) -> list[dict]:
    """Rank candidate transforms for the diagnosed bottlenecks.

    `exclude_transforms` carries the transforms already attempted in this run: a
    reverted transform is not retried, which is what eventually stalls the loop
    instead of cycling.
    """
    catalog = load_catalog()
    applied = _applied(variant) | {t for t in (exclude_transforms or set()) if t}
    out: list[dict] = []

    for b in bottlenecks:
        code = b["code"]
        if code == "UNVERIFIED":
            out.append({
                "source": "catalog", "bottleneck_code": code, "transform_id": "none",
                "title": "discharge the open proof obligations",
                "rationale_md": (
                    "The variant's proof is not `VERIFIED`; the profile cannot be trusted "
                    "for further optimisation until the open goals are closed. Obligations: "
                    + (b["evidence_md"])),
                "refs": [], "expected_effect": "proof status reaches VERIFIED",
                "severity": b["severity"],
            })
            continue
        for entry in catalog:
            if entry["code"] != code:
                continue
            fam = entry.get("family")
            if fam and fam != kernel.family:
                continue
            if entry["transform"] in applied:
                continue
            out.append({
                "source": "catalog",
                "bottleneck_code": code,
                "transform_id": entry["transform"],
                "title": f"{entry['transform']} for {code}",
                "rationale_md": entry["rationale"].strip(),
                "refs": entry.get("refs", []),
                "expected_effect": entry["effect"],
                "severity": b["severity"],
                "level": entry.get("level", variant.level + 1),
            })
            break

    if llm is None:
        llm = bool(os.environ.get("HARNESS_LLM_BASE_URL"))
    if llm:
        out.extend(_llm_proposals(bottlenecks, profile, kernel, variant))

    # Rank by the bottleneck the transform addresses (roofline first), then severity.
    CODE_ORDER = {
        "MEM_BW_BOUND": 0, "COMPUTE_BOUND": 0, "NO_TENSOR_CORES": 1, "NO_ASYNC_COPY": 2,
        "UNCOALESCED": 3, "LOW_OCCUPANCY": 4, "HIGH_REGISTERS": 5, "SYNC_OVERHEAD": 6,
        "SMALL_PROBLEM": 7, "UNVERIFIED": 8, "BASELINE_UNAVAILABLE": 9,
    }
    sev = {"high": 0, "medium": 1, "low": 2}
    out.sort(key=lambda h: (CODE_ORDER.get(h.get("bottleneck_code"), 10),
                            sev.get(h.get("severity", "low"), 3)))
    return out


def _llm_proposals(bottlenecks, profile, kernel, variant) -> list[dict]:
    base = os.environ.get("HARNESS_LLM_BASE_URL")
    model = os.environ.get("HARNESS_LLM_MODEL", "qwen/qwen3-coder-30b")
    if not base:
        return []
    try:
        import httpx
        with httpx.Client(timeout=3.0) as client:
            models = client.get(f"{base.rstrip('/')}/models")
            models.raise_for_status()
            prompt = {
                "kernel": kernel.id,
                "family": kernel.family,
                "level": variant.level,
                "bottlenecks": [b["code"] for b in bottlenecks],
                "profile": {k: profile.get(k) for k in
                            ("sol_pct", "bound", "occupancy_pct", "regs_per_thread",
                             "smem_bytes", "p50_ms")},
                "allowed_transform_ids": sorted(KNOWN_TRANSFORMS),
            }
            req = {
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": ("Propose one optimisation. Reply with JSON "
                                '{"hypotheses":[{"transform_id":..., "rationale_md":..., '
                                '"refs":[...], "expected_effect":...}]} for: '
                                + json.dumps(prompt)),
                }],
                "temperature": 0.2,
            }
            r = client.post(f"{base.rstrip('/')}/chat/completions", json=req)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            start, end = text.find("{"), text.rfind("}")
            payload = json.loads(text[start:end + 1])
    except Exception:
        return []

    out = []
    for h in payload.get("hypotheses", []):
        tid = h.get("transform_id")
        if tid not in KNOWN_TRANSFORMS:
            continue  # unknown transforms are rejected outright
        out.append({
            "source": f"llm:{model}",
            "bottleneck_code": (bottlenecks[0]["code"] if bottlenecks else "UNKNOWN"),
            "transform_id": tid,
            "title": f"{tid} (llm proposal)",
            "rationale_md": str(h.get("rationale_md", ""))[:4000],
            "refs": [str(r) for r in h.get("refs", [])][:8],
            "expected_effect": str(h.get("expected_effect", ""))[:500],
            "severity": "medium",
        })
    return out
