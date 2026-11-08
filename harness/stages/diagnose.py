"""DIAGNOSE stage: evidence-based bottleneck classification.

Rules are evaluated in a fixed order; each emits `evidence_md` quoting the metric
that triggered it, so the UI never shows a conclusion without its evidence.
"""

from __future__ import annotations

SEVERITIES = ("high", "medium", "low")


def _fmt(v, unit="") -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.2f}{unit}"
    return f"{v}{unit}"


def diagnose(profile: dict, variant, kernel, proof: dict | None = None,
             feats: dict | None = None) -> list[dict]:
    """Return a list of `{code, severity, evidence_md}` in reporting order."""
    out: list[dict] = []
    feats = feats or {}
    sol = float(profile.get("sol_pct") or 0.0)
    bound = profile.get("bound")
    occ = float(profile.get("occupancy_pct") or 0.0)
    regs = int(profile.get("regs_per_thread") or 0)
    smem = int(profile.get("smem_bytes") or 0)
    grid = int(profile.get("grid") or 1)
    sm_count = 132
    mix = feats.get("instr_mix", {})
    ncu = profile.get("ncu") or {}

    if occ < 50:
        out.append({
            "code": "LOW_OCCUPANCY", "severity": "high",
            "evidence_md": (
                f"`occupancy_pct = {occ:.1f}%` is below the 50% threshold; "
                f"the profiler names `{profile.get('occupancy_limiter')}` as the limiter "
                f"({regs} registers/thread, {smem} B shared memory/block).")})

    if bound == "memory":
        out.append({
            "code": "MEM_BW_BOUND", "severity": "high" if sol < 70 else "low",
            "evidence_md": (
                f"Roofline says memory-bound at `{_fmt(sol, '%')}` of peak: "
                f"`eff_bw_gbps = {_fmt(profile.get('eff_bw_gbps'))}` of "
                f"`{_fmt(profile.get('peak_bw_gbps'))}` GB/s, moving "
                f"`{profile.get('bytes_moved')}` bytes in `{_fmt(profile.get('p50_ms'), ' ms')}`.")})
    elif bound == "compute":
        out.append({
            "code": "COMPUTE_BOUND", "severity": "high" if sol < 70 else "low",
            "evidence_md": (
                f"Roofline says compute-bound at `{_fmt(sol, '%')}` of peak: "
                f"`tflops = {_fmt(profile.get('tflops'))}` of "
                f"`{_fmt(profile.get('peak_tflops'))}` TFLOPs for "
                f"`{profile.get('flops')}` FLOPs.")})

    sectors = ncu.get("sectors_per_request")
    stride = max(feats.get("max_stride", 1) or 1, 1)
    if (sectors is not None and float(sectors) > 1.5) or stride > 1:
        detail = (f"`ncu.sectors_per_request = {float(sectors):.2f}` > 1.5" if sectors
                  else f"static analysis: hot `ld`/`st` address stride is {stride} elements")
        out.append({"code": "UNCOALESCED", "severity": "high", "evidence_md":
                    f"Uncoalesced access detected ({detail})."})

    if kernel.family in ("gemm", "attention"):
        if not (feats.get("has_mma") or feats.get("has_wgmma") or feats.get("has_tcgen05")):
            out.append({
                "code": "NO_TENSOR_CORES", "severity": "high",
                "evidence_md": (
                    f"Family `{kernel.family}` but the instruction mix has no tensor-core "
                    f"MMA: `{_sorted_mix(mix)}`.")})

    if smem > 0 and not feats.get("has_async"):
        out.append({
            "code": "NO_ASYNC_COPY", "severity": "medium",
            "evidence_md": (
                f"`smem_bytes = {smem}` but the instruction mix has no `cp.async*`: "
                f"global→shared traffic is not overlapped with compute.")})

    if grid < 2 * sm_count:
        out.append({
            "code": "SMALL_PROBLEM", "severity": "medium",
            "evidence_md": (
                f"`grid.x*grid.y = {grid}` is below `2 * sm_count = {2 * sm_count}`: "
                f"launch/tail overhead dominates.")})

    if regs > 96:
        out.append({
            "code": "HIGH_REGISTERS", "severity": "medium",
            "evidence_md": f"`regs_per_thread = {regs}` exceeds 96."})

    barrier_stall = ncu.get("barrier_stall_pct")
    if barrier_stall is not None and float(barrier_stall) > 10.0:
        out.append({
            "code": "SYNC_OVERHEAD", "severity": "medium",
            "evidence_md": f"`ncu` barrier-stall share `{float(barrier_stall):.1f}%` > 10%."})

    if proof is not None and proof.get("status") != "VERIFIED":
        out.append({
            "code": "UNVERIFIED", "severity": "high",
            "evidence_md": (
                f"Latest proof status is `{proof.get('status')}`; open goals: "
                f"{proof.get('open_goals') or 'none listed'}.")})

    baseline = profile.get("baseline") or {}
    if baseline.get("name") in (None, "none") or baseline.get("speedup") is None:
        out.append({
            "code": "BASELINE_UNAVAILABLE", "severity": "low",
            "evidence_md": (
                f"`baseline.name = {baseline.get('name')}` and `speedup = "
                f"{baseline.get('speedup')}`: the speedup gate fails closed.")})

    return out


def _sorted_mix(mix: dict) -> str:
    if not mix:
        return "(empty)"
    return ", ".join(f"{k}×{v}" for k, v in sorted(mix.items(), key=lambda kv: -kv[1])[:8])
