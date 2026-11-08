"""FastAPI surface: run control, run snapshots, SSE stream, artifact endpoints."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import WEB_DIR, RunConfig, Sla
from .db import (connect, create_run, init_db, latest_profile, latest_proof, list_runs,
                 run_snapshot)
from .orchestrator import run_loop
from .registry import geometry, registry
from .stages import ledger as ledger_stage

app = FastAPI(title="kernel-harness")
_conn = None
_tasks: dict[str, asyncio.Task] = {}
_cancels: dict[str, asyncio.Event] = {}
_subscribers: dict[str, list[asyncio.Queue]] = {}


def conn():
    global _conn
    if _conn is None:
        _conn = init_db()
    return _conn


class RunRequest(BaseModel):
    kernel_id: str
    gpu: str = "H100"
    backend: str = "modal"
    sla: dict | None = None


@app.get("/api/registry")
def api_registry():
    out = []
    for k in registry():
        env = k.shape_env()
        out.append({
            "id": k.id, "family": k.family, "dtype": k.dtype, "level": k.level,
            "levels": k.levels(), "shape": k.shape, "shape_resolved": env,
            "sol": {"kind": k.sol.kind, "bytes": k.sol.bytes, "flops": k.sol.flops,
                    "bytes_resolved": k.sol_numbers(env)[0],
                    "flops_resolved": k.sol_numbers(env)[1]},
            "geometry": geometry(k), "baseline": k.baseline, "spec_lean": k.spec_lean,
            "tol": k.tol, "buffers": k.buffers, "out_index": k.out_index,
        })
    return {"kernels": out}


@app.post("/api/runs")
async def api_start(req: RunRequest):
    cfg = RunConfig(kernel_id=req.kernel_id, gpu=req.gpu, backend=req.backend,
                    sla=Sla(**(req.sla or {})))
    c = conn()
    run_id = create_run(c, cfg)
    cancel = asyncio.Event()
    _cancels[run_id] = cancel
    _subscribers[run_id] = []

    async def emit(stage, kind, title, body_md, payload):
        for q in list(_subscribers.get(run_id, [])):
            await q.put({"stage": stage, "kind": kind, "title": title,
                         "body_md": body_md, "payload": payload})

    async def runner():
        try:
            await run_loop(connect(), cfg, run_id, emit=emit, cancel_evt=cancel)
        finally:
            for q in list(_subscribers.get(run_id, [])):
                await q.put(None)

    _tasks[run_id] = asyncio.create_task(runner())
    return {"run_id": run_id}


@app.get("/api/runs")
def api_runs():
    return {"runs": list_runs(conn())}


@app.get("/api/runs/{run_id}")
def api_run(run_id: str):
    snap = run_snapshot(conn(), run_id)
    if snap is None:
        raise HTTPException(404, "unknown run")
    return snap


@app.post("/api/runs/{run_id}/stop")
def api_stop(run_id: str):
    ev = _cancels.get(run_id)
    if ev is None:
        raise HTTPException(404, "unknown run")
    ev.set()
    return {"stopping": True}


def _sse_name(kind: str) -> str:
    """Map an event kind to the SSE frame name the UI subscribes to."""
    return {
        "stage_start": "stage", "stage_end": "stage",
        "measurement": "round", "error": "error",
        "reasoning": "log", "decision": "log", "artifact": "log", "log": "log",
    }.get(kind, "log")


@app.get("/api/runs/{run_id}/stream")
async def api_stream(run_id: str):
    snap = run_snapshot(conn(), run_id)
    if snap is None:
        raise HTTPException(404, "unknown run")
    q: asyncio.Queue = asyncio.Queue()
    _subscribers.setdefault(run_id, []).append(q)
    backlog = [
        {"stage": e["stage"], "kind": e["kind"], "title": e["title"],
         "body_md": e["body_md"], "payload": e["payload"], "replay": True}
        for e in snap["events"]
    ]

    terminal = snap["status"] != "running"

    async def gen():
        try:
            yield "event: sync\ndata: " + json.dumps({"events": backlog}) + "\n\n"
            for item in backlog:
                yield (f"event: {_sse_name(item['kind'])}\n"
                       f"data: {json.dumps(item)}\n\n")
            if terminal:
                yield "event: done\ndata: {}\n\n"
                return
            while True:
                try:
                    item = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield "event: heartbeat\ndata: {}\n\n"
                    continue
                if item is None:
                    yield "event: done\ndata: {}\n\n"
                    return
                yield f"event: {_sse_name(item['kind'])}\ndata: {json.dumps(item)}\n\n"
        finally:
            if q in _subscribers.get(run_id, []):
                _subscribers[run_id].remove(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


def _variant_row(variant_id: str):
    row = conn().execute("SELECT * FROM variants WHERE id=?", (variant_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "unknown variant")
    return dict(row)


@app.get("/api/variants/{variant_id}/source")
def api_source(variant_id: str):
    return PlainTextResponse(_variant_row(variant_id)["kernel_cu"])


@app.get("/api/variants/{variant_id}/ptx")
def api_ptx(variant_id: str):
    return PlainTextResponse(_variant_row(variant_id)["ptx"] or "")


@app.get("/api/variants/{variant_id}/diff")
def api_diff(variant_id: str, base: str | None = None):
    row = _variant_row(variant_id)
    if base is None:
        parent = conn().execute(
            "SELECT * FROM variants WHERE run_id=? AND round_n<? ORDER BY round_n DESC LIMIT 1",
            (row["run_id"], row["round_n"])).fetchone()
        base_src = parent["kernel_cu"] if parent else ""
    else:
        base_src = _variant_row(base)["kernel_cu"]
    return PlainTextResponse(ledger_stage.diff_text(base_src, row["kernel_cu"]))


@app.get("/api/variants/{variant_id}/proof")
def api_proof(variant_id: str):
    row = latest_proof(conn(), variant_id)
    if row is None:
        raise HTTPException(404, "no proof for variant")
    modules = {}
    for name, path in (row.get("modules") or {}).items():
        p = Path(__file__).resolve().parent.parent / "lean" / path
        modules[name] = p.read_text() if p.exists() else ""
    return {**row, "sources": modules,
            "numerical_scope": "exact real arithmetic; IEEE rounding not modeled"}


@app.get("/api/runs/{run_id}/report")
def api_report(run_id: str):
    snap = run_snapshot(conn(), run_id)
    if snap is None:
        raise HTTPException(404, "unknown run")
    lines = [f"# Run {run_id} — {snap['kernel_id']} on {snap['config']['gpu']}", "",
             f"Status: **{snap['status']}** ({snap['stop_reason']}).", "",
             "| round | variant | transform | sol % | p50 ms | proof | commit |",
             "|---|---|---|---|---|---|---|"]
    by_variant = {p["variant_id"]: p["profile"] for p in snap["profiles"]}
    proofs = {p["variant_id"]: p["status"] for p in snap["proofs"]}
    commits = {c["round_n"]: c["sha"] for c in snap["commits"]}
    for v in snap["variants"]:
        prof = by_variant.get(v["id"], {})
        lines.append(
            f"| {v['round_n']} | `{v['id']}` | `{v['transform_id'] or 'baseline'}` | "
            f"{prof.get('sol_pct', 0):.1f} | {prof.get('p50_ms', 0):.3f} | "
            f"{proofs.get(v['id'], 'n/a')} | `{(commits.get(v['round_n']) or '')[:12]}` |")
    lines += ["", f"numerical_scope: \"exact real arithmetic; IEEE rounding not modeled\"", ""]
    return PlainTextResponse("\n".join(lines))


def mount_static() -> None:
    dist = WEB_DIR / "dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="web")


mount_static()
