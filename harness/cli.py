"""Command line interface.

    harness init
    harness registry [--scan]
    harness run --kernel ID --gpu H100 --backend modal --max-rounds N [--serve]
    harness verify --variant ID
    harness profile --variant ID
    harness serve --port 8787
    harness report --run ID
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import subprocess
import sys
from pathlib import Path

from . import db as dbm
from .config import ROOT, RunConfig, Sla, ensure_dirs
from .registry import geometry, registry

LEDGER_DIRNAME = "ledgers"


def cmd_init(args) -> int:
    ensure_dirs()
    conn = dbm.init_db()
    conn.close()
    from .stages.ledger import ensure_repo
    ensure_repo()
    print(f"initialized .harness/ and ledgers/ in {ROOT}")
    return 0


def cmd_registry(args) -> int:
    rows = registry()
    if not rows:
        print("no kernels registered under harness/kernels/")
        return 1
    for k in rows:
        env = k.shape_env()
        b, f = k.sol_numbers(env)
        g = geometry(k)
        print(f"{k.id:22s} family={k.family:11s} dtype={k.dtype:4s} "
              f"level={k.level} levels={k.levels()}")
        print(f"{'':22s} shape={env} grid={g['grid']} block={g['block']}")
        print(f"{'':22s} sol: kind={k.sol.kind} bytes={b} flops={f}")
        if args.scan:
            print(f"{'':22s} manifest={k.path}")
    return 0


def cmd_run(args) -> int:
    cfg = RunConfig(
        kernel_id=args.kernel, gpu=args.gpu, backend=args.backend,
        sla=Sla(max_rounds=args.max_rounds if args.max_rounds else 8,
                sol_pct=args.sol, speedup=args.speedup,
                require_verified=not args.no_require_verified),
    )
    ensure_dirs()
    conn = dbm.init_db()
    run_id = dbm.create_run(conn, cfg)

    async def emit(stage, kind, title, body_md, payload):
        print(f"[{stage or '-':9s}] {kind:11s} {title}")
        if body_md:
            for line in body_md.splitlines()[:6]:
                print(f"            {line[:160]}")

    cancel = asyncio.Event()
    if args.serve:
        import threading
        import uvicorn

        from .server import app as server_app
        from .server import _cancels

        _cancels[run_id] = cancel

        def serve():
            uvicorn.run(server_app, host="127.0.0.1", port=args.port, log_level="warning")

        threading.Thread(target=serve, daemon=True).start()
        print(f"UI: http://127.0.0.1:{args.port}  (run {run_id})")

    from .orchestrator import run_loop
    stop = asyncio.run(run_loop(conn, cfg, run_id, emit=emit, cancel_evt=cancel))
    print(f"\nstop reason: {stop}  (run {run_id})")
    return 0 if stop in ("SLA_MET", "MAX_ROUNDS", "STALLED", "STOPPED_BY_USER") else 1


def _variant(variant_id: str):
    from .model import Variant

    conn = dbm.init_db()
    row = dbm.get_variant(conn, variant_id)
    if row is None:
        print(f"unknown variant: {variant_id}", file=sys.stderr)
        return None, None
    from .registry import get

    kernel = get(
        conn.execute("SELECT kernel_id FROM runs WHERE id=?", (row["run_id"],)).fetchone()[0])
    params = json.loads(row["params_json"] or "{}")
    v = Variant.build(variant_id=row["id"], run_id=row["run_id"], round_n=row["round_n"],
                      kernel=kernel, level=kernel.level, transform_id=row["transform_id"],
                      params=params)
    v.source = row["kernel_cu"]
    v.ptx = row["ptx"] or ""
    return conn, v


def cmd_verify(args) -> int:
    from .stages import verify as verify_stage

    conn, variant = _variant(args.variant)
    if variant is None:
        return 2
    result = verify_stage.verify(variant)
    print(json.dumps({"status": result.status, "model": result.model,
                      "open_goals": result.open_goals,
                      "axiom_audit": result.axiom_audit,
                      "modules": result.modules,
                      "instantiation": result.instantiation}, indent=2))
    out = Path("proof") / "status.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.as_row(variant.id), indent=2) + "\n")
    return 0 if result.status == "VERIFIED" else 1


def cmd_profile(args) -> int:
    from .backends import make_backend
    from .backends.base import ProfileJob
    from .runner.gen import bench_driver, harness_header

    conn, variant = _variant(args.variant)
    if variant is None:
        return 2
    k = variant.kernel
    env = k.shape_env()
    g = geometry(k)
    b, f = k.sol_numbers(env)
    job = ProfileJob(
        variant_id=variant.id, gpu=args.gpu, source=variant.source,
        harness_header=harness_header(k, env, k.buffers),
        shape={"N": env.get("N", 0), "M": env.get("M", 0), "K": env.get("K", 0),
               "N_out": env.get("N_out", env.get("K", 0)), "N_rows": env.get("N_rows", 0),
               "T": env.get("T", 0), "D": env.get("D", 0)},
        tol=k.tol, reps=args.reps, warmup=args.warmup, baseline=k.baseline,
        bytes_moved=b, flops=f, sol_kind=k.sol.kind, out_index=k.out_index,
        buffers=k.buffers, dtype=k.dtype, family=k.family, grid=g["grid"], block=g["block"],
        extra={"level": variant.level, "family": variant.family, "transforms": []},
    )
    result = make_backend(args.backend, args.gpu).profile(job)
    print(json.dumps(result.data, indent=2))
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from .server import app as server_app

    uvicorn.run(server_app, host="127.0.0.1", port=args.port)
    return 0


def cmd_report(args) -> int:
    conn = dbm.init_db()
    snap = dbm.run_snapshot(conn, args.run)
    if snap is None:
        print(f"unknown run: {args.run}", file=sys.stderr)
        return 2
    rows = []
    for v in snap["variants"]:
        prof = next((p["profile"] for p in snap["profiles"] if p["variant_id"] == v["id"]), {})
        rows.append({"n": v["round_n"], "transform_id": v["transform_id"],
                     "sol_pct": prof.get("sol_pct", 0.0), "p50_ms": prof.get("p50_ms", 0.0),
                     "proof": next((p["status"] for p in snap["proofs"]
                                    if p["variant_id"] == v["id"]), "n/a"),
                     "sha": next((c["sha"] for c in snap["commits"]
                                  if c["round_n"] == v["round_n"]), "")})
    print(json.dumps({"run": args.run, "status": snap["status"],
                      "stop_reason": snap["stop_reason"], "rows": rows}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    # Behave like a normal Unix filter when piped into `head`/`less`.
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    p = argparse.ArgumentParser(prog="harness", description="kernel discovery harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("registry")
    s.add_argument("action", nargs="?", choices=["scan"], default=None)
    s.set_defaults(fn=cmd_registry, scan=False)
    s.add_argument("--scan", action="store_true")

    s = sub.add_parser("run")
    s.add_argument("--kernel", required=True)
    s.add_argument("--gpu", default="H100")
    s.add_argument("--backend", default="modal", choices=["modal", "sim"])
    s.add_argument("--max-rounds", type=int, default=8)
    s.add_argument("--sol", type=float, default=60.0)
    s.add_argument("--speedup", type=float, default=1.5)
    s.add_argument("--no-require-verified", action="store_true")
    s.add_argument("--serve", action="store_true")
    s.add_argument("--port", type=int, default=8787)
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("verify")
    s.add_argument("--variant", required=True)
    s.set_defaults(fn=cmd_verify)

    s = sub.add_parser("profile")
    s.add_argument("--variant", required=True)
    s.add_argument("--gpu", default="H100")
    s.add_argument("--backend", default="sim", choices=["modal", "sim"])
    s.add_argument("--reps", type=int, default=50)
    s.add_argument("--warmup", type=int, default=25)
    s.set_defaults(fn=cmd_profile)

    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8787)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("report")
    s.add_argument("--run", required=True)
    s.set_defaults(fn=cmd_report)

    args = p.parse_args(argv)
    if not hasattr(args, "scan"):
        args.scan = False
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
