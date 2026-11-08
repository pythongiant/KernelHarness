"""Command line interface (`harness ...`). Run `harness <command> --help` for details."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import threading
from pathlib import Path

from . import db as dbm
from .config import LEDGER_DIR, STATE_DIR, RunConfig, Sla, ensure_dirs
from .registry import geometry, registry

#: Stop reasons that count as a normal end of a run (exit code 0).
_CLEAN_STOPS = ("SLA_MET", "MAX_ROUNDS", "STALLED", "STOPPED_BY_USER")


def cmd_init(args: argparse.Namespace) -> int:
    from .stages.ledger import ensure_repo

    ensure_dirs()
    dbm.init_db().close()
    ensure_repo()
    print(f"state:  {STATE_DIR}")
    print(f"ledger: {LEDGER_DIR} (git repository)")
    return 0


def cmd_registry(args: argparse.Namespace) -> int:
    kernels = registry()
    if not kernels:
        print("no kernels registered under harness/kernels/", file=sys.stderr)
        return 1
    for k in kernels:
        env = k.shape_env()
        b, f = k.sol_numbers(env)
        g = geometry(k)
        print(
            f"{k.id:22s} family={k.family:11s} dtype={k.dtype:4s} "
            f"level={k.level} levels={k.levels()}"
        )
        print(f"{'':22s} shape={env} grid={g['grid']} block={g['block']}")
        print(f"{'':22s} sol: kind={k.sol.kind} bytes={b} flops={f}")
        if args.verbose:
            print(f"{'':22s} manifest={k.path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .orchestrator import run_loop

    cfg = RunConfig(
        kernel_id=args.kernel,
        gpu=args.gpu,
        backend=args.backend,
        sla=Sla(
            max_rounds=args.max_rounds,
            sol_pct=args.sol,
            speedup=args.speedup,
            require_verified=not args.no_require_verified,
        ),
    )
    ensure_dirs()
    conn = dbm.init_db()
    run_id = dbm.create_run(conn, cfg)

    async def emit(stage, kind, title, body_md, payload):
        print(f"[{stage or '-':9s}] {kind:11s} {title}")
        for line in (body_md or "").splitlines()[:6]:
            print(f"            {line[:160]}")

    cancel = asyncio.Event()
    if args.serve:
        import uvicorn

        from .server import _cancels
        from .server import app as server_app

        _cancels[run_id] = cancel
        threading.Thread(
            target=uvicorn.run,
            args=(server_app,),
            kwargs={"host": "127.0.0.1", "port": args.port, "log_level": "warning"},
            daemon=True,
        ).start()
        print(f"UI: http://127.0.0.1:{args.port}  (run {run_id})")

    stop = asyncio.run(run_loop(conn, cfg, run_id, emit=emit, cancel_evt=cancel))
    print(f"\nstop reason: {stop}  (run {run_id})")
    return 0 if stop in _CLEAN_STOPS else 1


def _load_variant(variant_id: str):
    """Rebuild a stored variant from the database, or return None if unknown."""
    from .model import Variant
    from .registry import get

    conn = dbm.init_db()
    row = dbm.get_variant(conn, variant_id)
    if row is None:
        print(f"unknown variant: {variant_id}", file=sys.stderr)
        return None
    kernel_id = conn.execute("SELECT kernel_id FROM runs WHERE id=?", (row["run_id"],)).fetchone()[
        0
    ]
    kernel = get(kernel_id)
    variant = Variant.build(
        variant_id=row["id"],
        run_id=row["run_id"],
        round_n=row["round_n"],
        kernel=kernel,
        level=kernel.level,
        transform_id=row["transform_id"],
        params=json.loads(row["params_json"] or "{}"),
    )
    variant.source = row["kernel_cu"]
    variant.ptx = row["ptx"] or ""
    return variant


def cmd_verify(args: argparse.Namespace) -> int:
    from .stages import verify as verify_stage

    variant = _load_variant(args.variant)
    if variant is None:
        return 2
    result = verify_stage.verify(variant)
    summary = result.summary()
    print(json.dumps(summary, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if result.status == "VERIFIED" else 1


def cmd_profile(args: argparse.Namespace) -> int:
    from .backends import make_backend
    from .runner.gen import build_profile_job

    variant = _load_variant(args.variant)
    if variant is None:
        return 2
    job = build_profile_job(variant, args.gpu, reps=args.reps, warmup=args.warmup)
    result = make_backend(args.backend, args.gpu).profile(job)
    print(json.dumps(result.data, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server import app as server_app

    uvicorn.run(server_app, host="127.0.0.1", port=args.port)
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    runs = dbm.list_runs(dbm.init_db())[: args.limit]
    if args.quiet:
        for r in runs:
            print(r["id"])
        return 0
    if not runs:
        print("no runs yet: try `harness run --kernel elementwise_affine --backend sim`")
        return 0
    print(f"{'run':18s} {'kernel':22s} {'status':11s} {'stop reason':16s} started")
    for r in runs:
        print(
            f"{r['id']:18s} {r['kernel_id']:22s} {r['status']:11s} "
            f"{r['stop_reason'] or '-':16s} {r['created_at']}"
        )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    conn = dbm.init_db()
    run_id = args.run
    if run_id == "latest":
        runs = dbm.list_runs(conn)
        if not runs:
            print("no runs yet", file=sys.stderr)
            return 2
        run_id = runs[0]["id"]
    snap = dbm.run_snapshot(conn, run_id)
    if snap is None:
        print(f"unknown run: {run_id}", file=sys.stderr)
        return 2
    profiles = {p["variant_id"]: p["profile"] for p in snap["profiles"]}
    proofs = {p["variant_id"]: p["status"] for p in snap["proofs"]}
    commits = {c["round_n"]: c["sha"] for c in snap["commits"]}
    rows = []
    for v in snap["variants"]:
        prof = profiles.get(v["id"], {})
        rows.append(
            {
                "n": v["round_n"],
                "transform_id": v["transform_id"],
                "sol_pct": prof.get("sol_pct", 0.0),
                "p50_ms": prof.get("p50_ms", 0.0),
                "proof": proofs.get(v["id"], "n/a"),
                "sha": commits.get(v["round_n"], ""),
            }
        )
    print(
        json.dumps(
            {
                "run": run_id,
                "status": snap["status"],
                "stop_reason": snap["stop_reason"],
                "rows": rows,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="harness",
        description="Optimise a CUDA kernel round by round, proving each variant correct.",
    )
    sub = p.add_subparsers(dest="cmd", required=True, metavar="<command>")

    s = sub.add_parser("init", help="create the state directory and the ledger repository")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("registry", help="list registered kernels, shapes and SOL formulas")
    s.add_argument("-v", "--verbose", action="store_true", help="also print manifest paths")
    s.set_defaults(fn=cmd_registry)

    s = sub.add_parser("run", help="run the optimisation loop on a kernel")
    s.add_argument("--kernel", required=True, help="kernel id (see `harness registry`)")
    s.add_argument("--gpu", default="H100", help="GPU type (default: H100)")
    s.add_argument(
        "--backend",
        default="modal",
        choices=["modal", "sim"],
        help="modal = real GPU, sim = offline cost model (default: modal)",
    )
    s.add_argument("--max-rounds", type=int, default=8, help="round cap (default: 8)")
    s.add_argument(
        "--sol", type=float, default=60.0, help="target percent of roofline peak (default: 60)"
    )
    s.add_argument(
        "--speedup",
        type=float,
        default=1.5,
        help="target speedup over the torch baseline (default: 1.5)",
    )
    s.add_argument(
        "--no-require-verified",
        action="store_true",
        help="let the target pass without a VERIFIED proof (experiments only)",
    )
    s.add_argument("--serve", action="store_true", help="also start the web UI")
    s.add_argument("--port", type=int, default=8787, help="web UI port (default: 8787)")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("runs", help="list recorded runs, newest first")
    s.add_argument("-n", "--limit", type=int, default=20, help="how many (default: 20)")
    s.add_argument("-q", "--quiet", action="store_true", help="print run ids only")
    s.set_defaults(fn=cmd_runs)

    s = sub.add_parser("report", help="print a run's rounds as JSON")
    s.add_argument("--run", default="latest", help="run id, or `latest` (default)")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("verify", help="re-check one variant against the Lean model")
    s.add_argument("--variant", required=True, help="variant id")
    s.add_argument("--out", help="also write the proof status JSON to this path")
    s.set_defaults(fn=cmd_verify)

    s = sub.add_parser("profile", help="re-measure one variant")
    s.add_argument("--variant", required=True, help="variant id")
    s.add_argument("--gpu", default="H100", help="GPU type (default: H100)")
    s.add_argument("--backend", default="sim", choices=["modal", "sim"], help="default: sim")
    s.add_argument("--reps", type=int, default=50, help="timed repetitions (default: 50)")
    s.add_argument("--warmup", type=int, default=25, help="warmup launches (default: 25)")
    s.set_defaults(fn=cmd_profile)

    s = sub.add_parser("serve", help="start the web UI")
    s.add_argument("--port", type=int, default=8787, help="port (default: 8787)")
    s.set_defaults(fn=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    # Behave like a normal Unix filter when piped into `head` or `less`.
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
