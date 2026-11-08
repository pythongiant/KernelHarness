"""SQLite persistence. One file, WAL, row factories."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .config import STATE_DIR, RunConfig, ensure_dirs

DDL = """
CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, kernel_id TEXT, config_json TEXT, status TEXT,
  stop_reason TEXT, created_at TEXT, ended_at TEXT);
CREATE TABLE IF NOT EXISTS rounds(run_id TEXT, n INTEGER, status TEXT, variant_id TEXT, parent_variant_id TEXT,
  started_at TEXT, ended_at TEXT, PRIMARY KEY(run_id,n));
CREATE TABLE IF NOT EXISTS variants(id TEXT PRIMARY KEY, run_id TEXT, round_n INTEGER, family TEXT,
  transform_id TEXT, params_json TEXT, kernel_cu TEXT, ptx TEXT, sha256 TEXT);
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, round_n INTEGER,
  stage TEXT, kind TEXT, title TEXT, body_md TEXT, payload_json TEXT, ts TEXT);
CREATE TABLE IF NOT EXISTS profiles(id INTEGER PRIMARY KEY AUTOINCREMENT, variant_id TEXT, json TEXT);
CREATE TABLE IF NOT EXISTS bottlenecks(id INTEGER PRIMARY KEY AUTOINCREMENT, variant_id TEXT, code TEXT,
  severity TEXT, evidence_md TEXT);
CREATE TABLE IF NOT EXISTS hypotheses(id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, round_n INTEGER,
  variant_id TEXT, source TEXT, bottleneck_code TEXT, transform_id TEXT, title TEXT, rationale_md TEXT,
  refs_json TEXT, expected_effect TEXT);
CREATE TABLE IF NOT EXISTS proofs(id INTEGER PRIMARY KEY AUTOINCREMENT, variant_id TEXT, status TEXT,
  model TEXT, open_goals_json TEXT, modules_json TEXT, axiom_audit_json TEXT, log_path TEXT);
CREATE TABLE IF NOT EXISTS commits(id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, round_n INTEGER,
  sha TEXT, message TEXT);
"""


def db_path() -> Path:
    return STATE_DIR / "harness.db"


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def connect(path: Path | None = None) -> sqlite3.Connection:
    ensure_dirs()
    p = path or db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(path: Path | None = None) -> sqlite3.Connection:
    conn = connect(path)
    conn.executescript(DDL)
    conn.commit()
    return conn


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# --- run ------------------------------------------------------------------
def create_run(conn: sqlite3.Connection, cfg: RunConfig) -> str:
    run_id = new_id("run")
    conn.execute(
        "INSERT INTO runs(id,kernel_id,config_json,status,stop_reason,created_at,ended_at)"
        " VALUES(?,?,?,?,?,?,?)",
        (run_id, cfg.kernel_id, cfg.db_json(), "running", None, now(), None),
    )
    conn.commit()
    return run_id


def finish_run(conn: sqlite3.Connection, run_id: str, status: str, stop_reason: str) -> None:
    conn.execute(
        "UPDATE runs SET status=?, stop_reason=?, ended_at=? WHERE id=?",
        (status, stop_reason, now(), run_id),
    )
    conn.commit()


# --- rounds ---------------------------------------------------------------
def open_round(
    conn: sqlite3.Connection, run_id: str, n: int, parent_variant_id: str | None
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO rounds(run_id,n,status,variant_id,parent_variant_id,started_at,ended_at)"
        " VALUES(?,?,?,?,?,?,?)",
        (run_id, n, "running", None, parent_variant_id, now(), None),
    )
    conn.commit()


def close_round(
    conn: sqlite3.Connection, run_id: str, n: int, status: str, variant_id: str | None
) -> None:
    conn.execute(
        "UPDATE rounds SET status=?, variant_id=?, ended_at=? WHERE run_id=? AND n=?",
        (status, variant_id, now(), run_id, n),
    )
    conn.commit()


# --- events ---------------------------------------------------------------
def add_event(
    conn: sqlite3.Connection,
    run_id: str,
    round_n: int | None,
    stage: str | None,
    kind: str,
    title: str,
    body_md: str = "",
    payload: dict | None = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO events(run_id,round_n,stage,kind,title,body_md,payload_json,ts)"
        " VALUES(?,?,?,?,?,?,?,?)",
        (
            run_id,
            round_n,
            stage,
            kind,
            title,
            body_md,
            json.dumps(payload) if payload is not None else None,
            now(),
        ),
    )
    conn.commit()
    return int(cur.lastrowid or 0)


# --- variants -------------------------------------------------------------
def add_variant(
    conn: sqlite3.Connection,
    *,
    variant_id: str,
    run_id: str,
    round_n: int,
    family: str,
    transform_id: str | None,
    params: dict,
    kernel_cu: str,
    ptx: str,
    sha256: str,
) -> None:
    conn.execute(
        "INSERT INTO variants(id,run_id,round_n,family,transform_id,params_json,kernel_cu,ptx,sha256)"
        " VALUES(?,?,?,?,?,?,?,?,?)",
        (
            variant_id,
            run_id,
            round_n,
            family,
            transform_id,
            json.dumps(params),
            kernel_cu,
            ptx,
            sha256,
        ),
    )
    conn.commit()


def get_variant(conn: sqlite3.Connection, variant_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM variants WHERE id=?", (variant_id,)).fetchone()


def add_profile(conn: sqlite3.Connection, variant_id: str, profile: dict) -> None:
    conn.execute(
        "INSERT INTO profiles(variant_id,json) VALUES(?,?)",
        (variant_id, json.dumps(profile)),
    )
    conn.commit()


def add_bottlenecks(conn: sqlite3.Connection, variant_id: str, rows: list[dict]) -> None:
    conn.executemany(
        "INSERT INTO bottlenecks(variant_id,code,severity,evidence_md) VALUES(?,?,?,?)",
        [(variant_id, r["code"], r["severity"], r["evidence_md"]) for r in rows],
    )
    conn.commit()


def add_hypothesis(conn: sqlite3.Connection, row: dict) -> int:
    cur = conn.execute(
        "INSERT INTO hypotheses(run_id,round_n,variant_id,source,bottleneck_code,transform_id,title,"
        "rationale_md,refs_json,expected_effect) VALUES(?,?,?,?,?,?,?,?,?,?)",
        (
            row["run_id"],
            row["round_n"],
            row["variant_id"],
            row["source"],
            row["bottleneck_code"],
            row["transform_id"],
            row["title"],
            row["rationale_md"],
            json.dumps(row.get("refs", [])),
            row["expected_effect"],
        ),
    )
    conn.commit()
    return int(cur.lastrowid or 0)


def add_proof(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        "INSERT INTO proofs(variant_id,status,model,open_goals_json,modules_json,axiom_audit_json,log_path)"
        " VALUES(?,?,?,?,?,?,?)",
        (
            row["variant_id"],
            row["status"],
            row["model"],
            json.dumps(row.get("open_goals", [])),
            json.dumps(row.get("modules", {})),
            json.dumps(row.get("axiom_audit", {})),
            row.get("log_path"),
        ),
    )
    conn.commit()


def add_commit(conn: sqlite3.Connection, run_id: str, round_n: int, sha: str, message: str) -> None:
    conn.execute(
        "INSERT INTO commits(run_id,round_n,sha,message) VALUES(?,?,?,?)",
        (run_id, round_n, sha, message),
    )
    conn.commit()


# --- queries --------------------------------------------------------------
def latest_profile(conn: sqlite3.Connection, variant_id: str) -> dict | None:
    row = conn.execute(
        "SELECT json FROM profiles WHERE variant_id=? ORDER BY id DESC LIMIT 1", (variant_id,)
    ).fetchone()
    return json.loads(row["json"]) if row else None


def latest_proof(conn: sqlite3.Connection, variant_id: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM proofs WHERE variant_id=? ORDER BY id DESC LIMIT 1", (variant_id,)
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    for k in ("open_goals_json", "modules_json", "axiom_audit_json"):
        d[k.removesuffix("_json")] = json.loads(d.pop(k) or "null")
    return d


def _rows(conn: sqlite3.Connection, sql: str, args: tuple = ()) -> list[dict]:
    return [dict(r) for r in conn.execute(sql, args).fetchall()]


def run_snapshot(conn: sqlite3.Connection, run_id: str) -> dict | None:
    run = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if not run:
        return None
    out = dict(run)
    out["config"] = json.loads(out.pop("config_json"))
    out["rounds"] = _rows(conn, "SELECT * FROM rounds WHERE run_id=? ORDER BY n", (run_id,))
    out["events"] = _rows(conn, "SELECT * FROM events WHERE run_id=? ORDER BY id", (run_id,))
    for e in out["events"]:
        e["payload"] = json.loads(e.pop("payload_json") or "null")
    vids = [
        r["id"]
        for r in conn.execute("SELECT id FROM variants WHERE run_id=?", (run_id,)).fetchall()
    ]
    out["variants"] = _rows(
        conn, "SELECT * FROM variants WHERE run_id=? ORDER BY round_n", (run_id,)
    )
    profiles: list[dict] = []
    bottlenecks: list[dict] = []
    proofs: list[dict] = []
    for vid in vids:
        profiles.extend(
            _rows(conn, "SELECT * FROM profiles WHERE variant_id=? ORDER BY id", (vid,))
        )
        bottlenecks.extend(
            _rows(conn, "SELECT * FROM bottlenecks WHERE variant_id=? ORDER BY id", (vid,))
        )
        proofs.extend(_rows(conn, "SELECT * FROM proofs WHERE variant_id=? ORDER BY id", (vid,)))
    for p in profiles:
        p["profile"] = json.loads(p.pop("json"))
    for p in proofs:
        for k in ("open_goals_json", "modules_json", "axiom_audit_json"):
            p[k.removesuffix("_json")] = json.loads(p.pop(k) or "null")
    out["profiles"] = profiles
    out["bottlenecks"] = bottlenecks
    out["proofs"] = proofs
    out["hypotheses"] = _rows(
        conn, "SELECT * FROM hypotheses WHERE run_id=? ORDER BY id", (run_id,)
    )
    for h in out["hypotheses"]:
        h["refs"] = json.loads(h.pop("refs_json") or "[]")
    out["commits"] = _rows(conn, "SELECT * FROM commits WHERE run_id=? ORDER BY id", (run_id,))
    return out


def list_runs(conn: sqlite3.Connection) -> list[dict]:
    return _rows(
        conn,
        "SELECT id,kernel_id,status,stop_reason,created_at,ended_at FROM runs ORDER BY created_at DESC",
    )
