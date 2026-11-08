"""Global settings, SLA definition and the run configuration schema.

Fixed literals live here so the DB, UI, orchestrator and tests agree.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
LEDGER_DIR = Path(os.environ.get("HARNESS_LEDGER_DIR", ROOT / "ledgers"))
LEAN_DIR = ROOT / "lean"
WEB_DIR = ROOT / "web"
STATE_DIR = Path(os.environ.get("HARNESS_STATE_DIR", ROOT / ".harness"))
KERNEL_DIR = ROOT / "harness" / "kernels"
TEMPLATE_DIR = ROOT / "harness" / "templates"

# --- fixed literals -------------------------------------------------------
STAGES = ("PROFILE", "DIAGNOSE", "RESEARCH", "OPTIMIZE", "VERIFY", "LEDGER")

RUN_STATUS = ("running", "sla_met", "max_rounds", "stalled", "blocked", "stopped")

STOP_REASONS = ("SLA_MET", "MAX_ROUNDS", "STALLED", "BLOCKED", "STOPPED_BY_USER")

PROOF_STATUS = ("VERIFIED", "PARTIAL", "FAILED", "UNSUPPORTED")

EVENT_KINDS = (
    "stage_start",
    "stage_end",
    "reasoning",
    "measurement",
    "decision",
    "artifact",
    "error",
    "log",
)

NUMERICAL_SCOPE = "exact real arithmetic; IEEE rounding not modeled"


class Sla(BaseModel):
    sol_pct: float = 60.0
    speedup: float = 1.5
    latency_ms: float | None = None
    require_verified: bool = True
    reps: int = 50
    # A fresh sandbox starts with idle GPU clocks; a longer warmup keeps the first
    # kernel measured in a container comparable with later ones.
    warmup: int = 25
    tolerance_pct: float = 3.0
    max_rounds: int = 8
    stall_rounds: int = 2


class RunConfig(BaseModel):
    kernel_id: str
    gpu: str = "H100"
    backend: str = "modal"
    sla: Sla = Field(default_factory=Sla)

    def db_json(self) -> str:
        return self.model_dump_json()


def ensure_dirs() -> None:
    for d in (STATE_DIR, LEDGER_DIR, LEDGER_DIR / "runs"):
        d.mkdir(parents=True, exist_ok=True)
