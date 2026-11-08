"""The closed loop: PROFILE → DIAGNOSE → RESEARCH → OPTIMIZE → VERIFY → LEDGER.

Invariants enforced here:
* a profile with a failing numeric check or a launch failure blocks the run (no commit),
* the SLA gate needs measured (non-simulated) numbers *and* a `VERIFIED` proof,
* a candidate is kept only if it improves `p50_ms` by `tolerance_pct`, otherwise the
  next round starts again from the parent (the attempt is still committed),
* every stage transition writes an `events` row whose `body_md` is the reasoning the
  UI shows.
"""

from __future__ import annotations

import asyncio
import json
import traceback
from collections.abc import Awaitable, Callable

from .backends import make_backend
from .config import STATUS_FOR_STOP, RunConfig
from .db import (
    add_bottlenecks,
    add_commit,
    add_event,
    add_hypothesis,
    add_profile,
    add_proof,
    add_variant,
    close_round,
    finish_run,
    open_round,
)
from .model import Variant
from .registry import Kernel
from .registry import get as get_kernel
from .runner.gen import build_profile_job
from .stages import diagnose as diagnose_stage
from .stages import ledger as ledger_stage
from .stages import optimize as optimize_stage
from .stages import research as research_stage
from .stages import verify as verify_stage
from .stages.features import features
from .translator import lower
from .translator.parse import UnsupportedInstruction, parse_ptx

EmitFn = Callable[[str | None, str, str, str, dict | None], Awaitable[None]]


def sla_met(p: dict, proof: dict, cfg: RunConfig) -> tuple[bool, dict]:
    checks = {
        "sol": p["sol_pct"] >= cfg.sla.sol_pct,
        "speedup": (
            p["baseline"].get("speedup") is not None and p["baseline"]["speedup"] >= cfg.sla.speedup
        ),
        "latency": cfg.sla.latency_ms is None or p["p50_ms"] <= cfg.sla.latency_ms,
        "verified": (not cfg.sla.require_verified) or proof["status"] == "VERIFIED",
        "numeric": p["numeric"] == "PASS",
        "measured": not p["simulated"],
    }
    return all(checks.values()), checks


def proof_row_view(proof) -> dict:
    """Adapt a `ProofResult` to the `{"status": ...}` shape the gate reads."""
    return {
        "status": getattr(proof, "status", proof.get("status") if isinstance(proof, dict) else None)
    }


def _fmt_sla(checks: dict) -> str:
    return ", ".join(f"`{k}`={'pass' if v else 'fail'}" for k, v in checks.items())


class Orchestrator:
    def __init__(
        self,
        conn,
        cfg: RunConfig,
        run_id: str,
        *,
        emit: EmitFn | None = None,
        cancel: asyncio.Event | None = None,
        backend=None,
    ):
        self.conn = conn
        self.cfg = cfg
        self.run_id = run_id
        self.emit = emit
        self.cancel = cancel or asyncio.Event()
        self.backend = backend or make_backend(cfg.backend, cfg.gpu)
        self.kernel: Kernel | None = None
        self.rounds_log: list[dict] = []

    # -- event plumbing --------------------------------------------------
    async def _event(
        self,
        stage: str | None,
        kind: str,
        title: str,
        body_md: str = "",
        payload: dict | None = None,
        round_n: int | None = None,
    ) -> None:
        add_event(self.conn, self.run_id, round_n, stage, kind, title, body_md, payload)
        if self.emit:
            await self.emit(stage, kind, title, body_md, payload)

    async def _stage(self, stage: str, round_n: int, title: str, body: str) -> None:
        await self._event(stage, "stage_start", f"{stage}: {title}", body, None, round_n)
        await self._event(stage, "stage_end", f"{stage} done", "", None, round_n)

    def _features(self, variant: Variant) -> dict:
        try:
            parsed = self._program(variant)
        except (UnsupportedInstruction, NotImplementedError):
            parsed = None
        if parsed is not None:
            return features(self._ptx_text(variant, parsed))
        return features(variant.source)

    def _program(self, variant: Variant):
        if variant.ptx:
            return parse_ptx(variant.ptx)["program"]
        return lower.lower(variant)[0]

    @staticmethod
    def _ptx_text(variant: Variant, program) -> str:
        if variant.ptx:
            return variant.ptx
        return "\n".join(i["op"] for i in program)

    # -- profiling -------------------------------------------------------
    async def _profile(self, variant: Variant, round_n: int, stage: str = "PROFILE") -> dict:
        job = build_profile_job(
            variant, self.cfg.gpu, reps=self.cfg.sla.reps, warmup=self.cfg.sla.warmup
        )
        try:
            result = self.backend.profile(job)
        except Exception as exc:
            await self._event(
                stage,
                "error",
                "profile failed",
                f"```\n{type(exc).__name__}: {exc}\n```",
                None,
                round_n,
            )
            raise
        data = dict(result.data)
        if not data.get("ptx"):
            data["ptx"] = variant.ptx or ""
        add_profile(self.conn, variant.id, data)
        if data.get("ptx"):
            variant.ptx = data["ptx"]
        return data

    # -- one round -------------------------------------------------------
    async def _round(
        self, n: int, parent: Variant, parent_profile: dict, parent_proof: dict
    ) -> tuple[Variant, dict, dict, str, bool]:
        """Run one round. Returns (best_variant, profile, proof, verdict, improved)."""
        open_round(self.conn, self.run_id, n, parent.id)

        diag_input = parent_profile
        await self._stage(
            "DIAGNOSE",
            n,
            f"diagnose round {n}",
            f"Analysing the parent variant `{parent.id}` "
            f"(`sol_pct={diag_input['sol_pct']:.1f}`, `p50_ms={diag_input['p50_ms']:.3f}`, "
            f"`bound={diag_input['bound']}`).",
        )

        bottlenecks = diagnose_stage.diagnose(
            diag_input, parent, self.kernel, proof=parent_proof, feats=self._features(parent)
        )
        add_bottlenecks(self.conn, parent.id, bottlenecks)
        diag_md = (
            "\n".join(
                f"- **{b['code']}** ({b['severity']}): {b['evidence_md']}" for b in bottlenecks
            )
            or "- no bottleneck rule fired"
        )
        await self._event("DIAGNOSE", "reasoning", "bottlenecks", diag_md, None, n)

        await self._event(
            "RESEARCH",
            "stage_start",
            "RESEARCH: rank fix candidates",
            f"{len(bottlenecks)} diagnosed bottleneck(s) -> catalog lookup, severity-ranked.",
            None,
            n,
        )
        attempted = {
            r[0]
            for r in self.conn.execute(
                "SELECT DISTINCT transform_id FROM hypotheses WHERE run_id=? AND round_n<?",
                (self.run_id, n),
            ).fetchall()
            if r[0]
        }
        hypotheses = research_stage.propose(
            bottlenecks, diag_input, self.kernel, parent, exclude_transforms=attempted
        )
        hyp_ids = []
        for h in hypotheses:
            hid = add_hypothesis(
                self.conn,
                {
                    "run_id": self.run_id,
                    "round_n": n,
                    "variant_id": parent.id,
                    "source": h["source"],
                    "bottleneck_code": h["bottleneck_code"],
                    "transform_id": h["transform_id"],
                    "title": h["title"],
                    "rationale_md": h["rationale_md"],
                    "refs": h["refs"],
                    "expected_effect": h["expected_effect"],
                },
            )
            h["id"] = hid
            hyp_ids.append(hid)
        hyp_md = (
            "\n\n".join(
                f"**{h['title']}** (source `{h['source']}`)\n\n{h['rationale_md']}\n\n"
                f"*Expected effect:* {h['expected_effect']}\n\nRefs: {', '.join(h['refs']) or '—'}"
                for h in hypotheses
            )
            or "No transform available for the diagnosed bottlenecks."
        )
        await self._event("RESEARCH", "reasoning", "ranked hypotheses", hyp_md, None, n)

        await self._event(
            "RESEARCH", "stage_end", f"RESEARCH produced {len(hypotheses)} hypotheses", "", None, n
        )
        if not hypotheses:
            why = (
                "the proof status is not `VERIFIED`, so optimisation is gated until the "
                "obligations close"
                if parent_proof.get("status") != "VERIFIED"
                else "no catalog entry matches the diagnosed bottlenecks"
            )
            await self._event(
                "RESEARCH",
                "error",
                "NO_TRANSFORM_AVAILABLE",
                f"No transform can be applied: {why}.\n\n"
                f"Diagnosis kept in the ledger; the run stops as `blocked` "
                f"rather than guessing.",
                None,
                n,
            )
            close_round(self.conn, self.run_id, n, "blocked", parent.id)
            return parent, parent_profile, parent_proof, "BLOCKED", False

        await self._event(
            "OPTIMIZE",
            "stage_start",
            "OPTIMIZE: materialize variant",
            f"Taking the top-ranked hypothesis `{hypotheses[0]['transform_id']}`.",
            None,
            n,
        )
        chosen = None
        child = None
        noop_reasons: list[str] = []
        for candidate in hypotheses:
            try:
                child = optimize_stage.apply_transform(parent, candidate, n)
                chosen = candidate
                break
            except optimize_stage.NoOpVariant as exc:
                noop_reasons.append(f"`{candidate['transform_id']}`: {exc}")
        if chosen is None or child is None:
            await self._event(
                "OPTIMIZE",
                "reasoning",
                "no applicable transform",
                "Every proposed transform is already applied at this level:\n\n"
                + "\n".join(f"- {r}" for r in noop_reasons),
                None,
                n,
            )
            close_round(self.conn, self.run_id, n, "reverted", parent.id)
            return parent, parent_profile, parent_proof, "exhausted", False

        await self._event(
            "OPTIMIZE",
            "stage_end",
            f"OPTIMIZE built `{child.id}`",
            f"level {child.level}, params `{child.params}`, sha256 `{child.sha256[:12]}`.",
            None,
            n,
        )
        add_variant(
            self.conn,
            variant_id=child.id,
            run_id=self.run_id,
            round_n=n,
            family=child.family,
            transform_id=child.transform_id,
            params=child.params,
            kernel_cu=child.source,
            ptx=child.ptx,
            sha256=child.sha256,
        )
        diff = ledger_stage.diff_text(parent.source, child.source)
        await self._event(
            "OPTIMIZE",
            "artifact",
            f"variant `{child.id}`",
            f"Applied `{chosen['transform_id']}` at level {child.level}.\n\n"
            f"```diff\n{diff[:4000]}\n```",
            None,
            n,
        )

        await self._event(
            "VERIFY",
            "stage_start",
            "VERIFY: emit Lean and check",
            "Lowering the template to a concrete instruction program, "
            "computing reference bytes, emitting the Lean module and "
            "running `lake env lean`.",
            None,
            n,
        )
        proof = verify_stage.verify(child)
        add_proof(self.conn, proof.as_row(child.id))
        await self._event(
            "VERIFY",
            "decision",
            f"proof {proof.status}",
            f"Status `{proof.status}` (model `{proof.model}`, instantiation "
            f"`{proof.instantiation}`).\n\nOpen goals: "
            f"{'; '.join(proof.open_goals) or 'none'}\n\n"
            f"Axiom audit: `{json.dumps(proof.axiom_audit)}`",
            None,
            n,
        )

        await self._event(
            "PROFILE",
            "stage_start",
            "PROFILE: measure candidate",
            f"Running {self.cfg.sla.reps} reps after "
            f"{self.cfg.sla.warmup} warmup on `{self.cfg.gpu}` "
            f"(`{self.cfg.backend}` backend).",
            None,
            n,
        )
        profile = await self._profile(child, n)
        # A candidate is kept when it earns its place: a latency win beyond the noise
        # margin, a SOL win beyond the noise margin, or simply satisfying the SLA (the
        # gate is the objective, so a variant that passes is never thrown away for being
        # "only" a small improvement).
        gate_now, _gate_checks = sla_met(profile, proof_row_view(proof), self.cfg)
        better_p50 = profile["p50_ms"] <= parent_profile["p50_ms"] * (
            1 - self.cfg.sla.tolerance_pct / 100.0
        )
        better_sol = profile["sol_pct"] >= parent_profile["sol_pct"] + self.cfg.sla.tolerance_pct
        improved = better_p50 or better_sol or gate_now
        await self._event(
            "PROFILE",
            "measurement",
            f"round {n} measurement",
            f"`p50_ms` {parent_profile['p50_ms']:.3f} → {profile['p50_ms']:.3f}, "
            f"`sol_pct` {parent_profile['sol_pct']:.1f} → {profile['sol_pct']:.1f}, "
            f"`numeric` {profile['numeric']}, `simulated` {profile['simulated']}.",
            None,
            n,
        )

        if profile["numeric"] == "FAIL":
            await self._event(
                "PROFILE",
                "error",
                f"round {n} numeric FAIL",
                "The optimized variant does not reproduce the reference "
                "output (see `max_abs_err`); the round is not usable.",
                None,
                n,
            )
            close_round(self.conn, self.run_id, n, "blocked", child.id)
            return parent, parent_profile, {"status": "BLOCKED"}, "BLOCKED", False

        best_variant, best_profile = (child, profile) if improved else (parent, parent_profile)
        why = (
            "satisfies the SLA"
            if gate_now
            else "p50 improvement beyond tolerance"
            if better_p50
            else "SOL improvement beyond tolerance"
            if better_sol
            else "within noise"
        )
        verdict = ("improved" if improved else "reverted") + f" ({why}), proof {proof.status}"

        artifacts = {
            "variant_id": child.id,
            "profile": profile,
            "diagnosis_md": diag_md,
            "hypotheses": hypotheses,
            "hypothesis_ids": [str(i) for i in hyp_ids],
            "kernel_cu": child.source,
            "kernel_ptx": profile.get("ptx", "") or child.ptx,
            "diff": diff,
            "proof": proof.summary(),
            "lean_modules": proof.module_sources(),
            "proof_log": proof.log,
        }
        codes = [b["code"] for b in bottlenecks]
        await self._event(
            "LEDGER",
            "stage_start",
            f"LEDGER: commit round {n}",
            f"vendoring profile/diagnosis/hypotheses/kernel/diff/proof "
            f"into the ledger at `runs/{self.run_id}/round-{n:02d}/`.",
            None,
            n,
        )
        sha, message = ledger_stage.commit_round(
            self.run_id,
            n,
            self.kernel.id,
            codes,
            chosen["transform_id"],
            parent_profile["sol_pct"],
            profile["sol_pct"],
            parent_profile["p50_ms"],
            profile["p50_ms"],
            proof.status,
            artifacts,
        )
        add_commit(self.conn, self.run_id, n, sha, message)
        await self._event(
            "LEDGER", "artifact", f"committed round {n}", f"`{sha[:12]}` — {message}", None, n
        )

        self.rounds_log.append(
            {
                "n": n,
                "transform_id": chosen["transform_id"],
                "sol_pct": profile["sol_pct"],
                "p50_ms": profile["p50_ms"],
                "proof": proof.status,
                "sha": sha,
            }
        )
        close_round(self.conn, self.run_id, n, "improved" if improved else "reverted", child.id)
        return (
            best_variant,
            best_profile,
            {
                "status": proof.status,
                "id": child.id,
                "open_goals": proof.open_goals,
                "model": proof.model,
                "axiom_audit": proof.axiom_audit,
                "instantiation": proof.instantiation,
            },
            verdict,
            improved,
        )

    # -- loop ------------------------------------------------------------
    async def run(self) -> str:
        self.kernel = self._kernel()
        await self._event(
            None,
            "log",
            "run started",
            f"kernel `{self.kernel.id}`, gpu `{self.cfg.gpu}`, "
            f"backend `{self.cfg.backend}`, max_rounds "
            f"{self.cfg.sla.max_rounds}.",
        )
        base = Variant.build(
            variant_id=f"{self.run_id}-v0",
            run_id=self.run_id,
            round_n=0,
            kernel=self.kernel,
            level=self.kernel.level,
            transform_id=None,
            params={},
        )
        add_variant(
            self.conn,
            variant_id=base.id,
            run_id=self.run_id,
            round_n=0,
            family=base.family,
            transform_id=None,
            params={},
            kernel_cu=base.source,
            ptx="",
            sha256=base.sha256,
        )

        try:
            profile = await self._profile(base, 0)
        except Exception:
            await self._event(
                None,
                "error",
                "baseline profile failed",
                f"```\n{traceback.format_exc()[-3000:]}\n```",
            )
            finish_run(self.conn, self.run_id, "blocked", "BLOCKED")
            return "BLOCKED"

        if profile["numeric"] == "FAIL":
            await self._event(
                "PROFILE",
                "error",
                "baseline numeric FAIL",
                "The baseline variant does not reproduce the reference "
                "output; no optimisation can be validated from here.",
            )
            finish_run(self.conn, self.run_id, "blocked", "BLOCKED")
            return "BLOCKED"

        await self._event(
            "PROFILE",
            "measurement",
            "baseline measurement",
            f"`p50_ms={profile['p50_ms']:.3f}`, `sol_pct={profile['sol_pct']:.1f}`, "
            f"`bound={profile['bound']}`, `numeric={profile['numeric']}`, "
            f"`simulated={profile['simulated']}`.",
        )

        await self._event(
            "VERIFY",
            "stage_start",
            "VERIFY: baseline proof",
            "Checking the unmodified kernel against the model before any "
            "optimisation, so the track record starts from a verified "
            "baseline.",
            None,
            0,
        )
        proof = verify_stage.verify(base)
        add_proof(self.conn, proof.as_row(base.id))
        await self._event(
            "VERIFY",
            "decision",
            f"baseline proof {proof.status}",
            f"Status `{proof.status}`; open goals: {'; '.join(proof.open_goals) or 'none'}",
            None,
            0,
        )
        cur_proof = {"status": proof.status, "open_goals": proof.open_goals}

        open_round(self.conn, self.run_id, 0, None)
        await self._event(
            "LEDGER",
            "stage_start",
            "LEDGER: commit baseline",
            f"`runs/{self.run_id}/round-00/`",
            None,
            0,
        )
        base_artifacts = {
            "variant_id": base.id,
            "profile": profile,
            "diagnosis_md": "- baseline: no diagnosis before the first measurement",
            "hypotheses": [],
            "hypothesis_ids": [],
            "kernel_cu": base.source,
            "kernel_ptx": profile.get("ptx", "") or base.ptx,
            "diff": "",
            "proof": proof.summary(),
            "lean_modules": {},
            "proof_log": proof.log,
        }
        base_sha, base_msg = ledger_stage.commit_round(
            self.run_id,
            0,
            self.kernel.id,
            [],
            None,
            profile["sol_pct"],
            profile["sol_pct"],
            profile["p50_ms"],
            profile["p50_ms"],
            proof.status,
            base_artifacts,
        )
        add_commit(self.conn, self.run_id, 0, base_sha, base_msg)
        self.rounds_log.append(
            {
                "n": 0,
                "transform_id": None,
                "sol_pct": profile["sol_pct"],
                "p50_ms": profile["p50_ms"],
                "proof": proof.status,
                "sha": base_sha,
            }
        )
        close_round(self.conn, self.run_id, 0, "baseline", base.id)
        await self._event(
            "LEDGER", "artifact", "committed baseline", f"`{base_sha[:12]}` — {base_msg}", None, 0
        )

        variant, cur_profile = base, profile
        stalls = 0
        stop = "MAX_ROUNDS"
        for n in range(1, self.cfg.sla.max_rounds + 1):
            if self.cancel.is_set():
                stop = "STOPPED_BY_USER"
                break
            gate_met, checks = sla_met(cur_profile, cur_proof, self.cfg)
            if gate_met:
                stop = "SLA_MET"
                break
            await self._event(
                "PROFILE",
                "reasoning",
                f"gate check before round {n}",
                f"SLA not met yet: {_fmt_sla(checks)}.",
            )
            try:
                variant, cur_profile, cur_proof, verdict, improved = await self._round(
                    n, variant, cur_profile, cur_proof
                )
            except Exception:
                await self._event(
                    None,
                    "error",
                    f"round {n} failed",
                    f"```\n{traceback.format_exc()[-4000:]}\n```",
                    None,
                    n,
                )
                finish_run(self.conn, self.run_id, "blocked", "BLOCKED")
                return "BLOCKED"
            stalls = 0 if improved else stalls + 1
            if cur_proof.get("status") == "BLOCKED":
                finish_run(self.conn, self.run_id, "blocked", "BLOCKED")
                return "BLOCKED"
            if stalls >= self.cfg.sla.stall_rounds:
                stop = "STALLED"
                break
        else:
            stop = "MAX_ROUNDS"

        gate, checks = sla_met(cur_profile, cur_proof, self.cfg)
        if gate:
            stop = "SLA_MET"
        await self._event(
            None,
            "decision",
            f"run finished: {stop}",
            f"Final gate: {_fmt_sla(checks)}.\n\n"
            f"`sol_pct={cur_profile['sol_pct']:.1f}`, "
            f"`p50_ms={cur_profile['p50_ms']:.3f}`, "
            f"proof `{cur_proof.get('status')}`.",
        )
        ledger_stage.regenerate_report(
            self.rounds_log,
            self.run_id,
            meta={
                "kernel_id": self.kernel.id,
                "gpu": self.cfg.gpu,
                "backend": self.cfg.backend,
                "status": STATUS_FOR_STOP[stop],
                "stop_reason": stop,
            },
        )
        finish_run(self.conn, self.run_id, STATUS_FOR_STOP[stop], stop)
        return stop

    def _kernel(self) -> Kernel:
        return get_kernel(self.cfg.kernel_id)


async def run_loop(
    conn,
    cfg: RunConfig,
    run_id: str,
    emit: EmitFn | None = None,
    cancel_evt: asyncio.Event | None = None,
    backend=None,
) -> str:
    """Entry point used by the CLI and the server."""
    orch = Orchestrator(conn, cfg, run_id, emit=emit, cancel=cancel_evt, backend=backend)
    return await orch.run()
