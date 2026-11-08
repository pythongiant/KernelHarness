import React, { useEffect, useState } from "react";
import { api } from "../api";
import { BottleneckCard, DiffView, HypothesisCard, MetricTable, ProofBadge } from "../components/primitives";

export function RoundDetail({ snap, n }: { snap: any; n: number }) {
  const round = snap.rounds.find((r: any) => r.n === n);
  const variant = snap.variants.find((v: any) => v.round_n === n);
  const profileRow = variant && snap.profiles.filter((p: any) => p.variant_id === variant.id).slice(-1)[0];
  const profile = profileRow?.profile;
  const pitfalls = variant && snap.bottlenecks.filter((b: any) => b.variant_id === variant.id);
  const hyps = snap.hypotheses.filter((h: any) => h.round_n === n);
  const proofRow = variant && snap.proofs.filter((p: any) => p.variant_id === variant.id).slice(-1)[0];
  const commit = snap.commits.find((c: any) => c.round_n === n);
  const [diff, setDiff] = useState("");
  const [src, setSrc] = useState("");

  useEffect(() => {
    if (!variant) return;
    api.diff(variant.id).then(setDiff).catch(() => setDiff(""));
    api.source(variant.id).then(setSrc).catch(() => setSrc(""));
  }, [variant?.id]);

  if (!round) return <div className="page"><p>No such round.</p></div>;

  return (
    <div className="page">
      <h2>Round {n} — {round.status}</h2>

      <section className="panel">
        <h3>1 · Profile</h3>
        {profile ? (
          <>
            <MetricTable rows={[
              ["p50_ms", profile.p50_ms], ["p90_ms", profile.p90_ms], ["sol_pct", profile.sol_pct],
              ["eff_bw_gbps", profile.eff_bw_gbps], ["tflops", profile.tflops],
              ["bound", profile.bound], ["numeric", profile.numeric],
              ["regs/thread", profile.regs_per_thread], ["smem_bytes", profile.smem_bytes],
              ["occupancy_pct", profile.occupancy_pct], ["limiter", profile.occupancy_limiter],
              ["grid", profile.grid], ["block", profile.block],
              ["speedup", profile.baseline?.speedup ?? "n/a"],
              ["simulated", String(profile.simulated)], ["counters", profile.counters],
            ]} />
            <div className="roofline">
              <div className="roofline-fill" style={{ width: `${Math.min(100, profile.sol_pct)}%` }} />
            </div>
          </>
        ) : <p className="muted">no profile recorded for this round</p>}
      </section>

      <section className="panel">
        <h3>2 · Diagnose</h3>
        <div className="cards">
          {pitfalls?.length ? pitfalls.map((b: any, i: number) => <BottleneckCard key={i} b={b} />)
            : <p className="muted">no bottleneck rule fired</p>}
        </div>
      </section>

      <section className="panel">
        <h3>3 · Research</h3>
        <div className="cards">
          {hyps.length ? hyps.map((h: any) => <HypothesisCard key={h.id} h={h} />)
            : <p className="muted">no hypotheses recorded</p>}
        </div>
      </section>

      <section className="panel">
        <h3>4 · Optimize</h3>
        <p className="muted">transform: <code>{variant?.transform_id ?? "baseline"}</code> · level {variant?.round_n}</p>
        <DiffView patch={diff} />
        <details><summary>rendered kernel source</summary><pre className="code">{src}</pre></details>
      </section>

      <section className="panel">
        <h3>5 · Verify</h3>
        <p><ProofBadge status={proofRow?.status} /> model <code>{proofRow?.model}</code></p>
        {proofRow?.open_goals?.length ? (
          <ul>{proofRow.open_goals.map((g: string, i: number) => <li key={i}><code>{g}</code></li>)}</ul>
        ) : <p className="muted">no open goals</p>}
        <p className="muted">axiom audit: <code>{JSON.stringify(proofRow?.axiom_audit)}</code></p>
        {variant && <a href={`#/proof/${variant.id}`}>open proof & Lean source →</a>}
      </section>

      <section className="panel">
        <h3>6 · Ledger</h3>
        <p>commit <code>{commit?.sha?.slice(0, 12) ?? "—"}</code></p>
        <p className="muted">{commit?.message ?? "no commit for this round"}</p>
      </section>
    </div>
  );
}
