import React, { useEffect, useState } from "react";
import { api } from "../api";
import {
  EventFeed, MetricTable, SolGauge, Sparkline, Timeline,
} from "../components/primitives";

export function RunControl({ runId, setRunId, snap, refresh }: {
  runId: string | null; setRunId: (id: string) => void; snap: any; refresh: () => void;
}) {
  const [kernels, setKernels] = useState<any[]>([]);
  const [kernel, setKernel] = useState("elementwise_affine");
  const [gpu, setGpu] = useState("H100");
  const [backend, setBackend] = useState("modal");
  const [sol, setSol] = useState(60);
  const [speedup, setSpeedup] = useState(1.5);
  const [maxRounds, setMaxRounds] = useState(3);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { api.registry().then((r) => setKernels(r.kernels)).catch((e) => setError(String(e))); }, []);

  const start = async () => {
    setError(null);
    try {
      const { run_id } = await api.start({
        kernel_id: kernel, gpu, backend,
        sla: { sol_pct: sol, speedup, max_rounds: maxRounds },
      });
      localStorage.setItem("harness.run", run_id);
      setRunId(run_id);
    } catch (e) { setError(String(e)); }
  };

  const profiles = snap?.profiles ?? [];
  const solSeries = profiles.map((p: any) => p.profile.sol_pct);
  const p50Series = profiles.map((p: any) => p.profile.p50_ms);
  const sla = snap?.config?.sla?.sol_pct ?? sol;
  const latest = profiles.length ? profiles[profiles.length - 1].profile : null;

  return (
    <div className="page">
      <section className="panel">
        <h2>Run control</h2>
        <div className="form">
          <label>Kernel
            <select value={kernel} onChange={(e) => setKernel(e.target.value)}>
              {kernels.map((k) => <option key={k.id} value={k.id}>{k.id} · {k.family}</option>)}
            </select>
          </label>
          <label>GPU
            <select value={gpu} onChange={(e) => setGpu(e.target.value)}>
              {["H100", "H200", "B200", "A100", "L4"].map((g) => <option key={g}>{g}</option>)}
            </select>
          </label>
          <label>Backend
            <select value={backend} onChange={(e) => setBackend(e.target.value)}>
              <option value="modal">modal</option>
              <option value="sim">sim (cannot meet SLA)</option>
            </select>
          </label>
          <label>SOL % <input type="number" value={sol} onChange={(e) => setSol(+e.target.value)} /></label>
          <label>Speedup <input type="number" step="0.1" value={speedup} onChange={(e) => setSpeedup(+e.target.value)} /></label>
          <label>Max rounds <input type="number" value={maxRounds} onChange={(e) => setMaxRounds(+e.target.value)} /></label>
        </div>
        <div className="row">
          <button onClick={start}>Start run</button>
          {runId && <button className="ghost" onClick={() => api.stop(runId)}>Stop</button>}
          {runId && <span className="muted">run {runId}</span>}
        </div>
        {error && <div className="error">{error}</div>}
      </section>

      {snap && (
        <section className="panel">
          <h2>{snap.kernel_id} — {snap.status}{snap.stop_reason ? ` (${snap.stop_reason})` : ""}</h2>
          <div className="row">
            {latest && <SolGauge value={latest.sol_pct} sla={sla} />}
            <Sparkline values={solSeries} label="sol %" />
            <Sparkline values={p50Series} label="p50" unit=" ms" />
          </div>
          <Timeline rounds={snap.rounds} onSelect={(n) => { window.location.hash = `#/round/${n}`; }} />
          {latest && (
            <MetricTable rows={[
              ["p50_ms", latest.p50_ms], ["sol_pct", latest.sol_pct], ["bound", latest.bound],
              ["numeric", latest.numeric], ["simulated", String(latest.simulated)],
              ["regs", latest.regs_per_thread], ["occupancy %", latest.occupancy_pct],
              ["counters", latest.counters],
            ]} />
          )}
          <p className="badge-line">
            numerical_scope: <code>exact real arithmetic; IEEE rounding not modeled</code>
          </p>
          <h3>Event feed</h3>
          <EventFeed events={snap.events.slice(-40)} />
        </section>
      )}
    </div>
  );
}
