import React from "react";

export function SolGauge({ value, sla }: { value: number; sla: number }) {
  const pct = Math.max(0, Math.min(100, value));
  return (
    <div className="gauge">
      <div className="gauge-track">
        <div className="gauge-fill" style={{ width: `${pct}%` }} />
        <div className="gauge-sla" style={{ left: `${Math.min(100, sla)}%` }} />
      </div>
      <div className="gauge-label">
        <span>{value.toFixed(1)}% SOL</span>
        <span className="muted">SLA {sla.toFixed(0)}%</span>
      </div>
    </div>
  );
}

export function Sparkline({ values, label, unit }: { values: number[]; label: string; unit?: string }) {
  if (!values.length) return <div className="muted">{label}: no data</div>;
  const w = 240, h = 44, pad = 4;
  const min = Math.min(...values), max = Math.max(...values);
  const span = max - min || 1;
  const pts = values.map((v, i) => {
    const x = pad + (i * (w - 2 * pad)) / Math.max(1, values.length - 1);
    const y = h - pad - ((v - min) / span) * (h - 2 * pad);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  return (
    <div className="spark">
      <div className="spark-label">
        {label}: <b>{values[values.length - 1].toFixed(3)}{unit ?? ""}</b>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} width={w} height={h} role="img" aria-label={label}>
        <polyline points={pts.join(" ")} fill="none" stroke="var(--accent)" strokeWidth="1.5" />
        {pts.map((p, i) => {
          const [x, y] = p.split(",");
          return <circle key={i} cx={x} cy={y} r="2" fill="var(--accent)" />;
        })}
      </svg>
    </div>
  );
}

export function DiffView({ patch }: { patch: string }) {
  if (!patch.trim()) return <div className="muted">no source change</div>;
  return (
    <pre className="diff">
      {patch.split("\n").map((line, i) => (
        <div key={i} className={
          line.startsWith("+++") || line.startsWith("---") ? "d-meta"
            : line.startsWith("+") ? "d-add"
            : line.startsWith("-") ? "d-del" : "d-ctx"
        }>{line || " "}</div>
      ))}
    </pre>
  );
}

export function ProofBadge({ status }: { status?: string }) {
  const s = status ?? "UNKNOWN";
  const cls = s === "VERIFIED" ? "ok" : s === "PARTIAL" ? "warn"
    : s === "FAILED" ? "bad" : "muted";
  return <span className={`badge ${cls}`}>{s}</span>;
}

export function Timeline({ rounds, onSelect }: { rounds: any[]; onSelect: (n: number) => void }) {
  return (
    <div className="timeline">
      {rounds.map((r) => (
        <button key={r.n} className={`chip ${r.status}`} onClick={() => onSelect(r.n)}>
          r{r.n} · {r.status}
        </button>
      ))}
    </div>
  );
}

export function MetricTable({ rows }: { rows: [string, any][] }) {
  return (
    <table className="metrics">
      <tbody>
        {rows.map(([k, v]) => (
          <tr key={k}>
            <th>{k}</th>
            <td>{typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(4)) : String(v)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function BottleneckCard({ b }: { b: any }) {
  return (
    <div className={`card sev-${b.severity}`}>
      <div className="card-head">
        <b>{b.code}</b>
        <span className="muted">{b.severity}</span>
      </div>
      <p>{b.evidence_md}</p>
    </div>
  );
}

export function HypothesisCard({ h }: { h: any }) {
  return (
    <div className="card">
      <div className="card-head">
        <b>{h.title}</b>
        <span className="muted">{h.source}</span>
      </div>
      <p>{h.rationale_md}</p>
      <p className="muted">expected: {h.expected_effect}</p>
      {h.refs?.length ? <p className="refs">refs: {h.refs.join(" · ")}</p> : null}
    </div>
  );
}

export function EventFeed({ events }: { events: any[] }) {
  return (
    <div className="feed">
      {events.map((e, i) => (
        <div key={e.id ?? i} className={`feed-row kind-${e.kind}`}>
          <span className="feed-stage">{e.stage ?? "—"}</span>
          <span className="feed-kind">{e.kind}</span>
          <span className="feed-title">{e.title}</span>
        </div>
      ))}
    </div>
  );
}
