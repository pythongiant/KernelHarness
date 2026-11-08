import React from "react";

export function GraphPage({ snap }: { snap: any }) {
  if (!snap) return <div className="page"><p>Start a run first.</p></div>;
  const profiles = new Map<string, any>();
  snap.profiles.forEach((p: any) => profiles.set(p.variant_id, p.profile));
  const variants = snap.variants.slice().sort((a: any, b: any) => a.round_n - b.round_n);
  const w = 220, h = 90, gap = 40;
  const width = Math.max(640, variants.length * (w + gap));
  return (
    <div className="page">
      <h2>Provenance</h2>
      <svg viewBox={`0 0 ${width} ${h + 60}`} width="100%" style={{ maxWidth: width }}>
        {variants.map((v: any, i: number) => {
          const p = profiles.get(v.id);
          const sol = p?.sol_pct ?? 0;
          const fill = sol >= 60 ? "var(--success)" : sol >= 35 ? "var(--warning)" : "var(--error)";
          const x = 20 + i * (w + gap);
          return (
            <g key={v.id}>
              {i > 0 && (
                <line x1={x - gap} y1={45} x2={x} y2={45} stroke="var(--border)" strokeWidth="2" />
              )}
              {i > 0 && (
                <text x={x - gap + 4} y={36} fontSize="10" fill="var(--muted)">
                  {v.transform_id ?? ""}
                </text>
              )}
              <a href={`#/round/${v.round_n}`}>
                <rect x={x} y={20} width={w} height={55} rx="6" fill={fill} opacity="0.14"
                      stroke="var(--border)" />
                <text x={x + 10} y={40} fontSize="12" fill="var(--fg)">
                  round {v.round_n} · {v.transform_id ?? "baseline"}
                </text>
                <text x={x + 10} y={58} fontSize="12" fill="var(--muted)">
                  {sol.toFixed(1)}% SOL · {p?.p50_ms?.toFixed(3) ?? "—"} ms
                </text>
              </a>
            </g>
          );
        })}
      </svg>
      <p className="badge-line">nodes are rounds; edge labels are transforms; fill encodes SOL band.</p>
    </div>
  );
}
