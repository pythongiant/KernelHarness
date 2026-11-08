import React from "react";

export function LogPage({ snap, live }: { snap: any; live: any[] }) {
  const rows = [...(snap?.events ?? []), ...live];
  return (
    <div className="page">
      <h2>Event log</h2>
      <pre className="code log">
        {rows.map((e: any, i: number) => (
          <div key={e.id ?? `l${i}`}>
            [{String(e.stage ?? "-").padEnd(9)}] {String(e.kind).padEnd(11)} {e.title}
            {e.body_md ? `\n${e.body_md}\n` : ""}
          </div>
        ))}
      </pre>
    </div>
  );
}
