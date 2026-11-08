import React, { useEffect, useState } from "react";
import { api } from "../api";
import { ProofBadge } from "../components/primitives";

export function ProofView({ variant }: { variant: string }) {
  const [proof, setProof] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.proof(variant).then(setProof).catch((e) => setError(String(e)));
  }, [variant]);

  if (error) return <div className="page"><p className="error">{error}</p></div>;
  if (!proof) return <div className="page"><p className="muted">loading…</p></div>;

  const src = Object.values(proof.sources ?? {})[0] as string | undefined;

  return (
    <div className="page">
      <h2>Proof · <code>{variant}</code> <ProofBadge status={proof.status} /></h2>
      <section className="panel">
        <p>model <code>{proof.model}</code></p>
        <p className="muted">numerical_scope: <code>{proof.numerical_scope}</code></p>
        <p>open goals: {proof.open_goals?.length ? (
          <ul>{proof.open_goals.map((g: string, i: number) => <li key={i}><code>{g}</code></li>)}</ul>
        ) : "none"}</p>
        <p>axiom audit: <code>{JSON.stringify(proof.axiom_audit)}</code></p>
      </section>
      <section className="panel">
        <h3>Lean module</h3>
        <pre className="code">
          {(src ?? "").split("\n").map((line, i) => (
            <div key={i} className={
              /sorry|admit/.test(line) ? "lean-bad" : /theorem|run_cmd|#eval/.test(line) ? "lean-key" : ""
            }>{line || " "}</div>
          ))}
        </pre>
      </section>
    </div>
  );
}
