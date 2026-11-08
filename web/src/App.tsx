import React, { useCallback, useEffect, useState } from "react";
import { api, Snapshot } from "./api";
import { RunControl } from "./pages/RunControl";
import { RoundDetail } from "./pages/RoundDetail";
import { ProofView } from "./pages/ProofView";
import { GraphPage } from "./pages/GraphPage";
import { LogPage } from "./pages/LogPage";

function useHash(): string {
  const [hash, setHash] = useState(window.location.hash || "#/");
  useEffect(() => {
    const on = () => setHash(window.location.hash || "#/");
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return hash;
}

export default function App() {
  const hash = useHash();
  const [runId, setRunId] = useState<string | null>(localStorage.getItem("harness.run"));
  const [snap, setSnap] = useState<Snapshot | null>(null);
  const [live, setLive] = useState<any[]>([]);

  const refresh = useCallback(() => {
    if (!runId) return;
    api.run(runId).then(setSnap).catch(() => setSnap(null));
  }, [runId]);

  useEffect(() => {
    if (!runId) return;
    refresh();
    const close = api.stream(runId, (ev) => {
      setLive((prev) => [...prev.slice(-400), ev]);
      if (ev.type === "done") refresh();
    });
    const timer = window.setInterval(refresh, 3000);
    return () => { close(); window.clearInterval(timer); };
  }, [runId, refresh]);

  const nav = (id: string, label: string) => (
    <a className={hash.startsWith(`#/${id}`) ? "nav-active" : ""} href={`#/${id}`}>{label}</a>
  );

  let body: React.ReactNode = null;
  const roundMatch = hash.match(/^#\/round\/(\d+)/);
  const proofMatch = hash.match(/^#\/proof\/([^/]+)/);

  if (roundMatch && snap) body = <RoundDetail snap={snap} n={+roundMatch[1]} />;
  else if (proofMatch) body = <ProofView variant={decodeURIComponent(proofMatch[1])} />;
  else if (hash.startsWith("#/graph")) body = <GraphPage snap={snap} />;
  else if (hash.startsWith("#/log")) body = <LogPage snap={snap} live={live} />;
  else body = <RunControl runId={runId} setRunId={setRunId} snap={snap} refresh={refresh} />;

  return (
    <div className="app">
      <header>
        <b className="brand">kernel discovery harness</b>
        <nav>
          <a className={hash === "#/" || hash === "" ? "nav-active" : ""} href="#/">run</a>
          {nav("graph", "graph")}
          {nav("log", "log")}
        </nav>
        <span className="muted">{snap ? `${snap.kernel_id} · ${snap.status}` : "no run"}</span>
      </header>
      <main>{body}</main>
      <footer>
        proof model: PTX small-step + rf/co/sw/causality + observer reads ·
        numerical_scope: <code>exact real arithmetic; IEEE rounding not modeled</code>
      </footer>
    </div>
  );
}
