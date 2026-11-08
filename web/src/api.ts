export type Profile = Record<string, any>;
export type Snapshot = {
  id: string; kernel_id: string; status: string; stop_reason: string | null;
  config: any; rounds: any[]; events: any[]; variants: any[]; profiles: any[];
  bottlenecks: any[]; hypotheses: any[]; proofs: any[]; commits: any[];
};

async function j<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return (await r.json()) as T;
}

export const api = {
  registry: () => j<{ kernels: any[] }>("/api/registry"),
  runs: () => j<{ runs: any[] }>("/api/runs"),
  run: (id: string) => j<Snapshot>(`/api/runs/${id}`),
  start: (body: any) =>
    j<{ run_id: string }>("/api/runs", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    }),
  stop: (id: string) => j(`/api/runs/${id}/stop`, { method: "POST" }),
  proof: (variant: string) => j<any>(`/api/variants/${variant}/proof`),
  diff: (variant: string) => fetch(`/api/variants/${variant}/diff`).then((r) => r.text()),
  source: (variant: string) => fetch(`/api/variants/${variant}/source`).then((r) => r.text()),
  report: (id: string) => fetch(`/api/runs/${id}/report`).then((r) => r.text()),
  stream: (id: string, onEvent: (ev: any) => void) => {
    const es = new EventSource(`/api/runs/${id}/stream`);
    const handler = (e: MessageEvent) => {
      try { onEvent({ ...JSON.parse(e.data), type: e.type }); } catch { onEvent({ type: e.type }); }
    };
    for (const name of ["stage_start", "stage_end", "reasoning", "measurement", "decision",
                        "artifact", "error", "log", "heartbeat", "done"]) {
      es.addEventListener(name, handler as EventListener);
    }
    return () => es.close();
  },
};

export function currentRun(): string | null {
  const m = window.location.hash.match(/^#\/run\/([^/]+)/);
  if (m) return m[1];
  return localStorage.getItem("harness.run");
}
