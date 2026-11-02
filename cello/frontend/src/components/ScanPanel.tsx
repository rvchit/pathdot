import React, { useEffect, useRef, useState } from "react";
import { errorMessage } from "../api/client";
import { GalleryScanItem, ScanResult, scanTile } from "../api/mlApi";

interface Props {
  /** The gallery scan item for the open slide (the panel is only rendered when there is one). */
  item: GalleryScanItem;
  onVerdict: (verdict: ScanResult["verdict"] | null) => void;
}

const STEPS = ["Encoding tile", "Fine-tuned Qwen3.5-9B", "Second opinion: Phikon-v2 probe"];
const STEP_MS = 350;

const fmtUsd = (v: number) => (v < 0.01 ? `$${v.toFixed(4)}` : `$${v.toFixed(2)}`);
const fmtMs = (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(1)} s` : `${Math.round(v)} ms`);
const pct = (v: number) => `${(v * 100).toFixed(0)}%`;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** "Scan for tumor" panel for demo-gallery Camelyon17 tiles: POST /ml/scan {id}. */
const ScanPanel: React.FC<Props> = ({ item, onVerdict }) => {
  const [step, setStep] = useState(-1); // -1 idle, 0..2 running, 3 done
  const [result, setResult] = useState<ScanResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [revealed, setRevealed] = useState(false);
  const runRef = useRef(0);

  // New slide -> reset.
  useEffect(() => {
    runRef.current++;
    setStep(-1);
    setResult(null);
    setError(null);
    setRevealed(false);
    onVerdict(null);
  }, [item.id]); // onVerdict is a stable setter; only re-run per slide

  const running = step >= 0 && step < STEPS.length;

  const run = async () => {
    if (running) return;
    const run = ++runRef.current;
    setError(null);
    setResult(null);
    setRevealed(false);
    onVerdict(null);
    setStep(0);
    const req = scanTile(item.id);
    // Walk the three steps with a minimum dwell each so the cascade is legible, while the request runs.
    const settled = req.then((r) => ({ ok: true as const, r })).catch((e: unknown) => ({ ok: false as const, e }));
    for (let i = 0; i < STEPS.length; i++) {
      if (runRef.current !== run) return;
      setStep(i);
      await sleep(STEP_MS);
    }
    const out = await settled;
    if (runRef.current !== run) return;
    if (!out.ok) {
      setStep(-1);
      setError(errorMessage(out.e));
      return;
    }
    setStep(STEPS.length);
    setResult(out.r);
    onVerdict(out.r.verdict);
  };

  const r = result;
  const vlm = r?.vlm && !("error" in r.vlm) ? r.vlm : null;
  const vlmErr = r?.vlm && "error" in r.vlm ? r.vlm.error : null;
  const correct = r ? r.verdict === r.truth.label_name : false;

  return (
    <section className="panel scan-panel">
      <header className="panel-head">
        <h2>Scan for tumor</h2>
        <span className="muted">{item.source === "camelyon17" ? "Camelyon17 tile" : `${item.source} tile`}</span>
      </header>

      <button className="btn btn-primary" onClick={run} disabled={running} title="POST /ml/scan">
        {running ? "Scanning…" : r ? "Scan again" : "Scan for tumor with fine-tuned Qwen3.5-9B"}
      </button>

      {(running || r) && (
        <ul className="scan-steps" aria-label="scan progress">
          {STEPS.map((s, i) => (
            <li key={s} className={i < step ? "done" : i === step ? "active" : ""}>
              <span className="scan-step-dot" />
              {s}
            </li>
          ))}
        </ul>
      )}

      {error && <div className="notice error">{error}</div>}

      {r && (
        <div className="scan-result">
          <div className={`verdict ${r.verdict}`}>
            {r.verdict === "tumor" ? "TUMOR DETECTED" : "NO TUMOR"}
            <span className="verdict-conf">{pct(r.confidence)} confident</span>
          </div>
          <div className="scan-model">
            decided by <strong>{r.decided_by === "vlm" ? "fine-tuned Qwen3.5-9B" : "Phikon-v2 probe"}</strong> · {r.model}
          </div>

          {(vlm || r.probe || vlmErr) && (
            <table className="opinion-table">
              <thead>
                <tr>
                  <th>opinion</th>
                  <th>verdict</th>
                  <th>p(tumor)</th>
                  <th>latency</th>
                </tr>
              </thead>
              <tbody>
                {vlm && (
                  <tr>
                    <td>Qwen3.5-9B (fine-tuned)</td>
                    <td className={`v-${vlm.verdict}`}>{vlm.verdict}</td>
                    <td>{vlm.p_tumor.toFixed(3)}</td>
                    <td>{fmtMs(vlm.latency_ms)}</td>
                  </tr>
                )}
                {vlmErr && (
                  <tr>
                    <td>Qwen3.5-9B (fine-tuned)</td>
                    <td className="muted" colSpan={3}>
                      unavailable: {vlmErr}
                    </td>
                  </tr>
                )}
                {r.probe && (
                  <tr>
                    <td>Phikon-v2 probe</td>
                    <td className={`v-${r.probe.verdict}`}>{r.probe.verdict}</td>
                    <td>{r.probe.p_tumor.toFixed(3)}</td>
                    <td>{fmtMs(r.probe.latency_ms)}</td>
                  </tr>
                )}
              </tbody>
            </table>
          )}

          <div className="scan-model">
            {fmtMs(r.latency_ms)} · {fmtUsd(r.cost_usd)}
            {r.cached ? " · cached" : ""}
          </div>

          {revealed ? (
            <div className={`truth-line ${correct ? "ok" : "bad"}`}>
              ground truth: <strong>{r.truth.label_name}</strong> {correct ? "✓ correct" : "✗ missed"}
            </div>
          ) : (
            <button className="btn" onClick={() => setRevealed(true)}>
              Reveal ground truth
            </button>
          )}
        </div>
      )}

      {!r && !running && <div className="scan-hint">Runs the fine-tuned VLM on this 224 px tile and asks the Phikon-v2 probe for a second opinion.</div>}
    </section>
  );
};

export default ScanPanel;
