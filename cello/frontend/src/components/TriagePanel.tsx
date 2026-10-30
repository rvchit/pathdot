import React, { useEffect, useRef, useState } from "react";
import { errorMessage } from "../api/client";
import { Fallback, Heatmap, getHeatmap, mlHealth, runSlideTriage } from "../api/mlApi";
import { OverlayOptions } from "./HeatmapOverlay";

interface Props {
  imageId?: string;
  heatmap: Heatmap | null;
  onHeatmap: (hm: Heatmap | null) => void;
  overlay: OverlayOptions;
  onOverlay: (o: OverlayOptions) => void;
  hasGroundTruth: boolean;
}

const fmtUsd = (v: number) => (v < 0.01 ? `$${v.toFixed(4)}` : `$${v.toFixed(2)}`);
const fmtMs = (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(1)} s` : `${Math.round(v)} ms`);
const pct = (v: number) => `${(v * 100).toFixed(0)}%`;

const TriagePanel: React.FC<Props> = ({ imageId, heatmap, onHeatmap, overlay, onOverlay, hasGroundTruth }) => {
  const [low, setLow] = useState(0.05);
  const [high, setHigh] = useState(0.95);
  const [fallback, setFallback] = useState<Fallback>("vlm_ft");
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [mlUp, setMlUp] = useState<boolean | null>(null);
  const [mlInfo, setMlInfo] = useState<string>("");
  const abortRef = useRef<AbortController | null>(null);

  // Probe the ML service once on mount and after each failure.
  const checkHealth = async () => {
    try {
      const h = await mlHealth();
      setMlUp(true);
      setMlInfo([h.encoder, h.probe ? `probe ${h.probe}` : null, h.vlm ? `vlm ${h.vlm}` : null, h.device].filter(Boolean).join(" · "));
    } catch {
      setMlUp(false);
      setMlInfo("");
    }
  };
  useEffect(() => {
    checkHealth();
    const t = setInterval(checkHealth, 15_000);
    return () => clearInterval(t);
  }, []);

  // Clear results when the slide changes.
  useEffect(() => {
    abortRef.current?.abort();
    setRunning(false);
    setError(null);
    setProgress(0);
  }, [imageId]);

  // /?image=<id>&triage=1 -> run once automatically (demo deep link).
  const autoRan = useRef(false);
  useEffect(() => {
    if (autoRan.current || !imageId) return;
    if (new URLSearchParams(window.location.search).get("triage") !== "1") return;
    autoRan.current = true;
    const t = setTimeout(() => run(), 300);
    return () => clearTimeout(t);
  }, [imageId]); // run() reads the latest slider state; intentionally only re-armed per slide

  const run = async () => {
    if (!imageId || running) return;
    setError(null);
    setRunning(true);
    setProgress(0);
    const ac = new AbortController();
    abortRef.current = ac;
    try {
      const hm = await runSlideTriage({ image_id: imageId, low, high, fallback }, setProgress, { signal: ac.signal });
      if (!ac.signal.aborted) onHeatmap(hm);
    } catch (err) {
      if (!ac.signal.aborted) {
        setError(errorMessage(err));
        checkHealth();
      }
    } finally {
      if (!ac.signal.aborted) setRunning(false);
    }
  };

  const loadLast = async () => {
    if (!imageId) return;
    setError(null);
    try {
      onHeatmap(await getHeatmap(imageId));
    } catch (err) {
      setError(errorMessage(err));
      checkHealth();
    }
  };

  const s = heatmap?.summary;
  const disabled = !imageId || running;

  return (
    <section className="panel">
      <header className="panel-head">
        <h2>AI triage</h2>
        <span className={`status-dot ${mlUp === null ? "" : mlUp ? "ok" : "down"}`} title={mlUp ? `ML service online: ${mlInfo}` : "ML service unreachable"} />
      </header>

      {mlUp === false && (
        <div className="notice warn">
          The ML service is not reachable through the backend. Start it with <code>make dev</code> (or <code>uvicorn pathfrontier.serve:app --port 8008</code>), or run <code>npm run ml-stub</code> in <code>cello/backend</code> for a canned demo.
        </div>
      )}

      <div className="field-row">
        <label>
          auto-clear below <strong>{low.toFixed(2)}</strong>
          <input type="range" min={0} max={0.5} step={0.01} value={low} onChange={(e) => setLow(Math.min(Number(e.target.value), high - 0.01))} disabled={running} />
        </label>
        <label>
          auto-flag above <strong>{high.toFixed(2)}</strong>
          <input type="range" min={0.5} max={1} step={0.01} value={high} onChange={(e) => setHigh(Math.max(Number(e.target.value), low + 0.01))} disabled={running} />
        </label>
        <label>
          uncertain band goes to
          <select value={fallback} onChange={(e) => setFallback(e.target.value as Fallback)} disabled={running}>
            <option value="vlm_ft">fine-tuned VLM</option>
            <option value="pathologist">pathologist</option>
            <option value="none">nobody (probe only)</option>
          </select>
        </label>
      </div>

      <div className="btn-row">
        <button className="btn btn-primary" onClick={run} disabled={disabled}>
          {running ? "Scoring…" : heatmap ? "Re-run AI triage" : "Run AI triage"}
        </button>
        <button className="btn" onClick={loadLast} disabled={disabled} title="GET /ml/heatmap/:id">
          Load last result
        </button>
      </div>

      {running && (
        <div className="progress" aria-label="triage progress">
          <div className="progress-bar" style={{ width: `${Math.round(progress * 100)}%` }} />
          <span className="progress-label">{Math.round(progress * 100)}%</span>
        </div>
      )}
      {error && <div className="notice error">{error}</div>}

      {s && (
        <div className="summary">
          <div className="stat big">
            <span className="stat-value">{pct(s.frac_auto_cleared)}</span>
            <span className="stat-label">auto-cleared</span>
          </div>
          <div className="stat-grid">
            <Stat label="tiles scored" value={s.n_tiles.toLocaleString()} />
            <Stat label="flagged" value={s.auto_flagged.toLocaleString()} tone="pos" />
            <Stat label="sent to review" value={s.review.toLocaleString()} tone="rev" />
            <Stat label="cleared" value={s.auto_cleared.toLocaleString()} tone="neg" />
            <Stat label="est. cost" value={fmtUsd(s.cost_usd)} />
            <Stat label="latency" value={fmtMs(s.latency_ms_total)} />
            <Stat label="pathologist min saved" value={s.pathologist_minutes_saved.toFixed(1)} />
            <Stat label="level / scale" value={`${heatmap.level} / ${heatmap.scale}x`} />
          </div>
          <div className="summary-model">{s.model}</div>
        </div>
      )}

      <div className="overlay-controls">
        <label className="check">
          <input type="checkbox" checked={overlay.showHeatmap} onChange={(e) => onOverlay({ ...overlay, showHeatmap: e.target.checked })} disabled={!heatmap} />
          show heatmap
        </label>
        <label>
          opacity
          <input type="range" min={0} max={1} step={0.05} value={overlay.opacity} onChange={(e) => onOverlay({ ...overlay, opacity: Number(e.target.value) })} disabled={!heatmap} />
        </label>
        <label className="check" title={hasGroundTruth ? "data/slides/<id>.gt.json" : "No ground truth file for this slide"}>
          <input type="checkbox" checked={overlay.showGroundTruth} onChange={(e) => onOverlay({ ...overlay, showGroundTruth: e.target.checked })} disabled={!hasGroundTruth} />
          show ground truth {hasGroundTruth ? "" : "(none)"}
        </label>
        <div className="legend">
          <span>
            <i className="sw pos" /> flagged
          </span>
          <span>
            <i className="sw rev" /> review
          </span>
          <span>
            <i className="sw neg" /> cleared
          </span>
          <span>
            <i className="sw gt" /> truth: tumor
          </span>
        </div>
        {heatmap && (
          <button className="btn btn-ghost" onClick={() => onHeatmap(null)}>
            Clear result
          </button>
        )}
      </div>
    </section>
  );
};

const Stat: React.FC<{ label: string; value: string; tone?: "pos" | "rev" | "neg" }> = ({ label, value, tone }) => (
  <div className={`stat ${tone ?? ""}`}>
    <span className="stat-value">{value}</span>
    <span className="stat-label">{label}</span>
  </div>
);

export default TriagePanel;
