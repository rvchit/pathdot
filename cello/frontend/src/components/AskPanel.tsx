import React, { useEffect, useRef, useState } from "react";
import axios from "axios";
import { BACKEND_URL, errorMessage } from "../api/client";
import { Heatmap, HeatmapTile, VqaResponse, askTile, mlHealth } from "../api/mlApi";
import { DECISION_STYLE, TileSelection } from "./HeatmapOverlay";

/** One answered question in this browser session. */
export interface AskEntry extends VqaResponse {
  imageId: string;
  level: number;
  x: number;
  y: number;
  at: number;
}

interface Props {
  imageId?: string;
  selection: TileSelection | null;
  heatmap: Heatmap | null;
  history: AskEntry[];
  onAnswer: (entry: AskEntry) => void;
}

const PRESETS = ["Is tumor present?", "What tissue is this?", "Is there necrosis?", "What does this image show?", "Are mitotic figures present?"];

const UNAVAILABLE = "assistant unavailable";

const AskPanel: React.FC<Props> = ({ imageId, selection, heatmap, history, onAnswer }) => {
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [last, setLast] = useState<AskEntry | null>(null);
  /** undefined = health not probed yet; null = endpoint not configured. */
  const [model, setModel] = useState<string | null | undefined>(undefined);
  const [open, setOpen] = useState(true);
  const abortRef = useRef<AbortController | null>(null);
  const rootRef = useRef<HTMLElement | null>(null);

  // Probe /ml/health for the assistant model; re-check periodically so a late-starting service shows up.
  useEffect(() => {
    let alive = true;
    const check = () =>
      mlHealth()
        .then((h) => alive && setModel(h.vqa ?? null))
        .catch(() => alive && setModel(null));
    check();
    const t = setInterval(check, 20_000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  // /?ask=1 -> make sure the panel is expanded and in view (demo deep link).
  useEffect(() => {
    if (new URLSearchParams(window.location.search).get("ask") === "1") {
      setOpen(true);
      setTimeout(() => rootRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 400);
    }
  }, []);

  // New tile or slide -> drop the stale answer.
  useEffect(() => {
    abortRef.current?.abort();
    setBusy(false);
    setError(null);
    setLast(null);
  }, [imageId, selection?.level, selection?.x, selection?.y]);

  const available = model !== null;
  const canAsk = !!imageId && !!selection && available && !busy;

  // The heatmap cell for this tile, if the selection is at the heatmap's level.
  const tile: HeatmapTile | null =
    selection?.tile ?? (heatmap && selection && heatmap.level === selection.level ? heatmap.tiles.find((t) => t.x === selection.x && t.y === selection.y) ?? null : null);

  const ask = async (q: string) => {
    const text = q.trim();
    if (!text || !imageId || !selection || !available || busy) return;
    setQuestion(text);
    setError(null);
    setBusy(true);
    const ac = new AbortController();
    abortRef.current = ac;
    const sel = selection;
    try {
      const res = await askTile({ image_id: imageId, level: sel.level, x: sel.x, y: sel.y, question: text });
      if (ac.signal.aborted) return;
      const entry: AskEntry = { ...res, imageId, level: sel.level, x: sel.x, y: sel.y, at: Date.now() };
      setLast(entry);
      onAnswer(entry);
    } catch (err) {
      if (ac.signal.aborted) return;
      const status = axios.isAxiosError(err) ? err.response?.status : undefined;
      if (status === 503) {
        setModel(null);
        setError(UNAVAILABLE);
      } else if (status === 404) {
        setError(`No tile at level ${sel.level}, (${sel.x}, ${sel.y}) for this slide.`);
      } else if (status === 502) {
        setError(`The assistant model failed to answer: ${errorMessage(err)}`);
      } else {
        setError(`Could not ask the assistant: ${errorMessage(err)}`);
      }
    } finally {
      if (!ac.signal.aborted) setBusy(false);
    }
  };

  const thumb = imageId && selection ? `${BACKEND_URL}/osd/tile/${encodeURIComponent(imageId)}/${selection.level}/${selection.x}/${selection.y}` : null;
  const style = tile ? DECISION_STYLE[tile.decision] : null;
  const recent = history.slice(0, 5);

  return (
    <section className="panel ask-panel" ref={rootRef}>
      <header className="panel-head">
        <h2>Ask the pathology assistant</h2>
        <span className="panel-head-right">
          <span className={`status-dot ${model === undefined ? "" : available ? "ok" : "down"}`} title={available ? `PathVQA: ${model ?? "probing…"}` : "PathVQA endpoint not configured"} />
          <button className="btn btn-ghost btn-sm" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
            {open ? "hide" : "show"}
          </button>
        </span>
      </header>

      {open && (
        <>
          {model === null && <div className="notice warn">{UNAVAILABLE} — the ML service reports no PathVQA model (<code>GET /ml/health</code> → <code>vqa: null</code>).</div>}

          {!selection || !thumb ? (
            <div className="ask-empty">Click a tile on the slide to ask about it</div>
          ) : (
            <div className="ask-tile">
              <img className="ask-thumb" src={thumb} alt={`tile ${selection.x},${selection.y}`} width={96} height={96} />
              <div className="ask-tile-meta">
                <div>
                  tile <strong>{selection.x},{selection.y}</strong> <span className="muted">· level {selection.level}</span>
                </div>
                {tile && style ? (
                  <div className="ask-tile-triage">
                    <span className="tt-dot" style={{ background: `rgb(${style.rgb})` }} />
                    <strong>{style.label}</strong>
                    <span className="muted">p_tumor {tile.p_tumor.toFixed(3)}</span>
                  </div>
                ) : (
                  <div className="muted">{heatmap ? "not scored by triage" : "no triage result yet"}</div>
                )}
              </div>
            </div>
          )}

          <div className="chips">
            {PRESETS.map((p) => (
              <button key={p} className={`chip ${question === p ? "chip-active" : ""}`} onClick={() => ask(p)} disabled={!canAsk} title={p}>
                {p}
              </button>
            ))}
          </div>

          <div className="ask-row">
            <input
              className="ask-input"
              type="text"
              value={question}
              placeholder={selection ? "Ask anything about this tile…" : "Select a tile first"}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") ask(question);
              }}
              disabled={!selection || !available || busy}
            />
            <button className="btn btn-primary" onClick={() => ask(question)} disabled={!canAsk || !question.trim()}>
              {busy ? "Asking…" : "Send"}
            </button>
          </div>

          {error && <div className="notice error">{error}</div>}

          {last && (
            <div className="ask-answer">
              <div className="ask-q muted">{last.question}</div>
              <div className="ask-a">{last.answer}</div>
              <div className="ask-meta">
                {Math.round(last.latency_ms)} ms · ${last.cost_usd.toFixed(4)}
                {last.cached ? " · cached" : ""}
                <span className="ask-model">{last.model}</span>
              </div>
            </div>
          )}

          {recent.length > 0 && (
            <ul className="ask-history">
              {recent.map((h) => (
                <li key={h.at}>
                  <span className="ask-h-q">{h.question}</span>
                  <span className="ask-h-a">{h.answer}</span>
                  <span className="ask-h-meta">
                    {h.imageId !== imageId ? `${h.imageId} · ` : ""}tile {h.x},{h.y} · {Math.round(h.latency_ms)} ms
                  </span>
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </section>
  );
};

export default AskPanel;
