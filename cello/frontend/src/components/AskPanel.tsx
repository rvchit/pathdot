import React, { useEffect, useRef, useState } from "react";
import axios from "axios";
import { BACKEND_URL, errorMessage } from "../api/client";
import { GalleryAskItem, GalleryQuestion, Heatmap, HeatmapTile, VqaResponse, askGallery, askTile, mlHealth } from "../api/mlApi";
import { DECISION_STYLE, TileSelection } from "./HeatmapOverlay";

/** One answered question in this browser session. */
export interface AskEntry extends VqaResponse {
  imageId: string;
  level: number;
  x: number;
  y: number;
  at: number;
  /** True when the question was about the whole demo-gallery image (no tile selected). */
  whole?: boolean;
  /** Dataset reference answer when the question came from the gallery manifest. */
  expected?: string | null;
  qtype?: GalleryQuestion["qtype"];
}

interface Props {
  imageId?: string;
  selection: TileSelection | null;
  heatmap: Heatmap | null;
  history: AskEntry[];
  onAnswer: (entry: AskEntry) => void;
  /** The demo-gallery "ask" item for the open slide, when it is one (dataset questions + whole-image asks). */
  galleryItem?: GalleryAskItem | null;
}

const PRESETS = ["Is tumor present?", "What tissue is this?", "Is there necrosis?", "What does this image show?", "Are mitotic figures present?"];

const UNAVAILABLE = "assistant unavailable";

/** Same normalisation as the /demo page: lowercase, strip punctuation; closed questions compare the first word. */
const norm = (s: string) =>
  s
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();

export const answersMatch = (got: string, expected: string, qtype: GalleryQuestion["qtype"]): boolean => {
  const a = norm(got);
  const b = norm(expected);
  if (!a || !b) return false;
  if (qtype === "closed") return a.split(" ")[0] === b.split(" ")[0];
  return a === b;
};

const AskPanel: React.FC<Props> = ({ imageId, selection, heatmap, history, onAnswer, galleryItem = null }) => {
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
  // Gallery images can be asked about as a whole; other slides need a tile.
  const wholeImage = !!galleryItem && !selection;
  const hasTarget = !!selection || !!galleryItem;
  const canAsk = !!imageId && hasTarget && available && !busy;
  const presets = galleryItem ? galleryItem.questions.map((q) => q.question) : PRESETS;

  // The heatmap cell for this tile, if the selection is at the heatmap's level.
  const tile: HeatmapTile | null =
    selection?.tile ?? (heatmap && selection && heatmap.level === selection.level ? heatmap.tiles.find((t) => t.x === selection.x && t.y === selection.y) ?? null : null);

  const ask = async (q: string) => {
    const text = q.trim();
    if (!text || !imageId || !hasTarget || !available || busy) return;
    setQuestion(text);
    setError(null);
    setBusy(true);
    const ac = new AbortController();
    abortRef.current = ac;
    const sel = selection;
    const ref = galleryItem?.questions.find((q) => q.question.trim().toLowerCase() === text.toLowerCase()) ?? null;
    try {
      let entry: AskEntry;
      if (sel) {
        const res = await askTile({ image_id: imageId, level: sel.level, x: sel.x, y: sel.y, question: text });
        if (ac.signal.aborted) return;
        entry = { ...res, imageId, level: sel.level, x: sel.x, y: sel.y, at: Date.now() };
      } else {
        // Demo-gallery image with no tile selected: ask about the whole image via {gallery_id, question}.
        const res = await askGallery(galleryItem!.id, text);
        if (ac.signal.aborted) return;
        entry = { tokens_in: 0, tokens_out: 0, ...res, imageId, level: 0, x: 0, y: 0, at: Date.now(), whole: true };
      }
      if (ref) {
        entry.expected = ref.answer;
        entry.qtype = ref.qtype;
      }
      setLast(entry);
      onAnswer(entry);
    } catch (err) {
      if (ac.signal.aborted) return;
      const status = axios.isAxiosError(err) ? err.response?.status : undefined;
      if (status === 503) {
        setModel(null);
        setError(UNAVAILABLE);
      } else if (status === 404) {
        setError(sel ? `No tile at level ${sel.level}, (${sel.x}, ${sel.y}) for this slide.` : `Gallery image ${imageId} not found on the ML service.`);
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
            wholeImage ? (
              <div className="ask-empty">
                Asking about the <strong>whole image</strong>
                <div className="ask-whole">PathVQA image · click a tile to ask about a region instead</div>
              </div>
            ) : (
              <div className="ask-empty">Click a tile on the slide to ask about it</div>
            )
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

          <div className={`chips ${galleryItem ? "dataset" : ""}`}>
            {presets.map((p) => (
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
              placeholder={selection ? "Ask anything about this tile…" : wholeImage ? "Ask anything about this image…" : "Select a tile first"}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") ask(question);
              }}
              disabled={!hasTarget || !available || busy}
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
              {last.expected != null && last.qtype && (
                <div className="ask-expected">
                  expected: <strong>{last.expected}</strong>{" "}
                  {answersMatch(last.answer, last.expected, last.qtype) ? <span className="match">✓ matches</span> : <span className="nomatch">○ differs</span>}
                </div>
              )}
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
                    {h.imageId !== imageId ? `${h.imageId} · ` : ""}
                    {h.whole ? "whole image" : `tile ${h.x},${h.y}`} · {Math.round(h.latency_ms)} ms
                    {h.expected != null && h.qtype ? (answersMatch(h.answer, h.expected, h.qtype) ? " · ✓" : " · ○") : ""}
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
