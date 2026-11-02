import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import axios from "axios";
import { BACKEND_URL, errorMessage } from "../api/client";
import { Gallery, GalleryAskItem, GalleryQuestion, GalleryScanItem, GalleryVqaResponse, MlHealth, ScanResult, askGallery, getGallery, mlHealth, scanTile } from "../api/mlApi";

const VLM_NAME = "fine-tuned Qwen3.5-9B";
const PROBE_NAME = "Phikon-v2 probe";

const imgUrl = (rel: string) => `${BACKEND_URL}/ml${rel}`;

/** lowercase, strip punctuation, collapse whitespace. */
const norm = (s: string) =>
  s
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();

const answersMatch = (got: string, expected: string, qtype: GalleryQuestion["qtype"]): boolean => {
  const a = norm(got);
  const b = norm(expected);
  if (!a || !b) return false;
  if (qtype === "closed") return a.split(" ")[0] === b.split(" ")[0];
  return a === b;
};

interface AskEntry extends GalleryVqaResponse {
  expected: string | null;
  qtype: GalleryQuestion["qtype"];
  at: number;
}

/* ---------------- header ---------------- */

const HealthPill: React.FC<{ health: MlHealth | null | undefined }> = ({ health }) => {
  const up = !!health;
  const vqa = health?.vqa ?? null;
  const scan = health?.scan_model ?? null;
  return (
    <span className="health-pill" title={health ? JSON.stringify(health) : "ML service unreachable"}>
      <span className={`status-dot ${health === undefined ? "" : up ? "ok" : "down"}`} /> ML service {health === undefined ? "…" : up ? "up" : "down"}
      <span className="sep">|</span>
      <span className={`status-dot ${health === undefined ? "" : vqa ? "ok" : "down"}`} /> assistant {vqa ? "ready" : "unavailable"}
      <span className="sep">|</span>
      <span className={`status-dot ${health === undefined ? "" : scan ? "ok" : "down"}`} /> scan model {scan ? "ready" : "unavailable"}
    </span>
  );
};

/* ---------------- section 1: tumor scan ---------------- */

const ScanSection: React.FC<{ items: GalleryScanItem[] }> = ({ items }) => {
  const [selected, setSelected] = useState<GalleryScanItem | null>(null);
  const [results, setResults] = useState<Record<string, ScanResult>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const res = selected ? results[selected.id] : undefined;

  const scan = async () => {
    if (!selected || busy) return;
    const id = selected.id;
    setBusy(true);
    setError(null);
    try {
      const r = await scanTile(id);
      setResults((m) => ({ ...m, [id]: r }));
    } catch (err) {
      setError(`Scan failed: ${errorMessage(err)}`);
    } finally {
      setBusy(false);
    }
  };

  const vlmErr = res?.vlm && "error" in res.vlm ? res.vlm.error : null;
  const vlmOk = res?.vlm && !("error" in res.vlm) ? res.vlm : null;
  const correct = res ? res.verdict === res.truth.label_name : false;

  return (
    <section className="demo-section">
      <h2>Tumor scan — fine-tuned Qwen3.5-9B on Camelyon17</h2>
      <p className="demo-explainer">Pick a 96 px lymph-node tile; a fast Phikon-v2 probe screens it and the fine-tuned VLM decides the uncertain band. Ground truth is revealed after the scan.</p>
      <div className="demo-cards">
        {items.map((it, i) => (
          <button key={it.id} className={`demo-card ${selected?.id === it.id ? "selected" : ""}`} onClick={() => { setSelected(it); setError(null); }}>
            <img src={imgUrl(it.image_url)} alt={`Tile ${i + 1}`} width={160} height={160} />
            <span className="demo-card-label">Tile {i + 1}</span>
            <span className="demo-card-sub">
              {it.source}
              {results[it.id] ? ` · ${results[it.id].verdict}` : ""}
            </span>
          </button>
        ))}
      </div>

      {selected && (
        <div className="demo-detail">
          <img className="demo-preview" src={imgUrl(selected.image_url)} alt={selected.id} width={320} height={320} />
          <div className="demo-result">
            <button className="btn btn-primary demo-scan-btn" onClick={scan} disabled={busy}>
              {busy ? "Scanning…" : res ? "Scan again" : "Scan for tumor"}
            </button>
            {error && <div className="notice error">{error}</div>}
            {res && (
              <>
                <div className={`verdict ${res.verdict}`}>
                  {res.verdict === "tumor" ? "TUMOR DETECTED" : "NO TUMOR"}
                  <span className="verdict-conf">{(res.confidence * 100).toFixed(0)}% confidence</span>
                </div>
                <div className="demo-line">
                  decided by: <strong>{res.decided_by === "vlm" ? `${VLM_NAME} (vlm)` : PROBE_NAME}</strong>
                  {res.cached ? " · cached" : ""}
                </div>
                <div className="demo-line">
                  {Math.round(res.latency_ms)} ms · ${res.cost_usd.toFixed(4)} · p_tumor {res.p_tumor.toFixed(3)} · <span title={res.model}>{res.model}</span>
                </div>
                {vlmErr && <div className="notice warn">VLM unavailable — probe decided ({vlmErr})</div>}
                {res.vlm && (
                  <table className="opinion-table">
                    <thead>
                      <tr>
                        <th>opinion</th>
                        <th>verdict</th>
                        <th>p_tumor</th>
                        <th>latency</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>fine-tuned VLM</td>
                        {vlmOk ? (
                          <>
                            <td className={`v-${vlmOk.verdict}`}>{vlmOk.verdict}</td>
                            <td>{vlmOk.p_tumor.toFixed(3)}</td>
                            <td>{Math.round(vlmOk.latency_ms)} ms</td>
                          </>
                        ) : (
                          <td colSpan={3} className="muted">error</td>
                        )}
                      </tr>
                      <tr>
                        <td>probe</td>
                        {res.probe ? (
                          <>
                            <td className={`v-${res.probe.verdict}`}>{res.probe.verdict}</td>
                            <td>{res.probe.p_tumor.toFixed(3)}</td>
                            <td>{Math.round(res.probe.latency_ms)} ms</td>
                          </>
                        ) : (
                          <td colSpan={3} className="muted">—</td>
                        )}
                      </tr>
                    </tbody>
                  </table>
                )}
                <div className={`truth-line ${correct ? "ok" : "bad"}`}>
                  Ground truth: <strong>{res.truth.label_name}</strong> {correct ? "✓" : "✗"}
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </section>
  );
};

/* ---------------- section 2: ask the assistant ---------------- */

const AskSection: React.FC<{ items: GalleryAskItem[]; available: boolean; onUnavailable: () => void }> = ({ items, available, onUnavailable }) => {
  const [selected, setSelected] = useState<GalleryAskItem | null>(null);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<Record<string, AskEntry[]>>({});

  const entries = selected ? history[selected.id] ?? [] : [];
  const last = entries[0];

  const ask = async (q: string, expected: GalleryQuestion | null) => {
    const text = q.trim();
    if (!text || !selected || busy || !available) return;
    const id = selected.id;
    setQuestion(text);
    setBusy(true);
    setError(null);
    try {
      const r = await askGallery(id, text);
      const entry: AskEntry = { ...r, expected: expected?.answer ?? null, qtype: expected?.qtype ?? "open", at: Date.now() };
      setHistory((h) => ({ ...h, [id]: [entry, ...(h[id] ?? [])].slice(0, 5) }));
    } catch (err) {
      const status = axios.isAxiosError(err) ? err.response?.status : undefined;
      if (status === 503) {
        onUnavailable();
        setError("Assistant unavailable");
      } else if (status === 502) {
        setError(`The assistant model failed to answer: ${errorMessage(err)}`);
      } else {
        setError(`Could not ask the assistant: ${errorMessage(err)}`);
      }
    } finally {
      setBusy(false);
    }
  };

  const canAsk = !!selected && available && !busy;

  return (
    <section className="demo-section">
      <h2>Ask the pathology assistant — fine-tuned Qwen3.5-9B on PathVQA</h2>
      <p className="demo-explainer">Pick an image, then ask one of its dataset questions (or your own). The model&apos;s answer is compared with the dataset&apos;s expected answer.</p>
      {!available && <div className="notice warn">Assistant unavailable — the ML service reports no PathVQA model (<code>GET /ml/health</code> → <code>vqa: null</code>).</div>}
      <div className="demo-cards">
        {items.map((it, i) => (
          <button key={it.id} className={`demo-card ${selected?.id === it.id ? "selected" : ""}`} onClick={() => { setSelected(it); setQuestion(""); setError(null); }}>
            <img src={imgUrl(it.image_url)} alt={`Image ${i + 1}`} width={160} height={160} />
            <span className="demo-card-label">Image {i + 1}</span>
            <span className="demo-card-sub">{it.questions.length} {it.questions.length === 1 ? "question" : "questions"}</span>
          </button>
        ))}
      </div>

      {selected && (
        <div className="demo-detail">
          <img className="demo-preview" src={imgUrl(selected.image_url)} alt={selected.id} width={320} height={320} />
          <div className="demo-result">
            <div className="chips demo-chips">
              {selected.questions.map((q) => (
                <button key={q.question} className={`chip ${question === q.question ? "chip-active" : ""}`} onClick={() => ask(q.question, q)} disabled={!canAsk} title={`${q.qtype} question`}>
                  {q.question}
                </button>
              ))}
            </div>
            <div className="ask-row">
              <input
                className="ask-input"
                type="text"
                value={question}
                placeholder="Ask anything about this image…"
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") ask(question, selected.questions.find((q) => q.question === question) ?? null);
                }}
                disabled={!available || busy}
              />
              <button className="btn btn-primary" onClick={() => ask(question, selected.questions.find((q) => q.question === question) ?? null)} disabled={!canAsk || !question.trim()}>
                {busy ? "Asking…" : "Send"}
              </button>
            </div>
            {error && <div className="notice error">{error}</div>}
            {last && (
              <div className="demo-answer">
                <div className="ask-q muted">{last.question}</div>
                <div className="demo-answer-a">{last.answer}</div>
                {last.expected !== null && (
                  <div className="demo-expected">
                    expected (dataset): <strong>{last.expected}</strong>{" "}
                    {answersMatch(last.answer, last.expected, last.qtype) ? <span className="match">✓ match</span> : <span className="nomatch">○ differs</span>}
                  </div>
                )}
                <div className="ask-meta">
                  {Math.round(last.latency_ms)} ms · ${last.cost_usd.toFixed(4)}
                  {last.cached ? " · cached" : ""}
                  <span className="ask-model">{last.model}</span>
                </div>
              </div>
            )}
            {entries.length > 1 && (
              <ul className="demo-history">
                {entries.slice(1).map((h) => (
                  <li key={h.at}>
                    <span className="ask-h-q">{h.question}</span>
                    <span className="ask-h-a">
                      {h.answer}
                      {h.expected !== null ? <span className="muted"> · expected: {h.expected} {answersMatch(h.answer, h.expected, h.qtype) ? "✓" : "○"}</span> : null}
                    </span>
                    <span className="ask-h-meta">{Math.round(h.latency_ms)} ms</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      )}
    </section>
  );
};

/* ---------------- page ---------------- */

const DemoPage: React.FC = () => {
  /** undefined = probing; null = unreachable. */
  const [health, setHealth] = useState<MlHealth | null | undefined>(undefined);
  const [gallery, setGallery] = useState<Gallery | null>(null);
  const [galleryError, setGalleryError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const check = () =>
      mlHealth()
        .then((h) => alive && setHealth(h))
        .catch(() => alive && setHealth(null));
    check();
    const t = setInterval(check, 20_000);
    getGallery()
      .then((g) => alive && setGallery(g))
      .catch((err) => alive && setGalleryError(errorMessage(err)));
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  const assistantUp = !!health?.vqa;

  return (
    <div className="demo-page">
      <header className="demo-header">
        <div>
          <h1>pathdot — AI for the pathologist&apos;s day</h1>
          <div className="demo-tagline">Two fine-tuned Qwen3.5-9B models: one screens lymph-node tiles for tumor, one answers questions about pathology images.</div>
        </div>
        <div className="demo-header-right">
          <HealthPill health={health} />
          <Link to="/" className="demo-link">Open slide viewer →</Link>
        </div>
      </header>

      {galleryError && (
        <div className="notice error">
          Could not load the demo gallery (<code>GET /ml/gallery</code>): {galleryError}
        </div>
      )}
      {!gallery && !galleryError && <div className="notice">Loading gallery…</div>}

      {gallery && <ScanSection items={gallery.scan} />}
      {gallery && <AskSection items={gallery.ask} available={assistantUp} onUnavailable={() => setHealth((h) => (h ? { ...h, vqa: null } : h))} />}
    </div>
  );
};

export default DemoPage;
