import React, { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import axios from "axios";
import { BACKEND_URL, errorMessage } from "../api/client";
import { Gallery, GalleryAskItem, GalleryQuestion, GalleryScanItem, GalleryVqaResponse, MlHealth, ScanResult, askGallery, getGallery, mlHealth, scanTile } from "../api/mlApi";

const VLM_NAME = "Fine-tuned Qwen3.5-9B";
const PROBE_NAME = "Phikon-v2 probe";

/** Scan progress steps; each is shown for at least STEP_MS so the sequence reads even on a cached response. */
const SCAN_STEPS = ["Encoding tile", `${VLM_NAME} inference`, `Second opinion: ${PROBE_NAME}`];
const STEP_MS = 420;

/** Pathologist-time assumptions behind the footer counter. */
const SEC_PER_TILE_READ = 2;
const SEC_PER_LOOKUP = 45;

const imgUrl = (rel: string) => `${BACKEND_URL}/ml${rel}`;
const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

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

const median = (xs: number[]): number => {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

interface AskEntry extends GalleryVqaResponse {
  expected: string | null;
  qtype: GalleryQuestion["qtype"];
  at: number;
}

/** Running per-session tallies, lifted to the page so the footer and the ask header share them. */
interface Session {
  scans: number;
  asks: { latency_ms: number; cost_usd: number }[];
}

/** Escape clears the current selection in a section. */
const useEscape = (onEscape: () => void) => {
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (e.key === "Escape") onEscape();
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [onEscape]);
};

/* ---------------- header ---------------- */

const HealthPill: React.FC<{ health: MlHealth | null | undefined }> = ({ health }) => {
  const up = !!health;
  const vqa = health?.vqa ?? null;
  const backend = health?.scan_backend;
  /** POST /scan works when a scan model is configured or the service names a scan backend. */
  const scan = !!(health?.scan_model || backend);
  return (
    <span className="health-pill" title={health ? JSON.stringify(health) : "ML service unreachable"}>
      <span className={`status-dot ${health === undefined ? "" : up ? "ok" : "down"}`} /> ML service {health === undefined ? "…" : up ? "up" : "down"}
      <span className="sep">|</span>
      <span className={`status-dot ${health === undefined ? "" : vqa ? "ok" : "down"}`} /> assistant {vqa ? "ready" : "unavailable"}
      <span className="sep">|</span>
      <span className={`status-dot ${health === undefined ? "" : scan ? "ok" : "down"}`} /> scan model {scan ? "ready" : "unavailable"}
      {backend ? <span className="health-backend">{backend}</span> : null}
    </span>
  );
};

const FlowSteps: React.FC = () => (
  <ol className="flow-steps" aria-label="Demo flow">
    <li className="flow-step"><span className="flow-n">1</span> Pick a tile</li>
    <li className="flow-arrow" aria-hidden>→</li>
    <li className="flow-step"><span className="flow-n">2</span> {VLM_NAME} scans it</li>
    <li className="flow-arrow" aria-hidden>→</li>
    <li className="flow-step"><span className="flow-n">3</span> Ask it anything</li>
  </ol>
);

/* ---------------- section 1: tumor scan ---------------- */

const ScanProgress: React.FC<{ step: number }> = ({ step }) => (
  <ol className="scan-progress" aria-live="polite">
    {SCAN_STEPS.map((label, i) => {
      const state = i < step ? "done" : i === step ? "active" : "pending";
      return (
        <li key={label} className={`scan-step ${state}`}>
          <span className="scan-step-icon">{state === "done" ? "✓" : state === "active" ? <span className="spinner" /> : "·"}</span>
          {label}
        </li>
      );
    })}
  </ol>
);

const ScanSection: React.FC<{ items: GalleryScanItem[]; onScanned: () => void }> = ({ items, onScanned }) => {
  const [selected, setSelected] = useState<GalleryScanItem | null>(null);
  const [results, setResults] = useState<Record<string, ScanResult>>({});
  const [busy, setBusy] = useState(false);
  /** Index of the active progress step while busy. */
  const [step, setStep] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const run = useRef(0);

  useEscape(() => setSelected(null));

  const res = selected ? results[selected.id] : undefined;

  const scan = async () => {
    if (!selected || busy) return;
    const id = selected.id;
    const my = ++run.current;
    setBusy(true);
    setStep(0);
    setError(null);
    // Run the request and the minimum-duration step animation concurrently; reveal when both are done.
    const animation = (async () => {
      for (let i = 1; i <= SCAN_STEPS.length; i++) {
        await sleep(STEP_MS);
        if (run.current === my) setStep(i);
      }
    })();
    try {
      const [r] = await Promise.all([scanTile(id), animation]);
      if (run.current !== my) return;
      setResults((m) => ({ ...m, [id]: r }));
      onScanned();
    } catch (err) {
      if (run.current !== my) return;
      setError(`Scan failed: ${errorMessage(err)}`);
    } finally {
      if (run.current === my) setBusy(false);
    }
  };

  const vlmErr = res?.vlm && "error" in res.vlm ? res.vlm.error : null;
  const vlmOk = res?.vlm && !("error" in res.vlm) ? res.vlm : null;
  const byVlm = !!res && res.decided_by === "vlm" && !!vlmOk;
  const correct = res ? res.verdict === res.truth.label_name : false;
  const pct = res ? Math.round(res.confidence * 100) : 0;

  return (
    <section className="demo-section">
      <h2>Tumor scan</h2>
      <p className="demo-subtitle">{VLM_NAME} · LoRA on Fireworks · second opinion: {PROBE_NAME}</p>
      <p className="demo-explainer">Pick a 96 px lymph-node tile (Camelyon17). The fine-tuned VLM reads it; the probe gives a second opinion. Ground truth is revealed after the scan.</p>
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
            {busy && <ScanProgress step={step} />}
            {error && <div className="notice error">{error}</div>}
            {res && !busy && (
              <div className="scan-outcome" key={`${res.id}-${res.latency_ms}`}>
                <div className={`verdict pop-in ${res.verdict}`}>
                  {res.verdict === "tumor" ? "TUMOR" : "NO TUMOR"}
                  <span className="verdict-conf">{pct}%</span>
                </div>
                <div className="conf-bar" title={`confidence ${pct}%`}>
                  <div className={`conf-fill ${res.verdict}`} style={{ width: `${pct}%` }} />
                </div>
                <div className="demo-says fade-in">
                  {byVlm ? (
                    <>
                      <strong>{VLM_NAME}</strong> says: <strong className={`v-${res.verdict}`}>{res.verdict === "tumor" ? "TUMOR" : "NO TUMOR"}</strong> ({pct}%)
                    </>
                  ) : (
                    <>
                      <strong>{PROBE_NAME}</strong> decided: <strong className={`v-${res.verdict}`}>{res.verdict === "tumor" ? "TUMOR" : "NO TUMOR"}</strong> ({pct}%)
                      <span className="muted"> · {VLM_NAME} {vlmErr ? "unavailable" : "did not decide"}</span>
                    </>
                  )}
                  {res.cached ? <span className="muted"> · cached</span> : null}
                </div>
                {vlmErr && <div className="notice warn">VLM unavailable — probe decided ({vlmErr})</div>}
                <table className="opinion-table fade-in">
                  <thead>
                    <tr>
                      <th>opinion</th>
                      <th>verdict</th>
                      <th>p_tumor</th>
                      <th>latency</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className={byVlm ? "decider" : ""}>
                      <td>
                        {VLM_NAME}
                        {vlmOk?.source ? <span className="muted"> · {vlmOk.source}</span> : null}
                      </td>
                      {vlmOk ? (
                        <>
                          <td className={`v-${vlmOk.verdict}`}>{vlmOk.verdict}</td>
                          <td>{vlmOk.p_tumor.toFixed(3)}</td>
                          <td>{Math.round(vlmOk.latency_ms)} ms</td>
                        </>
                      ) : (
                        <td colSpan={3} className="muted">{vlmErr ? "error" : "—"}</td>
                      )}
                    </tr>
                    <tr className={!byVlm ? "decider" : ""}>
                      <td>{PROBE_NAME}</td>
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
                <div className="demo-line">
                  {Math.round(res.latency_ms)} ms · ${res.cost_usd.toFixed(4)} · <span title={res.model}>{res.model}</span>
                </div>
                <div className={`truth-line reveal ${correct ? "ok" : "bad"}`}>
                  Ground truth: <strong>{res.truth.label_name}</strong> <span className="truth-mark">{correct ? "✓" : "✗"}</span>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  );
};

/* ---------------- section 2: ask the assistant ---------------- */

const AskSection: React.FC<{ items: GalleryAskItem[]; available: boolean; session: Session; onUnavailable: () => void; onAnswered: (latency_ms: number, cost_usd: number) => void }> = ({ items, available, session, onUnavailable, onAnswered }) => {
  const [selected, setSelected] = useState<GalleryAskItem | null>(null);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<Record<string, AskEntry[]>>({});

  useEscape(() => setSelected(null));

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
      onAnswered(r.latency_ms, r.cost_usd);
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
  const n = session.asks.length;
  const med = median(session.asks.map((a) => a.latency_ms));
  const total = session.asks.reduce((s, a) => s + a.cost_usd, 0);

  return (
    <section className="demo-section">
      <div className="demo-section-head">
        <div>
          <h2>Ask the pathology assistant</h2>
          <p className="demo-subtitle">{VLM_NAME} · LoRA on 8,000 PathVQA Q&amp;A · $0.33 to train</p>
        </div>
        <div className="session-counter" title="This session: questions answered, median latency, total inference cost">
          <strong>{n}</strong> {n === 1 ? "question" : "questions"} · median <strong>{Math.round(med)} ms</strong> · <strong>${total.toFixed(4)}</strong> total
        </div>
      </div>
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
            {busy && (
              <div className="ask-thinking muted">
                <span className="spinner" /> {VLM_NAME} is looking at the image…
              </div>
            )}
            {error && <div className="notice error">{error}</div>}
            {last && (
              <div className="demo-answer pop-in" key={last.at}>
                <div className="ask-q muted">{last.question}</div>
                <div className="demo-answer-a">{last.answer}</div>
                {last.expected !== null && (
                  <div className="demo-expected fade-in">
                    expected: <strong>{last.expected}</strong>{" "}
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

/* ---------------- footer ---------------- */

const SavedStrip: React.FC<{ session: Session }> = ({ session }) => {
  const secs = session.scans * SEC_PER_TILE_READ + session.asks.length * SEC_PER_LOOKUP;
  const mins = secs / 60;
  const label = secs < 60 ? `~${secs} s` : mins >= 10 ? `~${Math.round(mins)} min` : `~${mins.toFixed(1)} min`;
  return (
    <div className="saved-strip" title={`Assumption: ${SEC_PER_TILE_READ} s per tile read, ${SEC_PER_LOOKUP} s per lookup. ${session.scans} scans × ${SEC_PER_TILE_READ} s + ${session.asks.length} questions × ${SEC_PER_LOOKUP} s = ${secs} s.`}>
      <span className="saved-label">Pathologist time saved this session</span>
      <span className="saved-value" key={secs}>{label}</span>
      <span className="saved-detail muted">{session.scans} {session.scans === 1 ? "scan" : "scans"} · {session.asks.length} {session.asks.length === 1 ? "question" : "questions"} · {SEC_PER_TILE_READ} s / tile, {SEC_PER_LOOKUP} s / lookup</span>
    </div>
  );
};

/* ---------------- page ---------------- */

const DemoPage: React.FC = () => {
  /** undefined = probing; null = unreachable. */
  const [health, setHealth] = useState<MlHealth | null | undefined>(undefined);
  const [gallery, setGallery] = useState<Gallery | null>(null);
  const [galleryError, setGalleryError] = useState<string | null>(null);
  const [session, setSession] = useState<Session>({ scans: 0, asks: [] });

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
          <div className="demo-tagline">One fine-tuned Qwen3.5-9B, two jobs: screen lymph-node tiles for tumor, and answer questions about pathology images.</div>
        </div>
        <div className="demo-header-right">
          <HealthPill health={health} />
          <Link to="/" className="demo-link">Open slide viewer →</Link>
        </div>
        <FlowSteps />
      </header>

      {galleryError && (
        <div className="notice error">
          Could not load the demo gallery (<code>GET /ml/gallery</code>): {galleryError}
        </div>
      )}
      {!gallery && !galleryError && <div className="notice">Loading gallery…</div>}

      {gallery && <ScanSection items={gallery.scan} onScanned={() => setSession((s) => ({ ...s, scans: s.scans + 1 }))} />}
      {gallery && (
        <AskSection
          items={gallery.ask}
          available={assistantUp}
          session={session}
          onUnavailable={() => setHealth((h) => (h ? { ...h, vqa: null } : h))}
          onAnswered={(latency_ms, cost_usd) => setSession((s) => ({ ...s, asks: [...s.asks, { latency_ms, cost_usd }] }))}
        />
      )}

      <SavedStrip session={session} />
    </div>
  );
};

export default DemoPage;
