#!/usr/bin/env node
// Tiny stand-in for pathfrontier.serve (FastAPI on :8008) so the cello UI and the
// /ml/* proxy can be exercised with no model, no GPU and no Python.
// Usage: node scripts/ml-stub.mjs [port]   (default 8008)
// It answers /health, POST /predict/slide, GET /jobs/:id, GET /heatmap/:imageId and
// POST /predict/tile with canned JSON matching docs/CONTRACTS.md section 8.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";

// Read the real slide size from the local DZI (same lookup as the backend) so the
// canned heatmap lines up with whatever slide is open; falls back to 2304x2304.
const TILES_DIR = process.env.PATHFRONTIER_TILES_DIR ?? path.resolve(process.cwd(), "../../tiles");
function slideSize(imageId) {
  for (const p of [path.join(TILES_DIR, imageId, `${imageId}.dzi`), path.join(TILES_DIR, `${imageId}.dzi`)]) {
    try {
      const xml = fs.readFileSync(p, "utf8");
      const w = Number(/Width="(\d+)"/.exec(xml)?.[1]), h = Number(/Height="(\d+)"/.exec(xml)?.[1]);
      if (w && h) return { width: w, height: h };
    } catch { /* next */ }
  }
  return { width: 2304, height: 2304 };
}

const PORT = Number(process.argv[2] ?? process.env.PORT ?? 8008);
const JOB_MS = Number(process.env.STUB_JOB_MS ?? 3000); // how long a fake slide job takes
const jobs = new Map();
const heatmaps = new Map();

function makeHeatmap(imageId, low = 0.05, high = 0.95) {
  const { width, height } = slideSize(imageId);
  const tile_size = 96, max_level = Math.ceil(Math.log2(Math.max(width, height))), level = max_level;
  const scale = 2 ** (max_level - level);
  const cols = Math.ceil(width / (tile_size * scale)), rows = Math.ceil(height / (tile_size * scale));
  let seed = [...imageId].reduce((a, c) => a + c.charCodeAt(0), 7);
  const rnd = () => ((seed = (seed * 9301 + 49297) % 233280) / 233280);
  const tiles = [];
  const summary = { n_tiles: 0, auto_cleared: 0, auto_flagged: 0, review: 0 };
  for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) {
    const blob = Math.hypot(x - cols * 0.65, y - rows * 0.4) < rows * 0.18;
    const p = Math.min(0.999, Math.max(0.001, blob ? 0.6 + rnd() * 0.4 : rnd() * 0.3));
    const decision = p < low ? "negative" : p > high ? "positive" : "review";
    const routed_to = decision === "review" ? (rnd() < 0.7 ? "vlm_ft" : "pathologist") : "probe";
    const reason = routed_to === "vlm_ft" ? (p > 0.5 ? "Enlarged hyperchromatic nuclei in sheets; loss of nodal architecture." : "Lymphoid tissue with preserved follicles; no atypical cells.") : null;
    tiles.push({ x, y, p_tumor: +p.toFixed(3), decision, routed_to, reason });
    summary.n_tiles++;
    if (decision === "negative") summary.auto_cleared++; else if (decision === "positive") summary.auto_flagged++; else summary.review++;
  }
  const frac = summary.n_tiles ? summary.auto_cleared / summary.n_tiles : 0;
  return {
    image_id: imageId, width, height, tile_size, level, max_level, scale, cols, rows, tiles,
    summary: {
      ...summary, frac_auto_cleared: +frac.toFixed(3),
      cost_usd: +(summary.review * 0.0006 + summary.n_tiles * 0.0000005).toFixed(5),
      latency_ms_total: 650 + summary.review * 180,
      pathologist_minutes_saved: +(((summary.n_tiles - summary.review) * 2) / 60).toFixed(1),
      model: "stub/probe + stub/vlm_ft",
    },
  };
}

function json(res, status, body) {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(JSON.stringify(body));
}

http.createServer((req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  const chunks = [];
  req.on("data", (c) => chunks.push(c));
  req.on("end", () => {
    const raw = Buffer.concat(chunks);
    if (req.method === "GET" && url.pathname === "/health")
      return json(res, 200, { status: "ok", encoder: "stub", probe: "stub", vlm: null, device: "cpu" });
    if (req.method === "POST" && url.pathname === "/predict/tile")
      return json(res, 200, { p_tumor: 0.12, probs: [0.88, 0.12], classes: ["normal", "tumor"], decision: "negative", routed_to: "probe", reason: null, latency_ms: 4.2, cost_usd: 0.0000005, bytes_received: raw.length });
    if (req.method === "POST" && url.pathname === "/predict/slide") {
      let body = {};
      try { body = JSON.parse(raw.toString("utf8") || "{}"); } catch { return json(res, 422, { detail: "invalid JSON" }); }
      if (!body.image_id) return json(res, 422, { detail: "image_id required" });
      const job_id = `job_${Date.now().toString(36)}`;
      jobs.set(job_id, { started: Date.now(), image_id: body.image_id, low: body.low ?? 0.05, high: body.high ?? 0.95 });
      return json(res, 200, { job_id, status: "running" });
    }
    const jm = url.pathname.match(/^\/jobs\/([^/]+)$/);
    if (req.method === "GET" && jm) {
      const job = jobs.get(jm[1]);
      if (!job) return json(res, 404, { detail: "job not found" });
      const progress = Math.min(1, (Date.now() - job.started) / JOB_MS);
      if (progress < 1) return json(res, 200, { status: "running", progress: +progress.toFixed(2), result: null });
      const hm = makeHeatmap(job.image_id, job.low, job.high);
      heatmaps.set(job.image_id, hm);
      return json(res, 200, { status: "done", progress: 1, result: hm });
    }
    const hm = url.pathname.match(/^\/heatmap\/([^/]+)$/);
    if (req.method === "GET" && hm) {
      const h = heatmaps.get(decodeURIComponent(hm[1]));
      return h ? json(res, 200, h) : json(res, 404, { detail: "no heatmap yet" });
    }
    json(res, 404, { detail: "Not Found" });
  });
}).listen(PORT, () => console.log(`ml-stub listening on http://localhost:${PORT}`));
