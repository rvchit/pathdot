import { api } from "./client";

/** Types mirror docs/CONTRACTS.md section 8 (pathfrontier.serve). */
export type Decision = "negative" | "positive" | "review";
export type RoutedTo = "probe" | "vlm_ft" | "pathologist";
export type Fallback = "vlm_ft" | "pathologist" | "none";

export interface HeatmapTile {
  x: number;
  y: number;
  p_tumor: number;
  decision: Decision;
  routed_to: RoutedTo;
  reason: string | null;
}

export interface HeatmapSummary {
  n_tiles: number;
  auto_cleared: number;
  auto_flagged: number;
  review: number;
  frac_auto_cleared: number;
  cost_usd: number;
  latency_ms_total: number;
  pathologist_minutes_saved: number;
  model: string;
}

export interface Heatmap {
  image_id: string;
  width: number;
  height: number;
  tile_size: number;
  level: number;
  max_level: number;
  /** image pixels per tile pixel at `level` = 2^(max_level-level) */
  scale: number;
  cols: number;
  rows: number;
  tiles: HeatmapTile[];
  summary: HeatmapSummary;
}

export interface SlideJobRequest {
  image_id: string;
  level?: number | null;
  max_tiles?: number | null;
  low: number;
  high: number;
  fallback: Fallback;
}

export interface JobStarted {
  job_id: string;
  status: "running";
}

export interface JobStatus {
  status: "running" | "done" | "error";
  progress: number;
  result: Heatmap | null;
  error?: string;
  detail?: string;
}

export interface TilePrediction {
  p_tumor: number;
  probs: number[];
  classes: string[];
  decision: Decision;
  routed_to: RoutedTo;
  reason: string | null;
  latency_ms: number;
  cost_usd: number;
}

export interface MlHealth {
  status: string;
  encoder?: string;
  probe?: string;
  vlm?: string | null;
  /** PathVQA assistant model id, or null when the /vqa endpoint is not configured. */
  vqa?: string | null;
  /** Model id used by POST /scan (demo gallery), or null when not configured. */
  scan_model?: string | null;
  device?: string;
}

/** POST /ml/vqa — ask the fine-tuned PathVQA model a free-text question about one DZI tile. */
export interface VqaRequest {
  image_id: string;
  /** DZI level; null = full resolution. x/y are tile indices at that level (tile size 256). */
  level: number | null;
  x: number;
  y: number;
  question: string;
}

export interface VqaResponse {
  answer: string;
  question: string;
  model: string;
  latency_ms: number;
  cost_usd: number;
  tokens_in: number;
  tokens_out: number;
  cached?: boolean;
}

export async function mlHealth(): Promise<MlHealth> {
  const { data } = await api.get<MlHealth>("/ml/health", { timeout: 5_000 });
  return data;
}

export async function startSlideJob(req: SlideJobRequest): Promise<JobStarted> {
  const { data } = await api.post<JobStarted>("/ml/predict/slide", req);
  return data;
}

export async function pollJob(jobId: string): Promise<JobStatus> {
  const { data } = await api.get<JobStatus>(`/ml/jobs/${encodeURIComponent(jobId)}`);
  return data;
}

export async function getHeatmap(imageId: string): Promise<Heatmap> {
  const { data } = await api.get<Heatmap>(`/ml/heatmap/${encodeURIComponent(imageId)}`);
  return data;
}

export async function predictTile(file: Blob, filename = "tile.jpeg"): Promise<TilePrediction> {
  const form = new FormData();
  form.append("file", file, filename);
  const { data } = await api.post<TilePrediction>("/ml/predict/tile", form);
  return data;
}

export async function askTile(req: VqaRequest): Promise<VqaResponse> {
  const { data } = await api.post<VqaResponse>("/ml/vqa", req, { timeout: 90_000 });
  return data;
}

/**
 * Start a slide job and poll until done. Calls onProgress(0..1) every tick.
 * Resolves with the heatmap; rejects on error status or network failure.
 */
export async function runSlideTriage(
  req: SlideJobRequest,
  onProgress: (p: number) => void,
  opts: { intervalMs?: number; signal?: AbortSignal } = {},
): Promise<Heatmap> {
  const { intervalMs = 1000, signal } = opts;
  const { job_id } = await startSlideJob(req);
  onProgress(0);
  for (;;) {
    if (signal?.aborted) throw new Error("cancelled");
    await new Promise((r) => setTimeout(r, intervalMs));
    const status = await pollJob(job_id);
    onProgress(Math.max(0, Math.min(1, status.progress ?? 0)));
    if (status.status === "done" && status.result) return status.result;
    if (status.status === "error") throw new Error(status.error ?? status.detail ?? "ML job failed");
  }
}

/* ---------- demo gallery (/demo page) ---------- */

export interface GalleryQuestion {
  question: string;
  answer: string;
  qtype: "closed" | "open";
}

export interface GalleryScanItem {
  id: string;
  section: "scan";
  label: 0 | 1;
  label_name: string;
  source: string;
  meta: Record<string, unknown>;
  /** Relative to the ML service; browser URL = `${BACKEND_URL}/ml${image_url}`. */
  image_url: string;
}

export interface GalleryAskItem {
  id: string;
  section: "ask";
  image_url: string;
  questions: GalleryQuestion[];
  meta: Record<string, unknown>;
}

export interface Gallery {
  scan: GalleryScanItem[];
  ask: GalleryAskItem[];
}

export interface ScanOpinion {
  verdict: "tumor" | "normal";
  p_tumor: number;
  latency_ms: number;
  model: string;
}

export interface ScanResult {
  id: string;
  verdict: "tumor" | "normal";
  p_tumor: number;
  confidence: number;
  decided_by: "vlm" | "probe";
  model: string;
  latency_ms: number;
  cost_usd: number;
  truth: { label: 0 | 1; label_name: string };
  probe: ScanOpinion | null;
  vlm: (ScanOpinion & { answer: string; cost_usd: number }) | { error: string } | null;
  cached: boolean;
}

export interface GalleryVqaResponse {
  answer: string;
  question: string;
  model: string;
  latency_ms: number;
  cost_usd: number;
  cached?: boolean;
}

export async function getGallery(): Promise<Gallery> {
  const { data } = await api.get<Gallery>("/ml/gallery", { timeout: 10_000 });
  return data;
}

export async function scanTile(id: string): Promise<ScanResult> {
  const { data } = await api.post<ScanResult>("/ml/scan", { id }, { timeout: 90_000 });
  return data;
}

export async function askGallery(gallery_id: string, question: string): Promise<GalleryVqaResponse> {
  const { data } = await api.post<GalleryVqaResponse>("/ml/vqa", { gallery_id, question }, { timeout: 90_000 });
  return data;
}
