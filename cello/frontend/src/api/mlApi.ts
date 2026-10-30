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
  device?: string;
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
