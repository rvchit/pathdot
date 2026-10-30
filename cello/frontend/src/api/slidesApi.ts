import { api } from "./client";

/** GET /osd/dzi/:imageId */
export interface DziInfo {
  imageId: string;
  width: number;
  height: number;
  tileSize: number;
  overlap: number;
  format: string;
  maxLevel: number;
}

export async function listImages(): Promise<string[]> {
  const { data } = await api.get<{ images: string[] }>("/osd/images");
  return data.images ?? [];
}

export async function getDzi(imageId: string): Promise<DziInfo> {
  const { data } = await api.get<DziInfo>(`/osd/dzi/${encodeURIComponent(imageId)}`);
  return data;
}

// ---------------- ground truth (make_mosaic_slide.py --> data/slides/<id>.gt.json) ----------------

/** Normalised ground-truth cell in image pixels. */
export interface GtCell {
  x: number;
  y: number;
  w: number;
  h: number;
  label: string | number;
  tumor: boolean;
}

export interface GroundTruth {
  cells: GtCell[];
  positiveLabel: string | number;
  raw: unknown;
}

type RawCell = Record<string, unknown>;

/**
 * The tiler's gt.json shape is "per-cell label" (contract section 9) and may evolve,
 * so this parser is deliberately tolerant. Supported shapes:
 *   {grid|cols,rows, tile_px|tile_size|cell_px, cells:[{row,col,label}|{x,y,label}|{x,y,w,h,label}]}
 *   {labels:[[...row0...],[...row1...]], tile_px}
 *   plus optional {classes:[...], positive_index|positive_class|positive_label}
 * `imageWidth` (from the DZI) is used to infer the cell size when none is given.
 */
export function normaliseGroundTruth(raw: unknown, imageWidth: number, imageHeight: number): GroundTruth {
  const obj = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  const num = (v: unknown): number | undefined => (typeof v === "number" && Number.isFinite(v) ? v : undefined);

  const classes = Array.isArray(obj.classes) ? (obj.classes as unknown[]) : undefined;
  let positiveLabel: string | number = 1;
  if (num(obj.positive_index) !== undefined) {
    positiveLabel = classes ? String(classes[num(obj.positive_index)!]) : num(obj.positive_index)!;
  }
  if (typeof obj.positive_class === "string") positiveLabel = obj.positive_class;
  if (typeof obj.positive_label === "string" || typeof obj.positive_label === "number") positiveLabel = obj.positive_label;

  const isTumor = (label: unknown, cell?: RawCell): boolean => {
    if (cell && typeof cell.tumor === "boolean") return cell.tumor;
    if (cell && typeof cell.is_tumor === "boolean") return cell.is_tumor;
    if (label === positiveLabel) return true;
    if (typeof label === "string") {
      const l = label.toLowerCase();
      return l === "tumor" || l === "tum" || l === "positive" || l === "1";
    }
    if (typeof label === "number" && classes) return classes[label] === positiveLabel;
    return label === 1 || label === true;
  };

  let labels2d: unknown[][] | undefined = Array.isArray(obj.labels) && Array.isArray(obj.labels[0]) ? (obj.labels as unknown[][]) : undefined;
  if (!labels2d && Array.isArray(obj.grid) && Array.isArray(obj.grid[0])) labels2d = obj.grid as unknown[][];

  const rawCells = Array.isArray(obj.cells) ? (obj.cells as RawCell[]) : Array.isArray(obj.tiles) ? (obj.tiles as RawCell[]) : undefined;

  let cols = num(obj.cols) ?? num(obj.grid_cols) ?? (typeof obj.grid === "number" ? obj.grid : undefined);
  let rows = num(obj.rows) ?? num(obj.grid_rows) ?? (typeof obj.grid === "number" ? obj.grid : undefined);
  if (labels2d) {
    rows = rows ?? labels2d.length;
    cols = cols ?? Math.max(...labels2d.map((r) => r.length));
  }
  if (!cols && rawCells) {
    const maxCol = Math.max(-1, ...rawCells.map((c) => num(c.col) ?? num(c.c) ?? num(c.j) ?? -1));
    if (maxCol >= 0) cols = maxCol + 1;
  }
  if (!rows && rawCells) {
    const maxRow = Math.max(-1, ...rawCells.map((c) => num(c.row) ?? num(c.r) ?? num(c.i) ?? -1));
    if (maxRow >= 0) rows = maxRow + 1;
  }

  const cellW = num(obj.tile_px) ?? num(obj.tile_size) ?? num(obj.cell_px) ?? num(obj.cell_size) ?? num(obj.patch_px) ?? (cols ? imageWidth / cols : undefined);
  const cellH = num(obj.tile_px) ?? num(obj.tile_size) ?? num(obj.cell_px) ?? num(obj.cell_size) ?? num(obj.patch_px) ?? (rows ? imageHeight / rows : cellW);

  const cells: GtCell[] = [];
  if (labels2d && cellW && cellH) {
    labels2d.forEach((row, r) =>
      row.forEach((label, c) => {
        cells.push({ x: c * cellW, y: r * cellH, w: cellW, h: cellH, label: label as string | number, tumor: isTumor(label) });
      }),
    );
  } else if (rawCells) {
    for (const c of rawCells) {
      const label = (c.label ?? c.y_true ?? c.class ?? c.cls) as string | number;
      const col = num(c.col) ?? num(c.c) ?? num(c.j);
      const row = num(c.row) ?? num(c.r) ?? num(c.i);
      const w = num(c.w) ?? num(c.width) ?? cellW;
      const h = num(c.h) ?? num(c.height) ?? cellH;
      let x = num(c.x);
      let y = num(c.y);
      // {x,y} without a width and with a small range are grid indices, not pixels
      const looksLikeIndex = x !== undefined && y !== undefined && num(c.w) === undefined && cols !== undefined && x < cols && y < (rows ?? cols) && Number.isInteger(x) && Number.isInteger(y);
      if (col !== undefined && row !== undefined && cellW && cellH) {
        x = col * cellW;
        y = row * cellH;
      } else if (looksLikeIndex && cellW && cellH) {
        x = x! * cellW;
        y = y! * cellH;
      }
      if (x === undefined || y === undefined || !w || !h) continue;
      cells.push({ x, y, w, h, label, tumor: isTumor(label, c) });
    }
  }
  return { cells, positiveLabel, raw };
}

/** GET /osd/gt/:imageId ; returns null on 404. */
export async function getGroundTruth(imageId: string, imageWidth: number, imageHeight: number): Promise<GroundTruth | null> {
  try {
    const { data } = await api.get<unknown>(`/osd/gt/${encodeURIComponent(imageId)}`);
    return normaliseGroundTruth(data, imageWidth, imageHeight);
  } catch (err) {
    if (typeof err === "object" && err && "response" in err && (err as { response?: { status?: number } }).response?.status === 404) return null;
    throw err;
  }
}

// ---------------- comments ----------------

export interface Comment {
  id: string;
  imageId: string;
  annotationId: string;
  comment: string;
  geometry?: unknown;
  createdAt: string;
}

export async function listComments(imageId: string): Promise<Comment[]> {
  const { data } = await api.get<Comment[]>(`/comments/${encodeURIComponent(imageId)}`);
  return data;
}

export async function createComment(input: { imageId: string; annotationId: string; comment: string; geometry?: unknown }): Promise<Comment> {
  const { data } = await api.post<Comment>("/comments", input);
  return data;
}

export async function deleteComment(id: string): Promise<void> {
  await api.delete(`/comments/${encodeURIComponent(id)}`);
}

// ---------------- upload (local mode) ----------------

export interface LocalUploadResult {
  message: string;
  imageId: string;
  path: string;
  size: number;
  nextStep: string;
}

export async function uploadLocal(file: File, onProgress?: (frac: number) => void): Promise<LocalUploadResult> {
  const form = new FormData();
  form.append("file", file, file.name);
  const { data } = await api.post<LocalUploadResult>("/api/upload-local", form, {
    timeout: 0,
    onUploadProgress: (e) => {
      if (onProgress && e.total) onProgress(e.loaded / e.total);
    },
  });
  return data;
}

export async function uploadMode(): Promise<"local" | "s3"> {
  const { data } = await api.get<{ mode: "local" | "s3" }>("/api/upload-mode");
  return data.mode;
}
