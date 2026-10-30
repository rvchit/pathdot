import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";

dotenv.config();

/** Monorepo root: this file is cello/backend/{src,dist}/server/config.{ts,js}, four levels down. */
export const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../..");

function resolveDir(envValue: string | undefined, fallback: string): string {
  return envValue ? path.resolve(envValue) : fallback;
}

/**
 * Central runtime configuration for the cello backend.
 *
 * STORAGE_MODE=local (default): tiles are read from TILES_DIR on disk, uploads go to
 * UPLOAD_DIR, and no AWS SDK client is ever constructed.
 * STORAGE_MODE=s3: the original behaviour (tiles streamed from S3_TILES_BUCKET,
 * multipart uploads into S3_BUCKET). AWS_* env vars are required in this mode only.
 */
export type StorageMode = "local" | "s3";

export const STORAGE_MODE: StorageMode =
  (process.env.STORAGE_MODE ?? "local").toLowerCase() === "s3" ? "s3" : "local";

export const PORT = Number(process.env.PORT ?? 4001);

/** Local DZI pyramids: <TILES_DIR>/<imageId>/<imageId>.dzi (or <TILES_DIR>/<imageId>.dzi)
 *  and <TILES_DIR>/<imageId>_files/<level>/<x>_<y>.jpeg */
export const TILES_DIR = resolveDir(process.env.TILES_DIR, path.join(REPO_ROOT, "tiles"));

/** Raw uploaded slides (local mode) and ground-truth JSON for the mosaic demo. */
export const UPLOAD_DIR = resolveDir(process.env.UPLOAD_DIR, path.join(REPO_ROOT, "data", "slides"));

/** JSON file store for comments lives at <DATA_DIR>/comments.json. */
export const DATA_DIR = resolveDir(process.env.DATA_DIR, path.join(REPO_ROOT, "data"));

/** The FastAPI inference service (pathfrontier.serve). Everything under /ml/* is proxied here. */
export const ML_SERVICE_URL = (process.env.ML_SERVICE_URL ?? "http://localhost:8008").replace(/\/+$/, "");

export const JSON_BODY_LIMIT = process.env.JSON_BODY_LIMIT ?? "50mb";

/** Comma-separated list of allowed origins; "*" (default) allows everything. */
export const CORS_ORIGIN = process.env.CORS_ORIGIN ?? "*";

/** True if the string is a safe single path segment (no slashes, no "..", no NUL). */
export function isSafeSegment(s: string): boolean {
  return (
    typeof s === "string" &&
    s.length > 0 &&
    s.length <= 255 &&
    !s.includes("/") &&
    !s.includes("\\") &&
    !s.includes("\0") &&
    s !== "." &&
    s !== ".."
  );
}

/** True if the string is a safe image id: letters, digits, dot, dash, underscore. */
export function isSafeImageId(s: string): boolean {
  return isSafeSegment(s) && /^[A-Za-z0-9_][A-Za-z0-9._-]*$/.test(s);
}
