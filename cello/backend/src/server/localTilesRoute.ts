import { Router, Request, Response } from "express";
import fs from "fs";
import fsp from "fs/promises";
import path from "path";
import { TILES_DIR, UPLOAD_DIR, isSafeImageId } from "./config.js";

/**
 * Local-mode tile serving (STORAGE_MODE=local).
 *
 * Layout on disk (written by scripts/tile_slide.py / make_mosaic_slide.py):
 *   <TILES_DIR>/<imageId>/<imageId>.dzi        (preferred)   or
 *   <TILES_DIR>/<imageId>.dzi                  (also accepted)
 *   <TILES_DIR>/<imageId>_files/<level>/<x>_<y>.jpeg
 */
const router = Router();

export interface DziInfo {
  width: number;
  height: number;
  tileSize: number;
  overlap: number;
  format: string;
  maxLevel: number;
}

const CONTENT_TYPES: Record<string, string> = {
  jpeg: "image/jpeg",
  jpg: "image/jpeg",
  png: "image/png",
  webp: "image/webp",
};

function dziCandidates(imageId: string): string[] {
  return [
    path.join(TILES_DIR, imageId, `${imageId}.dzi`),
    path.join(TILES_DIR, `${imageId}.dzi`),
  ];
}

async function findDziPath(imageId: string): Promise<string | null> {
  for (const p of dziCandidates(imageId)) {
    try {
      const st = await fsp.stat(p);
      if (st.isFile()) return p;
    } catch {
      /* try next */
    }
  }
  return null;
}

/** Parse the DZI XML (no XML library needed; the format is a single <Image> element). */
export function parseDzi(xml: string): DziInfo {
  const attr = (name: string): string | undefined => {
    const m = xml.match(new RegExp(`${name}\\s*=\\s*"([^"]*)"`, "i"));
    return m?.[1];
  };
  const width = Number(attr("Width"));
  const height = Number(attr("Height"));
  const tileSize = Number(attr("TileSize") ?? 256);
  const overlap = Number(attr("Overlap") ?? 0);
  const format = (attr("Format") ?? "jpeg").toLowerCase();
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    throw new Error("DZI is missing a valid Size Width/Height");
  }
  const maxLevel = Math.ceil(Math.log2(Math.max(width, height)));
  return { width, height, tileSize, overlap, format, maxLevel };
}

export async function readDzi(imageId: string): Promise<DziInfo | null> {
  const p = await findDziPath(imageId);
  if (!p) return null;
  const xml = await fsp.readFile(p, "utf8");
  return parseDzi(xml);
}

/** All imageIds that have a .dzi in TILES_DIR (either layout). */
export async function listImageIds(): Promise<string[]> {
  let entries: fs.Dirent[];
  try {
    entries = await fsp.readdir(TILES_DIR, { withFileTypes: true });
  } catch {
    return [];
  }
  const ids = new Set<string>();
  for (const e of entries) {
    if (e.isFile() && e.name.endsWith(".dzi")) {
      ids.add(e.name.slice(0, -".dzi".length));
    } else if (e.isDirectory() && !e.name.endsWith("_files")) {
      try {
        await fsp.access(path.join(TILES_DIR, e.name, `${e.name}.dzi`));
        ids.add(e.name);
      } catch {
        /* not a pyramid dir */
      }
    }
  }
  return [...ids].filter(isSafeImageId).sort();
}

// GET /osd/images -> { images: ["pcam_mosaic_24", ...] }
router.get("/images", async (_req: Request, res: Response) => {
  try {
    const images = await listImageIds();
    res.json({ images, tilesDir: TILES_DIR });
  } catch (err) {
    console.error("listImageIds failed:", err);
    res.status(500).json({ error: "Failed to list images" });
  }
});

// GET /osd/dzi/:imageId -> {width,height,tileSize,overlap,format,maxLevel}
router.get("/dzi/:imageId", async (req: Request, res: Response) => {
  const { imageId } = req.params;
  if (!isSafeImageId(imageId)) return res.status(400).json({ error: "Invalid imageId" });
  try {
    const info = await readDzi(imageId);
    if (!info) return res.status(404).json({ error: `No .dzi found for ${imageId}` });
    res.json({ imageId, ...info });
  } catch (err) {
    console.error("readDzi failed:", err);
    res.status(500).json({ error: "Failed to parse .dzi" });
  }
});

// GET /osd/gt/:imageId -> data/slides/<imageId>.gt.json (ground truth for the mosaic demo)
router.get("/gt/:imageId", async (req: Request, res: Response) => {
  const { imageId } = req.params;
  if (!isSafeImageId(imageId)) return res.status(400).json({ error: "Invalid imageId" });
  const p = path.join(UPLOAD_DIR, `${imageId}.gt.json`);
  if (!p.startsWith(UPLOAD_DIR + path.sep)) return res.status(400).json({ error: "Invalid path" });
  try {
    const raw = await fsp.readFile(p, "utf8");
    res.type("application/json").send(raw);
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === "ENOENT") {
      return res.status(404).json({ error: `No ground truth for ${imageId}` });
    }
    console.error("gt read failed:", err);
    res.status(500).json({ error: "Failed to read ground truth" });
  }
});

// GET /osd/tile/:imageId/:level/:x/:y -> the jpeg tile
router.get("/tile/:imageId/:level/:x/:y", async (req: Request, res: Response) => {
  const { imageId, level, x, y } = req.params;
  if (!isSafeImageId(imageId)) return res.status(400).send("Invalid imageId");
  if (![level, x, y].every((v) => /^\d{1,4}$/.test(v))) {
    return res.status(400).send("Invalid tile coordinates");
  }
  const dir = path.join(TILES_DIR, `${imageId}_files`, level);
  // Try the common extensions in order; the .dzi Format tells us the real one but
  // reading it per tile is wasteful. jpeg first (contract), then jpg/png.
  for (const ext of ["jpeg", "jpg", "png", "webp"]) {
    const p = path.join(dir, `${x}_${y}.${ext}`);
    if (!p.startsWith(TILES_DIR + path.sep)) return res.status(400).send("Invalid path");
    try {
      const st = await fsp.stat(p);
      if (!st.isFile()) continue;
      res.setHeader("Content-Type", CONTENT_TYPES[ext]);
      res.setHeader("Content-Length", String(st.size));
      res.setHeader("Cache-Control", "public, max-age=3600");
      const stream = fs.createReadStream(p);
      stream.on("error", (err) => {
        console.error("tile stream error:", err);
        if (!res.headersSent) res.status(500).end();
        else res.end();
      });
      return stream.pipe(res);
    } catch {
      /* try next ext */
    }
  }
  res.status(404).send("Tile not found");
});

export default router;
