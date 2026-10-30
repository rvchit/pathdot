import { Router, Request, Response } from "express";
import fsp from "fs/promises";
import path from "path";
import { randomUUID } from "crypto";
import { DATA_DIR, isSafeImageId } from "./config.js";

/**
 * Comments attached to Annotorious annotations.
 *
 * Storage: a JSON file at <DATA_DIR>/comments.json, written atomically
 * (write to a temp file, then rename). Good enough for a single-process demo.
 *
 * Switching to Prisma/Postgres later: `prisma/schema.prisma` already has User/Image
 * models; add a `Comment` model {id String @id, imageId String, annotationId String,
 * comment String, geometry Json?, createdAt DateTime @default(now())}, run
 * `npx prisma migrate dev`, and replace `readStore`/`writeStore` below with
 * `prisma.comment.findMany / create / delete` guarded by `if (process.env.DATABASE_URL)`.
 */
export interface Comment {
  id: string;
  imageId: string;
  annotationId: string;
  comment: string;
  geometry?: unknown;
  createdAt: string;
}

const STORE_PATH = path.join(DATA_DIR, "comments.json");

let writeChain: Promise<void> = Promise.resolve();

async function readStore(): Promise<Comment[]> {
  try {
    const raw = await fsp.readFile(STORE_PATH, "utf8");
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as Comment[]) : [];
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw err;
  }
}

async function writeStore(comments: Comment[]): Promise<void> {
  await fsp.mkdir(DATA_DIR, { recursive: true });
  const tmp = `${STORE_PATH}.${process.pid}.${Date.now()}.tmp`;
  await fsp.writeFile(tmp, JSON.stringify(comments, null, 2), "utf8");
  await fsp.rename(tmp, STORE_PATH);
}

/** Serialise read-modify-write cycles so concurrent POSTs cannot clobber each other. */
function withStore<T>(fn: (comments: Comment[]) => Promise<[Comment[], T]>): Promise<T> {
  const run = writeChain.then(async () => {
    const current = await readStore();
    const [next, result] = await fn(current);
    await writeStore(next);
    return result;
  });
  writeChain = run.then(
    () => undefined,
    () => undefined,
  );
  return run;
}

const router = Router();

// GET /comments/:imageId -> Comment[]
router.get("/:imageId", async (req: Request, res: Response) => {
  const { imageId } = req.params;
  if (!isSafeImageId(imageId)) return res.status(400).json({ error: "Invalid imageId" });
  try {
    const all = await readStore();
    res.json(all.filter((c) => c.imageId === imageId));
  } catch (err) {
    console.error("comments read failed:", err);
    res.status(500).json({ error: "Failed to read comments" });
  }
});

// POST /comments {imageId, annotationId, comment, geometry?} -> Comment
router.post("/", async (req: Request, res: Response) => {
  const { imageId, annotationId, comment, geometry } = (req.body ?? {}) as Partial<Comment>;
  if (!imageId || !isSafeImageId(imageId)) return res.status(400).json({ error: "imageId is required" });
  if (!annotationId || typeof annotationId !== "string") {
    return res.status(400).json({ error: "annotationId is required" });
  }
  if (typeof comment !== "string" || comment.trim().length === 0) {
    return res.status(400).json({ error: "comment is required" });
  }
  const entry: Comment = {
    id: randomUUID(),
    imageId,
    annotationId,
    comment: comment.trim(),
    createdAt: new Date().toISOString(),
  };
  if (geometry !== undefined) entry.geometry = geometry;
  try {
    const saved = await withStore<Comment>(async (all) => [[...all, entry], entry]);
    res.status(201).json(saved);
  } catch (err) {
    console.error("comments write failed:", err);
    res.status(500).json({ error: "Failed to save comment" });
  }
});

// DELETE /comments/:id -> {deleted: true}
router.delete("/:id", async (req: Request, res: Response) => {
  const { id } = req.params;
  try {
    const removed = await withStore<boolean>(async (all) => {
      const next = all.filter((c) => c.id !== id);
      return [next, next.length !== all.length];
    });
    if (!removed) return res.status(404).json({ error: "Comment not found" });
    res.json({ deleted: true, id });
  } catch (err) {
    console.error("comments delete failed:", err);
    res.status(500).json({ error: "Failed to delete comment" });
  }
});

export default router;
