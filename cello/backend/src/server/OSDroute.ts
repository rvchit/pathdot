import { Router, Request, Response } from "express";
import { GetObjectCommand, ListObjectsV2Command } from "@aws-sdk/client-s3";
import { Readable } from "stream";
import { getS3Client } from "./s3client.js";
import { parseDzi } from "./localTilesRoute.js";
import { isSafeImageId } from "./config.js";

/**
 * S3-mode tile serving (STORAGE_MODE=s3). Layout in S3_TILES_BUCKET:
 *   <imageId>.dzi  (or <imageId>/<imageId>.dzi)
 *   <imageId>/<level>/<x>_<y>.jpeg        (legacy cello layout)
 *   <imageId>_files/<level>/<x>_<y>.jpeg  (standard DZI layout, tried second)
 */
const router = Router();

async function streamBody(body: unknown): Promise<string> {
  if (body instanceof Readable) {
    const chunks: Buffer[] = [];
    for await (const c of body) chunks.push(Buffer.from(c));
    return Buffer.concat(chunks).toString("utf8");
  }
  if (body && typeof (body as { transformToString?: () => Promise<string> }).transformToString === "function") {
    return (body as { transformToString: () => Promise<string> }).transformToString();
  }
  throw new Error("Unexpected S3 body type");
}

router.get("/images", async (_req: Request, res: Response) => {
  try {
    const out = await getS3Client().send(
      new ListObjectsV2Command({ Bucket: process.env.S3_TILES_BUCKET }),
    );
    const ids = new Set<string>();
    for (const obj of out.Contents ?? []) {
      const key = obj.Key ?? "";
      if (key.endsWith(".dzi")) ids.add(key.split("/").pop()!.slice(0, -4));
    }
    res.json({ images: [...ids].sort(), bucket: process.env.S3_TILES_BUCKET });
  } catch (error) {
    console.error("Error listing S3 images:", (error as Error).message);
    res.status(500).json({ error: "Failed to list images" });
  }
});

router.get("/dzi/:imageId", async (req: Request, res: Response) => {
  const { imageId } = req.params;
  if (!isSafeImageId(imageId)) return res.status(400).json({ error: "Invalid imageId" });
  for (const key of [`${imageId}.dzi`, `${imageId}/${imageId}.dzi`]) {
    try {
      const out = await getS3Client().send(
        new GetObjectCommand({ Bucket: process.env.S3_TILES_BUCKET, Key: key }),
      );
      const xml = await streamBody(out.Body);
      return res.json({ imageId, ...parseDzi(xml) });
    } catch {
      /* try next key */
    }
  }
  res.status(404).json({ error: `No .dzi found for ${imageId}` });
});

router.get("/gt/:imageId", (_req: Request, res: Response) => {
  res.status(404).json({ error: "Ground truth is only available in local mode" });
});

// Route to serve tiles directly from S3
router.get("/tile/:imageId/:level/:x/:y", async (req: Request, res: Response) => {
  const { imageId, level, x, y } = req.params;
  if (!isSafeImageId(imageId)) return res.status(400).send("Invalid imageId");
  const keys = [`${imageId}/${level}/${x}_${y}.jpeg`, `${imageId}_files/${level}/${x}_${y}.jpeg`];

  for (const tileKey of keys) {
    try {
      const tileResponse = await getS3Client().send(
        new GetObjectCommand({ Bucket: process.env.S3_TILES_BUCKET, Key: tileKey }),
      );
      if (tileResponse.Body instanceof Readable) {
        res.setHeader("Content-Type", "image/jpeg");
        return tileResponse.Body.pipe(res);
      }
    } catch (error) {
      const name = (error as { name?: string }).name;
      if (name !== "NoSuchKey" && name !== "NotFound") {
        console.error("Error fetching tile from S3:", (error as Error).message);
        return res.status(500).send("Failed to fetch tile");
      }
    }
  }
  res.status(404).send("Tile not found");
});

export default router;
