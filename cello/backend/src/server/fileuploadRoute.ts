import express, { Request, Response } from "express";
import {
  CreateMultipartUploadCommand,
  UploadPartCommand,
  CompleteMultipartUploadCommand,
} from "@aws-sdk/client-s3";
import multer from "multer";
import fs from "fs";
import path from "path";
import { STORAGE_MODE, UPLOAD_DIR, TILES_DIR } from "./config.js";
import { getS3Client } from "./s3client.js";

/**
 * Upload routes under /api.
 *
 * Local mode (default):
 *   POST /api/upload-local  multipart field "file"  -> saves to UPLOAD_DIR/<name>
 *     and replies with the exact tiler command to run next. The tiler
 *     (scripts/tile_slide.py) writes the DZI pyramid into TILES_DIR, after which
 *     the image shows up in GET /osd/images.
 *
 * S3 mode: the original three-step multipart upload (start-upload, upload-part,
 * complete-upload) into S3_BUCKET. Kept intact; only the client is now lazy.
 */
const router = express.Router();

const MAX_UPLOAD_MB = Number(process.env.MAX_UPLOAD_MB ?? 2048);

function safeFilename(name: string): string {
  const base = path.basename(name).replace(/[^A-Za-z0-9._-]+/g, "_");
  return base.replace(/^\.+/, "") || `upload_${Date.now()}`;
}

// ---------- local mode ----------
const localStorage = multer.diskStorage({
  destination: (_req, _file, cb) => {
    fs.mkdirSync(UPLOAD_DIR, { recursive: true });
    cb(null, UPLOAD_DIR);
  },
  filename: (_req, file, cb) => cb(null, safeFilename(file.originalname)),
});
const localUpload = multer({
  storage: localStorage,
  limits: { fileSize: MAX_UPLOAD_MB * 1024 * 1024 },
});

router.post("/upload-local", localUpload.single("file"), (req: Request, res: Response) => {
  if (!req.file) return res.status(400).json({ error: "Missing multipart field 'file'" });
  const savedPath = req.file.path;
  const imageId = path.parse(req.file.filename).name;
  const relSlide = path.relative(path.resolve(TILES_DIR, ".."), savedPath);
  const relTiles = path.relative(path.resolve(TILES_DIR, ".."), TILES_DIR);
  res.status(201).json({
    message: "File saved. Run the tiler to build the DZI pyramid, then it will appear in /osd/images.",
    imageId,
    path: savedPath,
    size: req.file.size,
    nextStep: `uv run python scripts/tile_slide.py ${relSlide} --out ${relTiles}/`,
  });
});

router.get("/upload-mode", (_req: Request, res: Response) => {
  res.json({ mode: STORAGE_MODE, uploadDir: STORAGE_MODE === "local" ? UPLOAD_DIR : null });
});

// ---------- s3 mode (original multipart flow) ----------
const memUpload = multer({ limits: { fileSize: 100 * 1024 * 1024 } });

function s3Only(_req: Request, res: Response, next: express.NextFunction) {
  if (STORAGE_MODE !== "s3") {
    return res.status(400).json({
      error: "s3_mode_only",
      message: "This endpoint needs STORAGE_MODE=s3. In local mode POST a file to /api/upload-local.",
    });
  }
  next();
}

router.post("/start-upload", s3Only, async (req: Request, res: Response) => {
  const { fileName, fileType }: { fileName: string; fileType: string } = req.body;
  try {
    const upload = await getS3Client().send(
      new CreateMultipartUploadCommand({ Bucket: process.env.S3_BUCKET!, Key: fileName, ContentType: fileType }),
    );
    res.send({ uploadId: upload.UploadId });
  } catch (error) {
    console.error("Error starting upload", error);
    res.status(500).send(error);
  }
});

router.post("/upload-part", s3Only, memUpload.single("fileChunk"), async (req: Request, res: Response) => {
  if (!req.file) return res.status(400).send({ message: "Missing file chunk" });
  const { fileName, partNumber, uploadId } = req.body as { fileName: string; partNumber: string; uploadId: string };
  try {
    const uploadParts = await getS3Client().send(
      new UploadPartCommand({
        Bucket: process.env.S3_BUCKET!,
        Key: fileName,
        PartNumber: Number(partNumber),
        UploadId: uploadId,
        Body: req.file.buffer,
      }),
    );
    res.send({ ETag: uploadParts.ETag });
  } catch (error) {
    console.error("Error uploading part", error);
    res.status(500).send(error);
  }
});

router.post("/complete-upload", s3Only, async (req: Request, res: Response) => {
  const { fileName, uploadId, parts } = req.body as {
    fileName: string;
    uploadId: string;
    parts: Array<{ ETag: string; PartNumber: number }>;
  };
  try {
    const complete = await getS3Client().send(
      new CompleteMultipartUploadCommand({
        Bucket: process.env.S3_BUCKET!,
        Key: fileName,
        UploadId: uploadId,
        MultipartUpload: { Parts: parts },
      }),
    );
    res.send({ fileUrl: complete.Location });
  } catch (error) {
    console.error("Error completing upload", error);
    res.status(500).send(error);
  }
});

export default router;
