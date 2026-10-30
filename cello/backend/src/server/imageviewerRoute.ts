import { Router, Request, Response } from "express";
import { GetObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { getS3Client } from "./s3client.js";
import { STORAGE_MODE } from "./config.js";

/**
 * Legacy /view routes (s3 mode only): presigned URL for the raw slide object.
 * The old sharp-based ad-hoc tiler that lived here was dead code (hardcoded
 * /path/to/images) and has been removed; tiles come from /osd in both modes.
 */
const router = Router();

router.get("/image/:id/url", async (req: Request, res: Response) => {
  if (STORAGE_MODE !== "s3") {
    return res.status(400).json({ error: "s3_mode_only", message: "Presigned URLs need STORAGE_MODE=s3" });
  }
  const { id } = req.params;
  try {
    const command = new GetObjectCommand({ Bucket: process.env.S3_BUCKET as string, Key: id });
    const url = await getSignedUrl(getS3Client(), command, { expiresIn: 3600 });
    res.json({ message: "Image URL fetched successfully", imageUrl: url });
  } catch (error) {
    console.error("Error fetching image URL from S3:", error);
    res.status(500).json({ message: "Failed to fetch image URL" });
  }
});

export default router;
