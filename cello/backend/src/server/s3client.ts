import { S3Client } from "@aws-sdk/client-s3";
import { STORAGE_MODE } from "./config.js";

/**
 * Lazily constructed S3 client. Only ever called from s3-mode code paths, so
 * local mode never requires AWS credentials and never loads a client.
 */
let client: S3Client | null = null;

export function getS3Client(): S3Client {
  if (STORAGE_MODE !== "s3") {
    throw new Error("S3 client requested while STORAGE_MODE=local");
  }
  if (!client) {
    const { AWS_REGION, AWS_ACCESS_KEY, AWS_SECRET_KEY } = process.env;
    if (!AWS_REGION || !AWS_ACCESS_KEY || !AWS_SECRET_KEY) {
      throw new Error(
        "STORAGE_MODE=s3 requires AWS_REGION, AWS_ACCESS_KEY and AWS_SECRET_KEY (see .env.example).",
      );
    }
    client = new S3Client({
      region: AWS_REGION,
      credentials: { accessKeyId: AWS_ACCESS_KEY, secretAccessKey: AWS_SECRET_KEY },
    });
  }
  return client;
}

export function requireS3Env(): void {
  if (STORAGE_MODE !== "s3") return;
  const missing = ["AWS_REGION", "AWS_ACCESS_KEY", "AWS_SECRET_KEY", "S3_TILES_BUCKET", "S3_BUCKET"].filter(
    (k) => !process.env[k],
  );
  if (missing.length) {
    throw new Error(`STORAGE_MODE=s3 but missing env: ${missing.join(", ")}`);
  }
}
