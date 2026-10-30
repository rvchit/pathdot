import express from "express";
import cors from "cors";
import api from "./fileuploadRoute.js";
import view from "./imageviewerRoute.js";
import s3Osd from "./OSDroute.js";
import localOsd from "./localTilesRoute.js";
import comments from "./commentsRoute.js";
import mlProxy from "./mlProxyRoute.js";
import authRoutes from "../backend/authk.js";
import {
  STORAGE_MODE,
  PORT,
  TILES_DIR,
  UPLOAD_DIR,
  DATA_DIR,
  ML_SERVICE_URL,
  JSON_BODY_LIMIT,
  CORS_ORIGIN,
} from "./config.js";
import { requireS3Env } from "./s3client.js";

requireS3Env(); // throws only when STORAGE_MODE=s3 and AWS env is incomplete

const app = express();

app.use(
  cors({
    origin: CORS_ORIGIN === "*" ? true : CORS_ORIGIN.split(",").map((s) => s.trim()),
    methods: ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
  }),
);

// The ML proxy streams the raw request body, so it is mounted BEFORE the body parsers.
app.use("/ml", mlProxy);

app.use(express.json({ limit: JSON_BODY_LIMIT }));
app.use(express.urlencoded({ extended: true, limit: JSON_BODY_LIMIT }));

app.get("/health", (_req, res) => {
  res.json({
    status: "ok",
    storageMode: STORAGE_MODE,
    tilesDir: STORAGE_MODE === "local" ? TILES_DIR : null,
    mlServiceUrl: ML_SERVICE_URL,
    uptimeSec: Math.round(process.uptime()),
  });
});

app.use("/api", api);
app.use("/api/auth", authRoutes);
app.use("/view", view);
app.use("/osd", STORAGE_MODE === "s3" ? s3Osd : localOsd);
app.use("/comments", comments);

app.use((_req, res) => res.status(404).json({ error: "not_found" }));

app.listen(PORT, () => {
  console.log(`cello backend listening on http://localhost:${PORT}`);
  console.log(`  STORAGE_MODE=${STORAGE_MODE}`);
  if (STORAGE_MODE === "local") {
    console.log(`  TILES_DIR=${TILES_DIR}`);
    console.log(`  UPLOAD_DIR=${UPLOAD_DIR}`);
    console.log(`  DATA_DIR=${DATA_DIR}`);
  } else {
    console.log(`  S3_TILES_BUCKET=${process.env.S3_TILES_BUCKET} (${process.env.AWS_REGION})`);
  }
  console.log(`  ML_SERVICE_URL=${ML_SERVICE_URL} (proxied at /ml/*)`);
});
