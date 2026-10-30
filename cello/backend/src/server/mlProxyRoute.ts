import { Router, Request, Response } from "express";
import http from "http";
import https from "https";
import { ML_SERVICE_URL } from "./config.js";

/**
 * Transparent proxy: ALL /ml/* -> ML_SERVICE_URL/* (pathfrontier.serve, FastAPI).
 *
 * The raw request stream is piped through untouched so JSON bodies, multipart
 * uploads (POST /ml/predict/tile with an image file) and query strings all work.
 * This router MUST be mounted before express.json() so the body is still unread.
 * Status codes and response headers are forwarded as-is.
 */
const router = Router();

const HOP_BY_HOP = new Set([
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
  "host",
]);

router.all("*", (req: Request, res: Response) => {
  const target = new URL(ML_SERVICE_URL);
  const upstreamPath = (req.baseUrl ? req.originalUrl.slice(req.baseUrl.length) : req.originalUrl) || "/";
  const isHttps = target.protocol === "https:";
  const client = isHttps ? https : http;

  const headers: http.OutgoingHttpHeaders = {};
  for (const [k, v] of Object.entries(req.headers)) {
    if (!HOP_BY_HOP.has(k.toLowerCase()) && v !== undefined) headers[k] = v;
  }
  headers["host"] = target.host;
  headers["x-forwarded-for"] = req.ip ?? "";
  headers["x-forwarded-proto"] = req.protocol;

  const upstream = client.request(
    {
      protocol: target.protocol,
      hostname: target.hostname,
      port: target.port || (isHttps ? 443 : 80),
      method: req.method,
      path: upstreamPath,
      headers,
    },
    (upstreamRes) => {
      const outHeaders: Record<string, string | string[]> = {};
      for (const [k, v] of Object.entries(upstreamRes.headers)) {
        if (!HOP_BY_HOP.has(k.toLowerCase()) && v !== undefined) outHeaders[k] = v;
      }
      res.writeHead(upstreamRes.statusCode ?? 502, outHeaders);
      upstreamRes.pipe(res);
    },
  );

  upstream.on("error", (err: NodeJS.ErrnoException) => {
    console.error(`ML proxy error (${req.method} ${upstreamPath} -> ${ML_SERVICE_URL}):`, err.message);
    if (!res.headersSent) {
      res.status(502).json({
        error: "ml_service_unreachable",
        message: `Could not reach the ML service at ${ML_SERVICE_URL} (${err.code ?? err.message}). ` +
          "Start it with `make dev` or `uvicorn pathfrontier.serve:app --port 8008`.",
      });
    } else {
      res.end();
    }
  });

  req.on("aborted", () => upstream.destroy());
  req.pipe(upstream);
});

export default router;
