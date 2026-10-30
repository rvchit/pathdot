# Cello backend API

Base URL (dev): `http://localhost:4001`. No authentication. All JSON unless noted.
`STORAGE_MODE=local` (default) serves everything from disk; `STORAGE_MODE=s3` keeps the
original S3 behaviour for tiles and uploads. The browser only ever talks to this server;
the ML service is reached through the `/ml/*` proxy.

## Health

| Method | Path | Response |
|---|---|---|
| GET | `/health` | `{"status":"ok","storageMode":"local","tilesDir":"…","mlServiceUrl":"http://localhost:8008","uptimeSec":12}` |

## Slides and tiles (`/osd`)

Local layout on disk (`TILES_DIR`, default `<repo>/tiles`):

```
tiles/<imageId>/<imageId>.dzi            # or tiles/<imageId>.dzi — both are accepted
tiles/<imageId>_files/<level>/<x>_<y>.jpeg
```

| Method | Path | Response |
|---|---|---|
| GET | `/osd/images` | `{"images":["pcam_mosaic_24", …],"tilesDir":"…"}` — every imageId with a `.dzi` |
| GET | `/osd/dzi/:imageId` | `{"imageId","width","height","tileSize","overlap","format","maxLevel"}` parsed from the `.dzi` XML; `maxLevel = ceil(log2(max(width,height)))` (DZI numbering, level 0 = 1 px). 404 if no `.dzi`. |
| GET | `/osd/tile/:imageId/:level/:x/:y` | the tile bytes (`image/jpeg`, or png/webp if that is what is on disk). 404 when missing, 400 on unsafe ids or non-numeric coordinates. |
| GET | `/osd/gt/:imageId` | raw contents of `data/slides/<imageId>.gt.json` (ground-truth cells written by `scripts/make_mosaic_slide.py`). 404 if absent. Local mode only. |

`imageId` must match `^[A-Za-z0-9_][A-Za-z0-9._-]*$`; `/`, `\` and `..` are rejected, and the
resolved path is checked to stay inside `TILES_DIR`.

Example OpenSeadragon tile source (what the frontend does):

```js
const dzi = await fetch(`${BACKEND}/osd/dzi/${id}`).then(r => r.json());
OpenSeadragon({ tileSources: {
  width: dzi.width, height: dzi.height, tileSize: dzi.tileSize, tileOverlap: dzi.overlap,
  minLevel: 0, maxLevel: dzi.maxLevel,
  getTileUrl: (level, x, y) => `${BACKEND}/osd/tile/${id}/${level}/${x}/${y}`,
}});
```

## Comments (`/comments`)

Stored in `<DATA_DIR>/comments.json` (atomic write). See `src/server/commentsRoute.ts` for the
Prisma switch.

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/comments/:imageId` | – | `Comment[]` |
| POST | `/comments` | `{"imageId","annotationId","comment","geometry"?}` | `201` + the stored `Comment` |
| DELETE | `/comments/:id` | – | `{"deleted":true,"id"}` or 404 |

```json
{"id":"3e80b7d2-…","imageId":"pcam_mosaic_24","annotationId":"a1","comment":"hello",
 "geometry":{"type":"RECTANGLE","geometry":{…}},"createdAt":"2026-10-30T06:57:54.231Z"}
```

## ML proxy (`/ml/*`)

Every method and path under `/ml` is forwarded verbatim to `ML_SERVICE_URL` (default
`http://localhost:8008`), streaming the raw body (JSON or multipart) and returning the upstream
status code and headers. If the service is unreachable the backend answers
`502 {"error":"ml_service_unreachable","message":"…"}`.

| Method | Path (via proxy) | Upstream contract (docs/CONTRACTS.md §8) |
|---|---|---|
| GET | `/ml/health` | `{"status":"ok","encoder","probe","vlm","device"}` |
| POST | `/ml/predict/tile` | multipart `file` or JSON `{"image_b64"}` → `{"p_tumor","probs","classes","decision","routed_to","reason","latency_ms","cost_usd"}` |
| POST | `/ml/predict/slide` | `{"image_id","low","high","fallback","level"?,"max_tiles"?}` → `{"job_id","status":"running"}` |
| GET | `/ml/jobs/:jobId` | `{"status":"running\|done\|error","progress":0..1,"result":<heatmap>\|null}` |
| GET | `/ml/heatmap/:imageId` | last saved heatmap, or 404 |

Heatmap JSON (what the overlay draws): `{image_id,width,height,tile_size,level,max_level,scale,cols,rows,
tiles:[{x,y,p_tumor,decision,routed_to,reason}],summary:{n_tiles,auto_cleared,auto_flagged,review,
frac_auto_cleared,cost_usd,latency_ms_total,pathologist_minutes_saved,model}}`. A tile covers
`[x*tile_size*scale, y*tile_size*scale, tile_size*scale, tile_size*scale]` image pixels.

`cello/backend/scripts/ml-stub.mjs` (`npm run ml-stub`) is a 100-line stand-in that answers all
of these with canned data; `STUB_JOB_MS` controls how long a fake slide job takes.

## Uploads (`/api`)

| Mode | Method | Path | Body | Response |
|---|---|---|---|---|
| any | GET | `/api/upload-mode` | – | `{"mode":"local\|s3","uploadDir"}` |
| local | POST | `/api/upload-local` | multipart `file` | `201 {"message","imageId","path","size","nextStep":"uv run python scripts/tile_slide.py data/slides/<file> --out tiles/"}` |
| s3 | POST | `/api/start-upload` | `{"fileName","fileType"}` | `{"uploadId"}` |
| s3 | POST | `/api/upload-part` | multipart `fileChunk` + `fileName`,`partNumber`,`uploadId` | `{"ETag"}` |
| s3 | POST | `/api/complete-upload` | `{"fileName","uploadId","parts":[{"ETag","PartNumber"}]}` | `{"fileUrl"}` |

The s3 endpoints answer `400 {"error":"s3_mode_only"}` in local mode. The local upload saves the raw
file to `UPLOAD_DIR` (default `<repo>/data/slides`) and does **not** tile it; run the printed
`nextStep` (or `make tile` for the demo mosaic) and the slide appears in `/osd/images`.

## Legacy / mock

| Method | Path | Notes |
|---|---|---|
| POST | `/api/auth/register`, `/api/auth/login` | in-memory bcrypt mock, not wired into the UI |
| GET | `/view/image/:id/url` | presigned S3 URL for the raw slide; s3 mode only |

## Errors

| Code | Meaning |
|---|---|
| 400 | bad/unsafe id, missing field, wrong storage mode |
| 404 | no such slide / tile / comment / ground truth |
| 502 | ML service unreachable (`/ml/*` only) |
| 500 | unexpected server error (logged to stdout) |
