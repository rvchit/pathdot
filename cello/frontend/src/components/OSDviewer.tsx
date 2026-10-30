import React, { useEffect, useRef, useState } from "react";
import OpenSeadragon from "openseadragon";
import { createOSDAnnotator, ImageAnnotation, OpenSeadragonAnnotator } from "@annotorious/openseadragon";
import "@annotorious/openseadragon/annotorious-openseadragon.css";
import { BACKEND_URL, errorMessage } from "../api/client";
import { DziInfo, getDzi, createComment } from "../api/slidesApi";
import { Heatmap } from "../api/mlApi";
import HeatmapOverlay, { OverlayOptions } from "./HeatmapOverlay";
import { GroundTruth } from "../api/slidesApi";

interface ImageViewerProps {
  imageId?: string;
  heatmap: Heatmap | null;
  groundTruth: GroundTruth | null;
  overlay: OverlayOptions;
  onDzi?: (info: DziInfo | null) => void;
  onCommentSaved?: () => void;
}

type DrawTool = "rectangle" | "polygon";

const OSD_PREFIX = "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/5.0.0/images/";

const OSDViewer: React.FC<ImageViewerProps> = ({ imageId, heatmap, groundTruth, overlay, onDzi, onCommentSaved }) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const annotatorRef = useRef<OpenSeadragonAnnotator | null>(null);
  const [viewer, setViewer] = useState<OpenSeadragon.Viewer | null>(null);
  const [dzi, setDzi] = useState<DziInfo | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [drawing, setDrawing] = useState(false);
  const [annotationsAvailable, setAnnotationsAvailable] = useState(true);
  const [tool, setTool] = useState<DrawTool>("rectangle");
  const [pendingAnnotation, setPendingAnnotation] = useState<ImageAnnotation | null>(null);
  const [comment, setComment] = useState("");
  const [saving, setSaving] = useState(false);

  // 1. Fetch DZI metadata whenever the image changes.
  useEffect(() => {
    let cancelled = false;
    setDzi(null);
    setLoadError(null);
    onDzi?.(null);
    if (!imageId) return;
    getDzi(imageId)
      .then((info) => {
        if (cancelled) return;
        setDzi(info);
        onDzi?.(info);
      })
      .catch((err) => {
        if (!cancelled) setLoadError(errorMessage(err));
      });
    return () => {
      cancelled = true;
    };
  }, [imageId]); // onDzi is a stable setter from ViewerPage

  // 2. Build the OpenSeadragon viewer from the real pyramid geometry.
  useEffect(() => {
    if (!containerRef.current || !dzi || !imageId) return;
    const id = imageId;
    const v = OpenSeadragon({
      element: containerRef.current,
      prefixUrl: OSD_PREFIX,
      tileSources: {
        width: dzi.width,
        height: dzi.height,
        tileSize: dzi.tileSize,
        tileOverlap: dzi.overlap,
        minLevel: 0,
        maxLevel: dzi.maxLevel,
        getTileUrl: (level: number, x: number, y: number) => `${BACKEND_URL}/osd/tile/${encodeURIComponent(id)}/${level}/${x}/${y}`,
      },
      showNavigator: true,
      navigatorPosition: "BOTTOM_LEFT",
      navigatorSizeRatio: 0.15,
      crossOriginPolicy: "Anonymous",
      visibilityRatio: 0.8,
      minZoomImageRatio: 0.6,
      maxZoomPixelRatio: 4,
      animationTime: 0.4,
      gestureSettingsMouse: { clickToZoom: false },
      showZoomControl: true,
      showHomeControl: true,
      showFullPageControl: false,
      showRotationControl: false,
    });
    // Annotorious needs WebGL; if the browser has none (headless, remote desktop)
    // we still want the slide + heatmap, just without drawing tools.
    let annotator: OpenSeadragonAnnotator | null = null;
    try {
      annotator = createOSDAnnotator(v, { drawingEnabled: false });
      annotator.on("createAnnotation", (annotation) => setPendingAnnotation(annotation));
    } catch (err) {
      console.warn("Annotorious unavailable (no WebGL?):", err);
    }
    annotatorRef.current = annotator;
    setAnnotationsAvailable(!!annotator);
    setViewer(v);
    setDrawing(false);

    return () => {
      annotator?.destroy();
      annotatorRef.current = null;
      v.destroy();
      setViewer(null);
    };
  }, [dzi, imageId]);

  const toggleDrawing = (next: boolean) => {
    annotatorRef.current?.setDrawingEnabled(next);
    setDrawing(next);
  };
  const pickTool = (t: DrawTool) => {
    annotatorRef.current?.setDrawingTool(t);
    setTool(t);
    if (!drawing) toggleDrawing(true);
  };

  const saveComment = async () => {
    if (!pendingAnnotation || !imageId || !comment.trim()) return;
    setSaving(true);
    try {
      await createComment({
        imageId,
        annotationId: pendingAnnotation.id,
        comment: comment.trim(),
        geometry: pendingAnnotation.target?.selector,
      });
      annotatorRef.current?.updateAnnotation({
        ...pendingAnnotation,
        bodies: [
          ...(pendingAnnotation.bodies ?? []),
          { id: `${pendingAnnotation.id}#c`, annotation: pendingAnnotation.id, purpose: "commenting", value: comment.trim() },
        ],
      });
      setComment("");
      setPendingAnnotation(null);
      onCommentSaved?.();
    } catch (err) {
      alert(`Could not save comment: ${errorMessage(err)}`);
    } finally {
      setSaving(false);
    }
  };

  const cancelComment = () => {
    if (pendingAnnotation) annotatorRef.current?.removeAnnotation(pendingAnnotation.id);
    setComment("");
    setPendingAnnotation(null);
  };

  return (
    <div className="viewer-wrap">
      <div className="viewer-toolbar">
        <span className="viewer-title">{imageId ?? "No slide selected"}</span>
        {dzi && (
          <span className="viewer-meta">
            {dzi.width} x {dzi.height} px · tile {dzi.tileSize} · {dzi.maxLevel + 1} levels
          </span>
        )}
        <span className="spacer" />
        <button className={`btn ${drawing ? "btn-active" : ""}`} onClick={() => toggleDrawing(!drawing)} disabled={!viewer || !annotationsAvailable} title={annotationsAvailable ? "Draw a region, then add a comment" : "Annotations need WebGL"}>
          {drawing ? "Drawing on" : "Annotate"}
        </button>
        <button className={`btn ${tool === "rectangle" && drawing ? "btn-active" : ""}`} onClick={() => pickTool("rectangle")} disabled={!viewer || !annotationsAvailable}>
          Rect
        </button>
        <button className={`btn ${tool === "polygon" && drawing ? "btn-active" : ""}`} onClick={() => pickTool("polygon")} disabled={!viewer || !annotationsAvailable}>
          Polygon
        </button>
      </div>

      <div className="viewer-stage">
        {!imageId && <div className="viewer-empty">Pick a slide from the list on the right.</div>}
        {imageId && loadError && (
          <div className="viewer-empty viewer-error">
            Could not load <code>{imageId}</code>: {loadError}
          </div>
        )}
        {imageId && !dzi && !loadError && <div className="viewer-empty">Loading slide metadata…</div>}
        <div ref={containerRef} className="osd-container" />
        <HeatmapOverlay viewer={viewer} heatmap={heatmap} groundTruth={groundTruth} options={overlay} />
      </div>

      {pendingAnnotation && (
        <div className="modal-backdrop">
          <div className="modal">
            <h3>Add a comment</h3>
            <textarea value={comment} onChange={(e) => setComment(e.target.value)} placeholder="What do you see in this region?" autoFocus />
            <div className="modal-actions">
              <button className="btn" onClick={cancelComment} disabled={saving}>
                Cancel
              </button>
              <button className="btn btn-primary" onClick={saveComment} disabled={saving || !comment.trim()}>
                {saving ? "Saving…" : "Save"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default OSDViewer;
