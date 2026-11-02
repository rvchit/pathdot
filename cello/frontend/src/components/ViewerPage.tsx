import React, { useEffect, useState } from "react";
import ImageViewer from "./OSDviewer";
import Sidebar from "./sidebar";
import { DEFAULT_OVERLAY, OverlayOptions, TileSelection } from "./HeatmapOverlay";
import { AskEntry } from "./AskPanel";
import { Heatmap, ScanResult } from "../api/mlApi";
import { DziInfo, GroundTruth, getGroundTruth } from "../api/slidesApi";
import "./viewer-extras.css";

interface ViewerPageProps {
  selectedImageId?: string;
  setSelectedImageId: React.Dispatch<React.SetStateAction<string | undefined>>;
}

const ViewerPage: React.FC<ViewerPageProps> = ({ selectedImageId, setSelectedImageId }) => {
  const [heatmap, setHeatmap] = useState<Heatmap | null>(null);
  const [dzi, setDzi] = useState<DziInfo | null>(null);
  const [groundTruth, setGroundTruth] = useState<GroundTruth | null>(null);
  const [overlay, setOverlay] = useState<OverlayOptions>(DEFAULT_OVERLAY);
  const [commentsRefreshKey, setCommentsRefreshKey] = useState(0);
  // Tile the user clicked on the slide (drives the "Ask the pathology assistant" panel).
  const [selectedTile, setSelectedTile] = useState<TileSelection | null>(null);
  // Session-wide Q&A history (feeds the AskPanel list and the ImpactStrip stats).
  const [askHistory, setAskHistory] = useState<AskEntry[]>([]);
  // Verdict from the demo-gallery Scan panel; tints the whole slide red/green.
  const [scanVerdict, setScanVerdict] = useState<ScanResult["verdict"] | null>(null);

  // Reset per-slide state when the slide changes.
  useEffect(() => {
    setHeatmap(null);
    setGroundTruth(null);
    setSelectedTile(null);
    setScanVerdict(null);
    setOverlay((o) => ({ ...o, showGroundTruth: false }));
  }, [selectedImageId]);

  // Ground truth needs the slide size (from the DZI) to place cells.
  useEffect(() => {
    if (!selectedImageId || !dzi || dzi.imageId !== selectedImageId) return;
    let cancelled = false;
    getGroundTruth(selectedImageId, dzi.width, dzi.height)
      .then((gt) => {
        if (cancelled) return;
        setGroundTruth(gt);
        // /?image=<id>&gt=1 -> ground truth on from the start (demo deep link)
        if (gt && new URLSearchParams(window.location.search).get("gt") === "1") setOverlay((o) => ({ ...o, showGroundTruth: true }));
      })
      .catch(() => {
        if (!cancelled) setGroundTruth(null);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedImageId, dzi]);

  return (
    <div className="app-shell">
      <main className="viewer-main" style={{ position: "relative" }}>
        <ImageViewer
          imageId={selectedImageId}
          heatmap={heatmap}
          groundTruth={groundTruth}
          overlay={overlay}
          onDzi={setDzi}
          onCommentSaved={() => setCommentsRefreshKey((k) => k + 1)}
          selectedTile={selectedTile}
          onSelectTile={setSelectedTile}
        />
        {/* Whole-slide verdict tint (demo gallery scan). Sits over the viewer; clicks pass through. */}
        <div className={`verdict-tint ${scanVerdict ?? ""} ${scanVerdict ? "show" : ""}`} aria-hidden={!scanVerdict}>
          {scanVerdict && <span className="verdict-tint-label">{scanVerdict === "tumor" ? "TUMOR DETECTED" : "NO TUMOR"}</span>}
        </div>
      </main>
      <Sidebar
        selectedImageId={selectedImageId}
        onImageSelect={setSelectedImageId}
        heatmap={heatmap}
        onHeatmap={setHeatmap}
        overlay={overlay}
        onOverlay={setOverlay}
        hasGroundTruth={!!groundTruth && groundTruth.cells.length > 0}
        commentsRefreshKey={commentsRefreshKey}
        selectedTile={selectedTile}
        askHistory={askHistory}
        onAsk={(e) => setAskHistory((h) => [e, ...h])}
        onScanVerdict={setScanVerdict}
      />
    </div>
  );
};

export default ViewerPage;
