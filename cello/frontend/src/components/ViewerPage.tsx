import React, { useEffect, useState } from "react";
import ImageViewer from "./OSDviewer";
import Sidebar from "./sidebar";
import { DEFAULT_OVERLAY, OverlayOptions } from "./HeatmapOverlay";
import { Heatmap } from "../api/mlApi";
import { DziInfo, GroundTruth, getGroundTruth } from "../api/slidesApi";

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

  // Reset per-slide state when the slide changes.
  useEffect(() => {
    setHeatmap(null);
    setGroundTruth(null);
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
      <main className="viewer-main">
        <ImageViewer
          imageId={selectedImageId}
          heatmap={heatmap}
          groundTruth={groundTruth}
          overlay={overlay}
          onDzi={setDzi}
          onCommentSaved={() => setCommentsRefreshKey((k) => k + 1)}
        />
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
      />
    </div>
  );
};

export default ViewerPage;
