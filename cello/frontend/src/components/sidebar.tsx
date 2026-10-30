import React from "react";
import FileUpload from "./FileUpload";
import Comments from "./comments";
import TriagePanel from "./TriagePanel";
import { Heatmap } from "../api/mlApi";
import { OverlayOptions } from "./HeatmapOverlay";

interface SidebarProps {
  selectedImageId?: string;
  onImageSelect: (imageId: string) => void;
  heatmap: Heatmap | null;
  onHeatmap: (hm: Heatmap | null) => void;
  overlay: OverlayOptions;
  onOverlay: (o: OverlayOptions) => void;
  hasGroundTruth: boolean;
  commentsRefreshKey: number;
}

const Sidebar: React.FC<SidebarProps> = ({ selectedImageId, onImageSelect, heatmap, onHeatmap, overlay, onOverlay, hasGroundTruth, commentsRefreshKey }) => {
  return (
    <aside className="sidebar">
      <div className="brand">
        <img src="/cello.png" alt="Cello" className="brand-logo" />
        <div>
          <div className="brand-name">Cello</div>
          <div className="brand-tag">AI slide triage</div>
        </div>
      </div>
      <FileUpload selectedImageId={selectedImageId} onImageSelect={onImageSelect} />
      <TriagePanel imageId={selectedImageId} heatmap={heatmap} onHeatmap={onHeatmap} overlay={overlay} onOverlay={onOverlay} hasGroundTruth={hasGroundTruth} />
      <Comments imageId={selectedImageId} refreshKey={commentsRefreshKey} />
    </aside>
  );
};

export default Sidebar;
