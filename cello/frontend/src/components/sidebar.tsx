import React from "react";
import { Link } from "react-router-dom";
import FileUpload from "./FileUpload";
import Comments from "./comments";
import TriagePanel from "./TriagePanel";
import AskPanel, { AskEntry } from "./AskPanel";
import ImpactStrip from "./ImpactStrip";
import { Heatmap } from "../api/mlApi";
import { OverlayOptions, TileSelection } from "./HeatmapOverlay";

interface SidebarProps {
  selectedImageId?: string;
  onImageSelect: (imageId: string) => void;
  heatmap: Heatmap | null;
  onHeatmap: (hm: Heatmap | null) => void;
  overlay: OverlayOptions;
  onOverlay: (o: OverlayOptions) => void;
  hasGroundTruth: boolean;
  commentsRefreshKey: number;
  selectedTile: TileSelection | null;
  askHistory: AskEntry[];
  onAsk: (entry: AskEntry) => void;
}

const Sidebar: React.FC<SidebarProps> = ({ selectedImageId, onImageSelect, heatmap, onHeatmap, overlay, onOverlay, hasGroundTruth, commentsRefreshKey, selectedTile, askHistory, onAsk }) => {
  return (
    <aside className="sidebar">
      <div className="brand">
        <img src="/cello.png" alt="Cello" className="brand-logo" />
        <div>
          <div className="brand-name">Cello</div>
          <div className="brand-tag">AI slide triage</div>
        </div>
        <Link to="/demo" className="brand-link">Demo gallery →</Link>
      </div>
      <FileUpload selectedImageId={selectedImageId} onImageSelect={onImageSelect} />
      <ImpactStrip summary={heatmap?.summary ?? null} history={askHistory} />
      <TriagePanel imageId={selectedImageId} heatmap={heatmap} onHeatmap={onHeatmap} overlay={overlay} onOverlay={onOverlay} hasGroundTruth={hasGroundTruth} />
      <AskPanel imageId={selectedImageId} selection={selectedTile} heatmap={heatmap} history={askHistory} onAnswer={onAsk} />
      <Comments imageId={selectedImageId} refreshKey={commentsRefreshKey} />
    </aside>
  );
};

export default Sidebar;
