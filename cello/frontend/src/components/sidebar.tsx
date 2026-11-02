import React from "react";
import { Link } from "react-router-dom";
import FileUpload from "./FileUpload";
import Comments from "./comments";
import TriagePanel from "./TriagePanel";
import AskPanel, { AskEntry } from "./AskPanel";
import ImpactStrip from "./ImpactStrip";
import ScanPanel from "./ScanPanel";
import { Heatmap, ScanResult } from "../api/mlApi";
import { isAskItem, isScanItem, useGallery } from "../api/galleryApi";
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
  /** Verdict from the demo-gallery Scan panel (tints the viewer); null clears it. */
  onScanVerdict: (v: ScanResult["verdict"] | null) => void;
}

const Sidebar: React.FC<SidebarProps> = ({ selectedImageId, onImageSelect, heatmap, onHeatmap, overlay, onOverlay, hasGroundTruth, commentsRefreshKey, selectedTile, askHistory, onAsk, onScanVerdict }) => {
  const { byId } = useGallery();
  const galleryItem = selectedImageId ? byId[selectedImageId] ?? null : null;
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
      {isScanItem(galleryItem) && <ScanPanel item={galleryItem} onVerdict={onScanVerdict} />}
      <TriagePanel imageId={selectedImageId} heatmap={heatmap} onHeatmap={onHeatmap} overlay={overlay} onOverlay={onOverlay} hasGroundTruth={hasGroundTruth} />
      <AskPanel imageId={selectedImageId} selection={selectedTile} heatmap={heatmap} history={askHistory} onAnswer={onAsk} galleryItem={isAskItem(galleryItem) ? galleryItem : null} />
      <Comments imageId={selectedImageId} refreshKey={commentsRefreshKey} />
    </aside>
  );
};

export default Sidebar;
