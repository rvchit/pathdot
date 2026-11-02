import React, { useCallback, useEffect, useRef, useState } from "react";
import OpenSeadragon from "openseadragon";
import { Heatmap, HeatmapTile } from "../api/mlApi";
import { DziInfo, GroundTruth } from "../api/slidesApi";

export interface OverlayOptions {
  showHeatmap: boolean;
  opacity: number; // 0..1 multiplier on the per-class alpha
  showGroundTruth: boolean;
}

export const DEFAULT_OVERLAY: OverlayOptions = { showHeatmap: true, opacity: 1, showGroundTruth: false };

/** A DZI tile the user clicked: `level`/`x`/`y` are DZI tile indices (tile size 256). */
export interface TileSelection {
  level: number;
  x: number;
  y: number;
  /** The heatmap tile at that position, when a heatmap exists at the same level. */
  tile?: HeatmapTile | null;
}

interface Props {
  viewer: OpenSeadragon.Viewer | null;
  heatmap: Heatmap | null;
  groundTruth: GroundTruth | null;
  options: OverlayOptions;
  /** Slide geometry; needed to select tiles when there is no heatmap yet. */
  dzi?: DziInfo | null;
  selected?: TileSelection | null;
  /** Fired on click (pin) with the tile, and with null when unpinned / clicked outside the slide. */
  onSelectTile?: (sel: TileSelection | null) => void;
}

const DZI_TILE = 256;

/** Image-pixel rectangle of a DZI tile at `level` (scale = 2^(maxLevel-level)). */
function dziTileRect(dzi: DziInfo, sel: TileSelection): { x: number; y: number; w: number; h: number } {
  const size = (dzi.tileSize || DZI_TILE) * Math.pow(2, dzi.maxLevel - sel.level);
  const x = sel.x * size;
  const y = sel.y * size;
  return { x, y, w: Math.min(size, dzi.width - x), h: Math.min(size, dzi.height - y) };
}

/** Colour per decision. Alpha is the contract's base value, scaled by the opacity slider. */
export const DECISION_STYLE: Record<HeatmapTile["decision"], { rgb: string; alpha: number; label: string }> = {
  positive: { rgb: "220, 38, 38", alpha: 0.45, label: "flagged" },
  review: { rgb: "245, 158, 11", alpha: 0.45, label: "review" },
  negative: { rgb: "34, 197, 94", alpha: 0.15, label: "cleared" },
};

const GT_TUMOR = "rgba(239, 68, 68, 0.95)";
const GT_OTHER = "rgba(148, 163, 184, 0.45)";

interface Hover {
  tile: HeatmapTile;
  px: number;
  py: number;
  pinned: boolean;
}

/** Image-pixel rectangle of a heatmap tile, clipped to the slide bounds. */
export function tileRect(hm: Heatmap, t: HeatmapTile): { x: number; y: number; w: number; h: number } {
  const size = hm.tile_size * hm.scale;
  const x = t.x * size;
  const y = t.y * size;
  return { x, y, w: Math.min(size, hm.width - x), h: Math.min(size, hm.height - y) };
}

/**
 * A single canvas laid over the OpenSeadragon element. Every tile rect is converted
 * image px -> viewer element px on each 'update-viewport'/'resize', so it stays glued
 * to the slide while panning and zooming. pointer-events are off so OSD keeps gestures;
 * hover/click are read from the viewer's own DOM element instead.
 */
const HeatmapOverlay: React.FC<Props> = ({ viewer, heatmap, groundTruth, options, dzi, selected, onSelectTile }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hover, setHover] = useState<Hover | null>(null);
  const hoverRef = useRef<Hover | null>(null);
  hoverRef.current = hover;
  const selectRef = useRef(onSelectTile);
  selectRef.current = onSelectTile;
  const selectedRef = useRef(selected);
  selectedRef.current = selected;

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !viewer || !viewer.viewport || !viewer.world) return;
    const el = viewer.element;
    const dpr = window.devicePixelRatio || 1;
    const cssW = el.clientWidth;
    const cssH = el.clientHeight;
    if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssH * dpr);
      canvas.style.width = `${cssW}px`;
      canvas.style.height = `${cssH}px`;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    if (viewer.world.getItemCount() === 0) return;

    const vp = viewer.viewport;
    const toEl = (x: number, y: number) => vp.imageToViewerElementCoordinates(new OpenSeadragon.Point(x, y));

    if (heatmap && options.showHeatmap) {
      for (const t of heatmap.tiles) {
        const style = DECISION_STYLE[t.decision];
        if (!style) continue; // "background" or unknown -> not painted
        const r = tileRect(heatmap, t);
        const a = toEl(r.x, r.y);
        const b = toEl(r.x + r.w, r.y + r.h);
        const w = b.x - a.x;
        const h = b.y - a.y;
        if (b.x < 0 || b.y < 0 || a.x > cssW || a.y > cssH) continue;
        ctx.fillStyle = `rgba(${style.rgb}, ${Math.min(1, style.alpha * options.opacity)})`;
        ctx.fillRect(a.x, a.y, w, h);
        if (w > 14) {
          ctx.strokeStyle = `rgba(${style.rgb}, ${Math.min(1, 0.6 * options.opacity)})`;
          ctx.lineWidth = 1;
          ctx.strokeRect(a.x + 0.5, a.y + 0.5, w - 1, h - 1);
        }
      }
      const hv = hoverRef.current;
      if (hv && !hv.pinned) {
        const r = tileRect(heatmap, hv.tile);
        const a = toEl(r.x, r.y);
        const b = toEl(r.x + r.w, r.y + r.h);
        ctx.strokeStyle = "rgba(255,255,255,0.95)";
        ctx.lineWidth = 2;
        ctx.strokeRect(a.x + 1, a.y + 1, b.x - a.x - 2, b.y - a.y - 2);
      }
    }

    // Selected tile (for the assistant): thin accent rectangle, works with or without a heatmap.
    if (selected && dzi) {
      const r = dziTileRect(dzi, selected);
      const a = toEl(r.x, r.y);
      const b = toEl(r.x + r.w, r.y + r.h);
      ctx.strokeStyle = "rgba(110, 168, 254, 0.95)";
      ctx.lineWidth = 2;
      ctx.strokeRect(a.x + 1, a.y + 1, b.x - a.x - 2, b.y - a.y - 2);
      ctx.strokeStyle = "rgba(17, 19, 23, 0.8)";
      ctx.lineWidth = 1;
      ctx.strokeRect(a.x + 2.5, a.y + 2.5, b.x - a.x - 5, b.y - a.y - 5);
    }

    if (groundTruth && options.showGroundTruth) {
      ctx.lineWidth = 1.5;
      for (const c of groundTruth.cells) {
        const a = toEl(c.x, c.y);
        const b = toEl(c.x + c.w, c.y + c.h);
        if (b.x < 0 || b.y < 0 || a.x > cssW || a.y > cssH) continue;
        ctx.strokeStyle = c.tumor ? GT_TUMOR : GT_OTHER;
        ctx.lineWidth = c.tumor ? 2 : 1;
        const inset = c.tumor ? 2 : 0.5;
        ctx.strokeRect(a.x + inset, a.y + inset, b.x - a.x - 2 * inset, b.y - a.y - 2 * inset);
      }
    }
  }, [viewer, heatmap, groundTruth, options, dzi, selected]);

  // Redraw on every viewport change.
  useEffect(() => {
    if (!viewer) return;
    const handler = () => draw();
    viewer.addHandler("update-viewport", handler);
    viewer.addHandler("resize", handler);
    viewer.addHandler("open", handler);
    viewer.addHandler("animation-finish", handler);
    draw();
    return () => {
      viewer.removeHandler("update-viewport", handler);
      viewer.removeHandler("resize", handler);
      viewer.removeHandler("open", handler);
      viewer.removeHandler("animation-finish", handler);
    };
  }, [viewer, draw]);

  useEffect(() => {
    draw();
  }, [draw, hover]);

  // Hover + click -> tooltip + tile selection. Read from the viewer element so OSD gestures keep working.
  useEffect(() => {
    if (!viewer) {
      setHover(null);
      return;
    }
    const el = viewer.element;
    const toImage = (px: number, py: number) => {
      if (viewer.world.getItemCount() === 0) return null;
      const img = viewer.viewport.viewerElementToImageCoordinates(new OpenSeadragon.Point(px, py));
      const w = heatmap?.width ?? dzi?.width ?? 0;
      const h = heatmap?.height ?? dzi?.height ?? 0;
      if (img.x < 0 || img.y < 0 || img.x >= w || img.y >= h) return null;
      return img;
    };
    const findTile = (clientX: number, clientY: number): { tile: HeatmapTile | null; px: number; py: number } => {
      const rect = el.getBoundingClientRect();
      const px = clientX - rect.left;
      const py = clientY - rect.top;
      if (!heatmap) return { tile: null, px, py };
      const img = toImage(px, py);
      if (!img) return { tile: null, px, py };
      const size = heatmap.tile_size * heatmap.scale;
      const tx = Math.floor(img.x / size);
      const ty = Math.floor(img.y / size);
      const tile = heatmap.tiles.find((t) => t.x === tx && t.y === ty) ?? null;
      return { tile, px, py };
    };
    const onMove = (e: PointerEvent) => {
      if (hoverRef.current?.pinned) return;
      if (e.buttons !== 0) return; // dragging
      const { tile, px, py } = findTile(e.clientX, e.clientY);
      if (!tile || !options.showHeatmap) {
        if (hoverRef.current) setHover(null);
        return;
      }
      setHover({ tile, px, py, pinned: false });
    };
    const onLeave = () => {
      if (!hoverRef.current?.pinned) setHover(null);
    };
    const onClick = (ev: OpenSeadragon.CanvasClickEvent) => {
      if (!ev.quick) return;
      const pos = ev.position;
      const img = toImage(pos.x, pos.y);
      if (!img) {
        setHover(null);
        selectRef.current?.(null);
        return;
      }
      // Heatmap present: select at the heatmap's level so the thumbnail matches the painted cell.
      // Otherwise: full-resolution DZI tile, floor(img / 256).
      let next: TileSelection;
      if (heatmap) {
        const size = heatmap.tile_size * heatmap.scale;
        const x = Math.floor(img.x / size);
        const y = Math.floor(img.y / size);
        next = { level: heatmap.level, x, y, tile: heatmap.tiles.find((t) => t.x === x && t.y === y) ?? null };
      } else if (dzi) {
        const size = dzi.tileSize || DZI_TILE;
        next = { level: dzi.maxLevel, x: Math.floor(img.x / size), y: Math.floor(img.y / size), tile: null };
      } else {
        return;
      }
      const cur = selectedRef.current;
      const same = !!cur && cur.level === next.level && cur.x === next.x && cur.y === next.y;
      if (same) {
        setHover(null);
        selectRef.current?.(null);
        return;
      }
      if (next.tile) setHover({ tile: next.tile, px: pos.x, py: pos.y, pinned: true });
      else setHover(null);
      selectRef.current?.(next);
    };
    el.addEventListener("pointermove", onMove);
    el.addEventListener("pointerleave", onLeave);
    viewer.addHandler("canvas-click", onClick);
    return () => {
      el.removeEventListener("pointermove", onMove);
      el.removeEventListener("pointerleave", onLeave);
      viewer.removeHandler("canvas-click", onClick);
    };
  }, [viewer, heatmap, dzi, options.showHeatmap]);

  // Selection cleared from outside (slide change) -> drop the pinned tooltip too.
  useEffect(() => {
    if (!selected && hoverRef.current?.pinned) setHover(null);
  }, [selected]);

  const tip = hover && options.showHeatmap ? hover : null;
  const style = tip ? DECISION_STYLE[tip.tile.decision] : null;

  return (
    <>
      <canvas ref={canvasRef} className="heatmap-canvas" />
      {tip && style && (
        <div
          className={`tile-tooltip ${tip.pinned ? "pinned" : ""}`}
          style={{ left: Math.min(tip.px + 14, Math.max(0, (viewer?.element.clientWidth ?? 400) - 240)), top: tip.py + 14 }}
        >
          <div className="tt-head">
            <span className="tt-dot" style={{ background: `rgb(${style.rgb})` }} />
            <strong>{style.label}</strong>
            <span className="tt-coord">
              tile {tip.tile.x},{tip.tile.y}
            </span>
          </div>
          <div className="tt-row">
            <span>p_tumor</span>
            <strong>{tip.tile.p_tumor.toFixed(3)}</strong>
          </div>
          <div className="tt-row">
            <span>routed_to</span>
            <strong>{tip.tile.routed_to}</strong>
          </div>
          {tip.tile.reason && <div className="tt-reason">{tip.tile.reason}</div>}
          {tip.pinned && <div className="tt-hint">selected for the assistant · click again to unpin</div>}
        </div>
      )}
    </>
  );
};

export default HeatmapOverlay;
