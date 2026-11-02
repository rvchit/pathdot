import { useEffect, useState } from "react";
import { Gallery, GalleryAskItem, GalleryScanItem, getGallery } from "./mlApi";

/** A demo gallery item; its `id` doubles as the DZI image id in the slide viewer (tiles/<id>/). */
export type GalleryItem = GalleryScanItem | GalleryAskItem;

export interface GalleryIndex {
  gallery: Gallery | null;
  byId: Record<string, GalleryItem>;
}

const EMPTY: GalleryIndex = { gallery: null, byId: {} };

// Module-scope cache: the gallery is static for the life of the ML service, so fetch it once.
let cached: GalleryIndex = EMPTY;
let inflight: Promise<GalleryIndex> | null = null;
const listeners = new Set<(g: GalleryIndex) => void>();

function index(gallery: Gallery): GalleryIndex {
  const byId: Record<string, GalleryItem> = {};
  for (const it of gallery.scan ?? []) byId[it.id] = it;
  for (const it of gallery.ask ?? []) byId[it.id] = it;
  return { gallery, byId };
}

/** Fetch the gallery once (shared across components); resolves to the cached index on later calls. */
export function loadGallery(): Promise<GalleryIndex> {
  if (cached.gallery) return Promise.resolve(cached);
  if (!inflight) {
    inflight = getGallery()
      .then((g) => {
        cached = index(g);
        listeners.forEach((l) => l(cached));
        return cached;
      })
      .catch(() => {
        inflight = null; // allow a retry on the next mount (e.g. ML service started late)
        return EMPTY;
      });
  }
  return inflight;
}

/** The gallery item whose id matches the open slide, or null for ordinary slides. */
export function galleryItemFor(imageId: string | undefined | null): GalleryItem | null {
  if (!imageId) return null;
  return cached.byId[imageId] ?? null;
}

export const isScanItem = (it: GalleryItem | null): it is GalleryScanItem => !!it && it.section === "scan";
export const isAskItem = (it: GalleryItem | null): it is GalleryAskItem => !!it && it.section === "ask";

/** Short human label for a gallery slide in the slide list ("Tile 1 · Camelyon17", "Image 1 · PathVQA"). */
export function galleryLabel(it: GalleryItem): string {
  const n = it.id.replace(/^\D+/, "") || it.id;
  if (it.section === "scan") return `Tile ${n} · ${it.source === "camelyon17" ? "Camelyon17" : it.source === "pcam" ? "PCam" : it.source}`;
  return `Image ${n} · PathVQA`;
}

/** React hook over the module cache: {gallery, byId}; re-renders once the gallery arrives. */
export function useGallery(): GalleryIndex {
  const [state, setState] = useState<GalleryIndex>(cached);
  useEffect(() => {
    let alive = true;
    listeners.add(setState);
    loadGallery().then((g) => alive && setState(g));
    return () => {
      alive = false;
      listeners.delete(setState);
    };
  }, []);
  return state;
}
