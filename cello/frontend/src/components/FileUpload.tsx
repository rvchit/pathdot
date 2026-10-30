import React, { useEffect, useRef, useState } from "react";
import { errorMessage } from "../api/client";
import { listImages, uploadLocal, uploadMode } from "../api/slidesApi";
import { uploadToS3 } from "../api/fileuploadApi";

interface Props {
  selectedImageId?: string;
  onImageSelect: (imageId: string) => void;
}

/** Slide list (GET /osd/images) + upload (local: /api/upload-local, s3: multipart). */
const FileUpload: React.FC<Props> = ({ selectedImageId, onImageSelect }) => {
  const [images, setImages] = useState<string[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [mode, setMode] = useState<"local" | "s3">("local");
  const [uploading, setUploading] = useState(false);
  const [uploadPct, setUploadPct] = useState(0);
  const [hint, setHint] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const refresh = async () => {
    try {
      const list = await listImages();
      setImages(list);
      setLoadError(null);
    } catch (err) {
      setLoadError(errorMessage(err));
    }
  };

  useEffect(() => {
    refresh();
    uploadMode().then(setMode).catch(() => setMode("local"));
    const t = setInterval(refresh, 10_000); // pick up freshly tiled slides
    return () => clearInterval(t);
  }, []);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    setUploading(true);
    setUploadPct(0);
    setHint(null);
    try {
      if (mode === "s3") {
        await uploadToS3(file, setUploadPct);
        setHint(`Uploaded ${file.name} to S3. Run the tiling Lambda, then it will appear here.`);
      } else {
        const res = await uploadLocal(file, setUploadPct);
        setHint(`${res.message}\n\n${res.nextStep}`);
      }
      refresh();
    } catch (err) {
      setHint(`Upload failed: ${errorMessage(err)}`);
    } finally {
      setUploading(false);
    }
  };

  return (
    <section className="panel">
      <header className="panel-head">
        <h2>Slides</h2>
        <button className="btn btn-ghost btn-sm" onClick={refresh} title="GET /osd/images">
          refresh
        </button>
      </header>

      {loadError && <div className="notice error">{loadError}</div>}
      {!loadError && images.length === 0 && (
        <div className="notice">
          No tiled slides yet. Run <code>make tile</code> to build the demo mosaic, or upload an image below.
        </div>
      )}
      <ul className="slide-list">
        {images.map((id) => (
          <li key={id}>
            <button className={`slide-item ${selectedImageId === id ? "selected" : ""}`} onClick={() => onImageSelect(id)}>
              {id}
            </button>
          </li>
        ))}
      </ul>

      <input type="file" ref={fileInputRef} style={{ display: "none" }} onChange={handleFileChange} accept=".png,.jpg,.jpeg,.tif,.tiff,.svs,.ndpi" />
      <button className="btn" onClick={() => fileInputRef.current?.click()} disabled={uploading}>
        {uploading ? `Uploading… ${Math.round(uploadPct * 100)}%` : mode === "s3" ? "Upload to S3" : "Upload slide (local)"}
      </button>
      {hint && <pre className="hint">{hint}</pre>}
    </section>
  );
};

export default FileUpload;
