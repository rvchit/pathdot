import { api } from "./client";

/**
 * S3-mode multipart upload (STORAGE_MODE=s3 on the backend). Local mode uses
 * `uploadLocal` from slidesApi.ts instead.
 */
export const initiateUpload = async (fileName: string) => {
  const response = await api.post("/api/start-upload", { fileName, fileType: "application/octet-stream" });
  return response.data as { uploadId: string };
};

export const uploadChunk = async (chunk: Blob, fileName: string, uploadId: string, partNumber: number) => {
  const formData = new FormData();
  formData.append("fileChunk", chunk);
  formData.append("fileName", fileName);
  formData.append("partNumber", partNumber.toString());
  formData.append("uploadId", uploadId);
  const response = await api.post("/api/upload-part", formData, { timeout: 0 });
  return response.data as { ETag: string };
};

export const completeUpload = async (fileName: string, uploadId: string, parts: Array<{ ETag: string; PartNumber: number }>) => {
  const response = await api.post("/api/complete-upload", { fileName, uploadId, parts });
  return response.data as { fileUrl: string };
};

export async function uploadToS3(file: File, onProgress?: (frac: number) => void): Promise<string> {
  const { uploadId } = await initiateUpload(file.name);
  const chunkSize = 5 * 1024 * 1024;
  const totalChunks = Math.ceil(file.size / chunkSize);
  const parts: Array<{ ETag: string; PartNumber: number }> = [];
  for (let i = 0; i < totalChunks; i++) {
    const chunk = file.slice(i * chunkSize, Math.min(file.size, (i + 1) * chunkSize));
    const { ETag } = await uploadChunk(chunk, file.name, uploadId, i + 1);
    parts.push({ ETag, PartNumber: i + 1 });
    onProgress?.((i + 1) / totalChunks);
  }
  const { fileUrl } = await completeUpload(file.name, uploadId, parts);
  return fileUrl;
}
