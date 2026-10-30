import axios from "axios";

/** Base URL of the cello Express backend. Everything (tiles, comments, /ml proxy) goes through it. */
export const BACKEND_URL = (import.meta.env.VITE_BACKEND_URL ?? "http://localhost:4001").replace(/\/+$/, "");

export const api = axios.create({
  baseURL: BACKEND_URL,
  timeout: 30_000,
});

/** Human-readable message for any axios/network error. */
export function errorMessage(err: unknown): string {
  if (axios.isAxiosError(err)) {
    const data = err.response?.data as { message?: string; error?: string; detail?: unknown } | undefined;
    if (data?.message) return data.message;
    if (typeof data?.detail === "string") return data.detail;
    if (data?.error) return String(data.error);
    if (err.code === "ERR_NETWORK") return `Cannot reach the backend at ${BACKEND_URL}. Is \`npm run dev\` running in cello/backend?`;
    return err.message;
  }
  return err instanceof Error ? err.message : String(err);
}
