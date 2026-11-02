import React from "react";
import { HeatmapSummary } from "../api/mlApi";
import { AskEntry } from "./AskPanel";

interface Props {
  summary: HeatmapSummary | null;
  history: AskEntry[];
}

const median = (xs: number[]): number => {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

/** Two-stat "AI saves pathologists hours" banner at the top of the sidebar. */
const ImpactStrip: React.FC<Props> = ({ summary, history }) => {
  const cleared = summary ? `${(summary.frac_auto_cleared * 100).toFixed(0)}%` : "—";
  const mins = summary ? `~${summary.pathologist_minutes_saved.toFixed(0)}` : "—";
  const n = history.length;
  const med = n ? `${Math.round(median(history.map((h) => h.latency_ms)))}` : "—";
  return (
    <div className="impact-strip" aria-label="AI impact">
      <div className="impact-item">
        <span className="impact-k">Triage</span>
        <span className="impact-v">
          <strong>{cleared}</strong> of tiles auto-cleared, <strong>{mins}</strong> min saved per slide
        </span>
      </div>
      <div className="impact-item">
        <span className="impact-k">Assistant</span>
        <span className="impact-v">
          <strong>{n}</strong> {n === 1 ? "question" : "questions"} answered, median <strong>{med}</strong> ms
        </span>
      </div>
    </div>
  );
};

export default ImpactStrip;
