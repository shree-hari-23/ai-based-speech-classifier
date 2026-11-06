"use client";

import React from "react";
import { Volume2 } from "lucide-react";

interface WaveformProps {
  points?: number[];
  isPlaying?: boolean;
  duration?: number;
  emotionColor?: string;
}

export default function Waveform({
  points,
  isPlaying = false,
  duration,
  emotionColor = "#6366f1"
}: WaveformProps) {
  // Generate fallback points if none provided
  const displayPoints = points && points.length > 0
    ? points
    : [0.15, 0.25, 0.45, 0.7, 0.9, 0.6, 0.4, 0.2, 0.35, 0.55, 0.8, 0.95, 0.7, 0.4, 0.2, 0.1];

  return (
    <div className="bg-slate-900 rounded-2xl p-4 text-white relative overflow-hidden shadow-inner">
      <div className="flex items-center justify-between mb-3 text-xs text-slate-400">
        <span className="flex items-center space-x-1.5 font-medium">
          <Volume2 className="w-3.5 h-3.5 text-indigo-400" />
          <span>Acoustic Waveform Analysis</span>
        </span>
        {duration && (
          <span className="font-mono text-slate-300 bg-slate-800 px-2 py-0.5 rounded">
            {duration.toFixed(2)}s
          </span>
        )}
      </div>

      {/* Waveform Bars */}
      <div className="h-24 flex items-center justify-between gap-[2px] px-1 relative">
        {displayPoints.map((val, idx) => {
          const heightPct = Math.max(8, Math.min(100, Math.round(val * 100)));
          return (
            <div
              key={idx}
              className="flex-1 rounded-full transition-all duration-300"
              style={{
                height: `${heightPct}%`,
                backgroundColor: isPlaying ? emotionColor : "#475569",
                opacity: isPlaying ? 0.9 : 0.6
              }}
            />
          );
        })}

        {/* Playhead glow animation when playing */}
        {isPlaying && (
          <div
            className="absolute top-0 bottom-0 w-1 bg-white shadow-[0_0_12px_#ffffff] animate-[pulse_1s_infinite]"
            style={{ left: "50%" }}
          />
        )}
      </div>

      <div className="mt-2 flex items-center justify-between text-[11px] text-slate-400 font-mono">
        <span>0.0s</span>
        <span>Time Frames (Librosa STFT: 130 frames)</span>
        <span>{duration ? `${duration.toFixed(1)}s` : "3.0s"}</span>
      </div>
    </div>
  );
}

