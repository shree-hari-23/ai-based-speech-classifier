"use client";

import React from "react";
import { Sparkles, Info, ShieldCheck } from "lucide-react";

export const EMOTION_CONFIG: Record<string, { label: string; emoji: string; color: string; bg: string; border: string }> = {
  happy: { label: "Happy", emoji: "😄", color: "#eab308", bg: "bg-amber-50", border: "border-amber-200" },
  sad: { label: "Sad", emoji: "😢", color: "#3b82f6", bg: "bg-blue-50", border: "border-blue-200" },
  angry: { label: "Angry", emoji: "😠", color: "#ef4444", bg: "bg-rose-50", border: "border-rose-200" },
  neutral: { label: "Neutral", emoji: "😐", color: "#64748b", bg: "bg-slate-50", border: "border-slate-200" },
  fearful: { label: "Fearful", emoji: "😨", color: "#a855f7", bg: "bg-purple-50", border: "border-purple-200" },
  disgust: { label: "Disgust", emoji: "🤢", color: "#10b981", bg: "bg-emerald-50", border: "border-emerald-200" },
  surprised: { label: "Surprised", emoji: "😲", color: "#f97316", bg: "bg-orange-50", border: "border-orange-200" },
  calm: { label: "Calm", emoji: "😌", color: "#06b6d4", bg: "bg-cyan-50", border: "border-cyan-200" },
};

interface EmotionCardProps {
  emotion: string;
  confidence: number;
  model: string;
  disclaimer?: string;
}

export default function EmotionCard({
  emotion,
  confidence,
  model,
  disclaimer
}: EmotionCardProps) {
  const emoKey = emotion.toLowerCase().trim();
  const config = EMOTION_CONFIG[emoKey] || {
    label: emotion,
    emoji: "🎙️",
    color: "#6366f1",
    bg: "bg-indigo-50",
    border: "border-indigo-200"
  };

  return (
    <div className={`p-6 rounded-2xl border ${config.border} ${config.bg} shadow-sm transition-all relative overflow-hidden`}>
      <div className="flex items-start justify-between">
        <div className="flex items-center space-x-4">
          <div
            className="w-16 h-16 rounded-2xl flex items-center justify-center text-4xl shadow-sm bg-white border border-slate-100"
          >
            {config.emoji}
          </div>

          <div>
            <div className="flex items-center space-x-2">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
                Predicted Emotion
              </span>
              <span className="text-xs px-2 py-0.5 rounded-full font-semibold bg-white text-slate-700 border border-slate-200">
                {model}
              </span>
            </div>
            <h2 className="text-3xl font-extrabold text-slate-900 capitalize tracking-tight mt-0.5">
              {config.label}
            </h2>
          </div>
        </div>

        {/* Confidence Percentage Gauge */}
        <div className="text-right">
          <span className="text-xs font-semibold text-slate-500 block mb-0.5">Confidence</span>
          <div className="flex items-baseline space-x-1">
            <span
              className="text-3xl font-black"
              style={{ color: config.color }}
            >
              {confidence.toFixed(1)}
            </span>
            <span className="text-sm font-bold text-slate-500">%</span>
          </div>
        </div>
      </div>

      {/* Confidence Bar */}
      <div className="mt-5">
        <div className="w-full bg-slate-200/80 rounded-full h-2.5 overflow-hidden">
          <div
            className="h-2.5 rounded-full transition-all duration-700 ease-out"
            style={{
              width: `${Math.min(100, Math.max(5, confidence))}%`,
              backgroundColor: config.color
            }}
          />
        </div>
      </div>

      {/* Disclaimer */}
      {disclaimer && (
        <div className="mt-4 pt-3 border-t border-slate-200/60 flex items-center space-x-2 text-[11px] text-slate-500">
          <Info className="w-3.5 h-3.5 text-slate-400 shrink-0" />
          <span>{disclaimer}</span>
        </div>
      )}
    </div>
  );
}

