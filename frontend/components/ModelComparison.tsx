"use client";

import React from "react";
import { CheckCircle2, AlertTriangle, Cpu, Brain, GitCompare } from "lucide-react";
import { EMOTION_CONFIG } from "./EmotionCard";

interface ModelPrediction {
  model: string;
  type: string;
  predicted_emotion: string;
  confidence: number;
  probabilities: Record<string, number>;
}

interface ModelComparisonProps {
  modelsData: Record<string, ModelPrediction>;
  consensusEmotion?: string;
}

export default function ModelComparison({
  modelsData,
  consensusEmotion
}: ModelComparisonProps) {
  const modelKeys = ["CNN", "MLP", "SVM"];
  const availableModels = modelKeys.filter((k) => modelsData[k]);

  // Check if all models agreed
  const emotionsPredicted = availableModels.map((k) => modelsData[k].predicted_emotion.toLowerCase());
  const allAgree = emotionsPredicted.length > 1 && emotionsPredicted.every((e) => e === emotionsPredicted[0]);

  return (
    <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2.5">
          <div className="w-8 h-8 rounded-lg bg-indigo-50 border border-indigo-200 flex items-center justify-center text-indigo-600">
            <GitCompare className="w-4 h-4" />
          </div>
          <div>
            <h3 className="font-bold text-slate-900 text-sm">Cross-Model Prediction Comparison</h3>
            <p className="text-xs text-slate-500">Side-by-side evaluation across Deep Learning & Traditional ML</p>
          </div>
        </div>

        {/* Agreement Status Badge */}
        {allAgree ? (
          <span className="flex items-center space-x-1 text-xs font-semibold px-2.5 py-1 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>Full Model Consensus</span>
          </span>
        ) : (
          <span className="flex items-center space-x-1 text-xs font-semibold px-2.5 py-1 rounded-full bg-amber-50 text-amber-700 border border-amber-200">
            <AlertTriangle className="w-3.5 h-3.5" />
            <span>Divergent Predictions</span>
          </span>
        )}
      </div>

      {/* 3 Model Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {availableModels.map((key) => {
          const item = modelsData[key];
          const emoConfig = EMOTION_CONFIG[item.predicted_emotion.toLowerCase()] || {
            label: item.predicted_emotion,
            emoji: "🎙️",
            color: "#6366f1"
          };
          const isBestModel = key === "CNN";

          return (
            <div
              key={key}
              className={`p-4 rounded-xl border transition-all ${
                isBestModel
                  ? "border-indigo-300 bg-indigo-50/40 shadow-sm"
                  : "border-slate-200 bg-slate-50/50"
              }`}
            >
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-bold text-slate-700 uppercase tracking-wider flex items-center space-x-1">
                  <span>{item.model}</span>
                  {isBestModel && (
                    <span className="text-[10px] bg-indigo-600 text-white font-bold px-1.5 py-0.2 rounded">
                      Best Acc
                    </span>
                  )}
                </span>
                <span className="text-[11px] text-slate-500">{item.type}</span>
              </div>

              <div className="flex items-center space-x-3 my-2">
                <span className="text-3xl">{emoConfig.emoji}</span>
                <div>
                  <h4 className="text-xl font-bold text-slate-900 capitalize">
                    {emoConfig.label}
                  </h4>
                  <p className="text-xs text-slate-500 font-medium">
                    Confidence: <span className="font-bold text-slate-800">{item.confidence}%</span>
                  </p>
                </div>
              </div>

              {/* Mini probability preview */}
              <div className="mt-3 pt-2.5 border-t border-slate-200/60 space-y-1 text-[11px]">
                {Object.entries(item.probabilities || {})
                  .slice(0, 3)
                  .map(([emo, prob]) => (
                    <div key={emo} className="flex items-center justify-between text-slate-600">
                      <span className="capitalize">{emo}</span>
                      <span className="font-semibold text-slate-800">{prob}%</span>
                    </div>
                  ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

