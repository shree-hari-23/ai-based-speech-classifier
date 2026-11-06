"use client";

import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell
} from "recharts";
import { EMOTION_CONFIG } from "./EmotionCard";

interface ConfidenceChartProps {
  probabilities: Record<string, number>;
  predictedEmotion?: string;
}

export default function ConfidenceChart({
  probabilities,
  predictedEmotion
}: ConfidenceChartProps) {
  // Convert dict to chart array sorted descending
  const data = Object.entries(probabilities || {})
    .map(([emotion, prob]) => ({
      emotion: emotion.charAt(0).toUpperCase() + emotion.slice(1),
      rawEmotion: emotion.toLowerCase(),
      probability: typeof prob === "number" ? Number(prob.toFixed(1)) : 0,
      color: EMOTION_CONFIG[emotion.toLowerCase()]?.color || "#6366f1"
    }))
    .sort((a, b) => b.probability - a.probability);

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const item = payload[0].payload;
      return (
        <div className="bg-slate-900 text-white text-xs px-3 py-2 rounded-xl shadow-lg border border-slate-700">
          <p className="font-bold flex items-center space-x-1.5">
            <span>{EMOTION_CONFIG[item.rawEmotion]?.emoji}</span>
            <span>{item.emotion}</span>
          </p>
          <p className="text-slate-300 mt-1">
            Probability: <span className="font-bold text-white">{item.probability}%</span>
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="font-bold text-slate-900 text-sm">Emotion Probability Distribution</h3>
          <p className="text-xs text-slate-500">Softmax / calibrated confidence across all 8 acoustic classes</p>
        </div>
      </div>

      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
          >
            <XAxis
              type="number"
              domain={[0, 100]}
              tick={{ fontSize: 11, fill: "#64748b" }}
              unit="%"
            />
            <YAxis
              dataKey="emotion"
              type="category"
              tick={{ fontSize: 12, fill: "#334155", fontWeight: 500 }}
              width={75}
            />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="probability" radius={[0, 8, 8, 0]} barSize={16}>
              {data.map((entry, index) => {
                const isSelected = predictedEmotion && entry.rawEmotion === predictedEmotion.toLowerCase();
                return (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.color}
                    opacity={isSelected ? 1.0 : 0.8}
                    stroke={isSelected ? "#0f172a" : "none"}
                    strokeWidth={isSelected ? 1.5 : 0}
                  />
                );
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

