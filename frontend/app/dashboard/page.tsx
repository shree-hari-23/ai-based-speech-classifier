"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  Mic,
  Activity,
  Layers,
  Award,
  Database,
  ArrowRight,
  TrendingUp,
  Clock,
  Sparkles,
  ShieldCheck,
  LogIn
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  PieChart,
  Pie
} from "recharts";
import { getAnalytics, getHistory, getModels } from "@/lib/api";
import { EMOTION_CONFIG } from "@/components/EmotionCard";
import { useAuth } from "@/lib/auth-context";

export default function DashboardPage() {
  const { user } = useAuth();
  const [analytics, setAnalytics] = useState<any>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [totalPredictions, setTotalPredictions] = useState<number>(0);
  const [models, setModels] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadDashboardData() {
      try {
        const [analyticsData, historyData, modelsData] = await Promise.all([
          getAnalytics().catch(() => null),
          getHistory({ limit: 5 }).catch(() => ({ total: 0, items: [] })),
          getModels().catch(() => ({ models: [] }))
        ]);

        if (analyticsData) setAnalytics(analyticsData);
        if (historyData) {
          setHistory(historyData.items || []);
          setTotalPredictions(historyData.total || 0);
        }
        if (modelsData) setModels(modelsData.models || []);
      } finally {
        setLoading(false);
      }
    }

    loadDashboardData();
  }, []);

  // Emotion distribution data for Pie Chart
  const emotionDistribution = [
    { name: "Angry", count: 592, color: "#ef4444" },
    { name: "Disgust", count: 592, color: "#10b981" },
    { name: "Fearful", count: 592, color: "#a855f7" },
    { name: "Happy", count: 592, color: "#eab308" },
    { name: "Sad", count: 592, color: "#3b82f6" },
    { name: "Surprised", count: 592, color: "#f97316" },
    { name: "Neutral", count: 496, color: "#64748b" },
    { name: "Calm", count: 192, color: "#06b6d4" },
  ];

  // Model comparison bar data
  const modelComparisonData = analytics?.comparison_table?.map((item: any) => ({
    name: item.model,
    accuracy: item.test_accuracy,
    f1: item.f1_score,
    type: item.type,
    color: item.model === "CNN" ? "#4f46e5" : item.model === "MLP" ? "#10b981" : "#3b82f6"
  })) || [
    { name: "CNN", accuracy: 82.63, f1: 82.3, color: "#4f46e5" },
    { name: "MLP", accuracy: 79.75, f1: 79.77, color: "#10b981" },
    { name: "SVM", accuracy: 79.12, f1: 79.68, color: "#3b82f6" },
  ];

  return (
    <div className="space-y-8">
      {/* Welcome Banner */}
      <div className="relative overflow-hidden bg-gradient-to-r from-indigo-900 via-indigo-800 to-slate-900 rounded-3xl p-8 text-white shadow-xl">
        <div className="relative z-10 max-w-3xl space-y-3">
          <div className="flex flex-wrap items-center gap-2">
            <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-indigo-500/20 text-indigo-200 text-xs font-semibold border border-indigo-400/30">
              <Sparkles className="w-3.5 h-3.5 text-indigo-300" />
              <span>Acoustic Speech Emotion Recognition</span>
            </div>
            {user ? (
              <span className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full bg-emerald-500/20 text-emerald-300 text-xs font-semibold border border-emerald-400/30">
                <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                <span>Logged In as {user.full_name}</span>
              </span>
            ) : (
              <Link
                href="/login"
                className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full bg-white/10 hover:bg-white/20 text-indigo-200 text-xs font-medium border border-white/20 transition-colors"
              >
                <LogIn className="w-3.5 h-3.5" />
                <span>Sign in for Email Alerts</span>
              </Link>
            )}
          </div>

          <h1 className="text-3xl lg:text-4xl font-extrabold tracking-tight">
            {user ? `Welcome back, ${user.full_name}` : "Intelligent Vocal Emotion Classifier"}
          </h1>
          <p className="text-slate-300 text-sm leading-relaxed">
            Analyze acoustic prosody, pitch contours, and spectral dynamics from human speech.
            Powered by trained 1D CNN, MLP, and SVM models evaluated on 4,240 authentic speech recordings.
          </p>
          <div className="pt-2 flex flex-wrap gap-3">
            <Link
              href="/predict"
              className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-indigo-500 text-white font-semibold text-sm hover:bg-indigo-400 transition-all shadow-md shadow-indigo-900/50"
            >
              <Mic className="w-4 h-4" />
              <span>Predict Speech Emotion</span>
              <ArrowRight className="w-4 h-4" />
            </Link>
            <Link
              href="/analytics"
              className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-white/10 text-white font-semibold text-sm hover:bg-white/20 transition-all border border-white/20"
            >
              <span>Explore Model Analytics</span>
            </Link>
          </div>
        </div>
      </div>

      {/* 4 Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        {/* Stat 1: Total Recordings */}
        <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm flex items-center space-x-4">
          <div className="w-12 h-12 rounded-2xl bg-blue-50 text-blue-600 flex items-center justify-center">
            <Database className="w-6 h-6" />
          </div>
          <div>
            <span className="text-xs font-medium text-slate-500">Recordings Processed</span>
            <h3 className="text-2xl font-bold text-slate-900">4,240</h3>
            <span className="text-[11px] text-emerald-600 font-semibold">RAVDESS (1,440) + TESS (2,800)</span>
          </div>
        </div>

        {/* Stat 2: Best Model */}
        <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm flex items-center space-x-4">
          <div className="w-12 h-12 rounded-2xl bg-indigo-50 text-indigo-600 flex items-center justify-center">
            <Award className="w-6 h-6" />
          </div>
          <div>
            <span className="text-xs font-medium text-slate-500">Best Model</span>
            <h3 className="text-2xl font-bold text-indigo-600">
              {analytics?.best_model || "1D CNN"}
            </h3>
            <span className="text-[11px] text-slate-500 font-medium">
              {analytics?.best_accuracy || "82.63"}% Test Accuracy
            </span>
          </div>
        </div>

        {/* Stat 3: Available Models */}
        <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm flex items-center space-x-4">
          <div className="w-12 h-12 rounded-2xl bg-purple-50 text-purple-600 flex items-center justify-center">
            <Layers className="w-6 h-6" />
          </div>
          <div>
            <span className="text-xs font-medium text-slate-500">ML & DL Models</span>
            <h3 className="text-2xl font-bold text-slate-900">3 Models</h3>
            <span className="text-[11px] text-slate-500 font-medium">1D CNN, MLP, SVM</span>
          </div>
        </div>

        {/* Stat 4: Total Predictions */}
        <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm flex items-center space-x-4">
          <div className="w-12 h-12 rounded-2xl bg-emerald-50 text-emerald-600 flex items-center justify-center">
            <TrendingUp className="w-6 h-6" />
          </div>
          <div>
            <span className="text-xs font-medium text-slate-500">Total Predictions</span>
            <h3 className="text-2xl font-bold text-slate-900">{totalPredictions}</h3>
            <span className="text-[11px] text-emerald-600 font-semibold">Persisted in SQLite</span>
          </div>
        </div>
      </div>

      {/* Two Analytics Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model Accuracy Comparison */}
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-bold text-slate-900 text-base">Model Accuracy Comparison</h3>
              <p className="text-xs text-slate-500">Empirical test accuracy across the held-out test split</p>
            </div>
            <Link href="/analytics" className="text-xs font-semibold text-indigo-600 hover:text-indigo-800 flex items-center space-x-1">
              <span>Full metrics</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={modelComparisonData} margin={{ top: 10, right: 20, left: -10, bottom: 5 }}>
                <XAxis dataKey="name" tick={{ fontSize: 12, fill: "#334155", fontWeight: 600 }} />
                <YAxis domain={[60, 100]} tick={{ fontSize: 11, fill: "#64748b" }} unit="%" />
                <Tooltip
                  formatter={(val: any) => [`${val}%`, "Test Accuracy"]}
                  contentStyle={{ borderRadius: "12px", border: "1px solid #e2e8f0", boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.1)" }}
                />
                <Bar dataKey="accuracy" radius={[8, 8, 0, 0]} barSize={40}>
                  {modelComparisonData.map((entry: any, index: number) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Emotion Class Distribution */}
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-bold text-slate-900 text-base">Dataset Emotion Distribution</h3>
              <p className="text-xs text-slate-500">Distribution across 8 supported emotion classes (4,240 samples)</p>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs">
            {emotionDistribution.map((item) => (
              <div key={item.name} className="flex items-center justify-between p-2 rounded-lg bg-slate-50 border border-slate-100">
                <div className="flex items-center space-x-2">
                  <span className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: item.color }} />
                  <span className="font-semibold text-slate-700">{item.name}</span>
                </div>
                <span className="font-mono text-slate-500">{item.count}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Recent Predictions Table */}
      <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-bold text-slate-900 text-base">Recent Predictions</h3>
            <p className="text-xs text-slate-500">Latest speech recordings processed by the inference engine</p>
          </div>
          <Link href="/history" className="text-xs font-semibold text-indigo-600 hover:text-indigo-800 flex items-center space-x-1">
            <span>View all history</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>

        {history.length === 0 ? (
          <div className="text-center py-8 text-slate-400 text-sm">
            No prediction history yet. Try uploading or recording speech on the{" "}
            <Link href="/predict" className="text-indigo-600 font-semibold underline">
              Prediction page
            </Link>
            .
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm text-slate-600">
              <thead className="bg-slate-50 text-xs uppercase text-slate-400 font-semibold border-b border-slate-100">
                <tr>
                  <th className="py-3 px-4">Filename</th>
                  <th className="py-3 px-4">Predicted Emotion</th>
                  <th className="py-3 px-4">Confidence</th>
                  <th className="py-3 px-4">Model Used</th>
                  <th className="py-3 px-4">Date & Time</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {history.map((item) => {
                  const emo = EMOTION_CONFIG[item.predicted_emotion.toLowerCase()] || {
                    label: item.predicted_emotion,
                    emoji: "🎙️",
                    color: "#6366f1"
                  };
                  return (
                    <tr key={item.id} className="hover:bg-slate-50/70 transition-colors">
                      <td className="py-3 px-4 font-medium text-slate-800 truncate max-w-xs">
                        {item.filename}
                      </td>
                      <td className="py-3 px-4">
                        <span className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-slate-100 text-slate-800">
                          <span>{emo.emoji}</span>
                          <span className="capitalize">{emo.label}</span>
                        </span>
                      </td>
                      <td className="py-3 px-4 font-bold" style={{ color: emo.color }}>
                        {item.confidence}%
                      </td>
                      <td className="py-3 px-4">
                        <span className="text-xs px-2 py-0.5 rounded bg-slate-100 font-medium text-slate-700">
                          {item.model_used}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-xs text-slate-400">
                        {new Date(item.created_at).toLocaleString()}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

