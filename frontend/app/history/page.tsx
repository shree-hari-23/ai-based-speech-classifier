"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  History,
  Search,
  Filter,
  Trash2,
  Calendar,
  Layers,
  ArrowUpDown,
  RefreshCw,
  AlertCircle,
  Play,
  Volume2
} from "lucide-react";
import { getHistory, deleteHistoryItem } from "@/lib/api";
import { EMOTION_CONFIG } from "@/components/EmotionCard";

export default function HistoryPage() {
  const [historyItems, setHistoryItems] = useState<any[]>([]);
  const [totalCount, setTotalCount] = useState<number>(0);
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedEmotion, setSelectedEmotion] = useState("all");
  const [selectedModel, setSelectedModel] = useState("all");
  const [loading, setLoading] = useState(true);
  const [deletingId, setDeletingId] = useState<number | null>(null);

  const fetchHistoryData = async () => {
    setLoading(true);
    try {
      const data = await getHistory({
        search: searchTerm || undefined,
        emotion: selectedEmotion !== "all" ? selectedEmotion : undefined,
        model: selectedModel !== "all" ? selectedModel : undefined,
        limit: 100
      });
      setHistoryItems(data.items || []);
      setTotalCount(data.total || 0);
    } catch (err) {
      console.error("Error fetching history:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const timer = setTimeout(() => {
      fetchHistoryData();
    }, 300);
    return () => clearTimeout(timer);
  }, [searchTerm, selectedEmotion, selectedModel]);

  const handleDelete = async (id: number) => {
    if (!confirm(`Delete prediction record #${id}?`)) return;
    setDeletingId(id);
    try {
      await deleteHistoryItem(id);
      setHistoryItems((prev) => prev.filter((item) => item.id !== id));
      setTotalCount((prev) => Math.max(0, prev - 1));
    } catch (err) {
      alert("Failed to delete history record.");
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl lg:text-3xl font-extrabold text-slate-900 tracking-tight">
            Prediction History
          </h1>
          <p className="text-slate-500 text-sm mt-1">
            Persisted prediction logs in SQLite with confidence scores, timestamps, and model provenance.
          </p>
        </div>

        <button
          type="button"
          onClick={fetchHistoryData}
          className="inline-flex items-center space-x-1.5 px-4 py-2 rounded-xl text-xs font-semibold bg-white border border-slate-200 text-slate-700 shadow-sm hover:bg-slate-50 transition-colors self-start"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Filter & Search Bar */}
      <div className="bg-white p-4 rounded-2xl border border-slate-200 shadow-sm flex flex-col md:flex-row gap-3">
        {/* Search Input */}
        <div className="relative flex-1">
          <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search by filename or emotion..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-9 pr-4 py-2 text-sm rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500"
          />
        </div>

        {/* Emotion Filter */}
        <div className="flex items-center space-x-2">
          <Filter className="w-4 h-4 text-slate-400 shrink-0" />
          <select
            value={selectedEmotion}
            onChange={(e) => setSelectedEmotion(e.target.value)}
            className="text-xs font-semibold text-slate-700 bg-white border border-slate-200 rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500/20"
          >
            <option value="all">All Emotions</option>
            <option value="angry">Angry</option>
            <option value="calm">Calm</option>
            <option value="disgust">Disgust</option>
            <option value="fearful">Fearful</option>
            <option value="happy">Happy</option>
            <option value="neutral">Neutral</option>
            <option value="sad">Sad</option>
            <option value="surprised">Surprised</option>
          </select>
        </div>

        {/* Model Filter */}
        <div className="flex items-center space-x-2">
          <Layers className="w-4 h-4 text-slate-400 shrink-0" />
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="text-xs font-semibold text-slate-700 bg-white border border-slate-200 rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500/20"
          >
            <option value="all">All Models</option>
            <option value="CNN">1D CNN</option>
            <option value="MLP">MLP</option>
            <option value="SVM">SVM</option>
            <option value="ALL_MODELS">Multi-Model</option>
          </select>
        </div>
      </div>

      {/* History Table */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
        {loading ? (
          <div className="text-center py-16 space-y-2">
            <div className="w-8 h-8 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin mx-auto" />
            <p className="text-xs text-slate-500 font-medium">Fetching prediction logs...</p>
          </div>
        ) : historyItems.length === 0 ? (
          <div className="text-center py-16 space-y-2 text-slate-500 text-sm">
            <History className="w-10 h-10 text-slate-300 mx-auto" />
            <p className="font-semibold text-slate-700">No prediction records found</p>
            <p className="text-xs text-slate-400">
              Try recording voice or uploading an audio file on the{" "}
              <Link href="/predict" className="text-indigo-600 font-semibold underline">
                Predict page
              </Link>
              .
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm text-slate-700">
              <thead className="bg-slate-50 text-xs uppercase text-slate-400 font-semibold border-b border-slate-200">
                <tr>
                  <th className="py-3.5 px-4">#ID</th>
                  <th className="py-3.5 px-4">Filename</th>
                  <th className="py-3.5 px-4">Predicted Emotion</th>
                  <th className="py-3.5 px-4">Confidence</th>
                  <th className="py-3.5 px-4">Model Used</th>
                  <th className="py-3.5 px-4">Duration</th>
                  <th className="py-3.5 px-4">Date & Time</th>
                  <th className="py-3.5 px-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {historyItems.map((item) => {
                  const emo = EMOTION_CONFIG[item.predicted_emotion.toLowerCase()] || {
                    label: item.predicted_emotion,
                    emoji: "🎙️",
                    color: "#6366f1"
                  };

                  return (
                    <tr key={item.id} className="hover:bg-slate-50/70 transition-colors">
                      <td className="py-3.5 px-4 text-xs font-mono text-slate-400">
                        #{item.id}
                      </td>
                      <td className="py-3.5 px-4 font-semibold text-slate-900 truncate max-w-xs">
                        {item.filename}
                      </td>
                      <td className="py-3.5 px-4">
                        <span className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-xs font-bold bg-slate-100 text-slate-800">
                          <span>{emo.emoji}</span>
                          <span className="capitalize">{emo.label}</span>
                        </span>
                      </td>
                      <td className="py-3.5 px-4 font-bold" style={{ color: emo.color }}>
                        {item.confidence}%
                      </td>
                      <td className="py-3.5 px-4">
                        <span className="text-xs px-2 py-0.5 rounded bg-slate-100 font-medium text-slate-700">
                          {item.model_used}
                        </span>
                      </td>
                      <td className="py-3.5 px-4 text-xs font-mono text-slate-500">
                        {item.duration ? `${item.duration}s` : "3.0s"}
                      </td>
                      <td className="py-3.5 px-4 text-xs text-slate-500">
                        {new Date(item.created_at).toLocaleString()}
                      </td>
                      <td className="py-3.5 px-4 text-right">
                        <button
                          type="button"
                          onClick={() => handleDelete(item.id)}
                          disabled={deletingId === item.id}
                          className="p-1.5 text-slate-400 hover:text-rose-600 hover:bg-rose-50 rounded-lg transition-colors"
                          title="Delete history entry"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {/* Footer count */}
        <div className="px-4 py-3 bg-slate-50 border-t border-slate-200 text-xs text-slate-500 flex items-center justify-between">
          <span>Total Records: <strong className="text-slate-800">{totalCount}</strong></span>
          <span>Database: SQLite (speech_emotion.db)</span>
        </div>
      </div>
    </div>
  );
}

