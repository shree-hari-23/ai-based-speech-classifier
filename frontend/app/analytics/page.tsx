"use client";

import React, { useEffect, useState } from "react";
import Image from "next/image";
import {
  BarChart3,
  Award,
  Layers,
  TrendingUp,
  Cpu,
  Brain,
  CheckCircle,
  AlertCircle,
  FileText
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  Legend
} from "recharts";
import { getAnalytics } from "@/lib/api";

export default function AnalyticsPage() {
  const [analytics, setAnalytics] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [activeMatrixTab, setActiveMatrixTab] = useState<"comparison" | "cnn" | "svm" | "mlp">("comparison");

  useEffect(() => {
    getAnalytics()
      .then((data) => setAnalytics(data))
      .catch((err) => console.error("Error loading analytics:", err))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-3">
        <div className="w-10 h-10 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin" />
        <p className="text-sm font-semibold text-slate-500">Loading model evaluation metrics...</p>
      </div>
    );
  }

  const comparisonTable = analytics?.comparison_table || [
    { model: "SVM", type: "Traditional Machine Learning", train_accuracy: 99.45, test_accuracy: 79.12, precision: 81.54, recall: 79.12, f1_score: 79.68 },
    { model: "MLP", type: "Traditional Machine Learning", train_accuracy: 99.19, test_accuracy: 79.75, precision: 81.15, recall: 79.75, f1_score: 79.77 },
    { model: "CNN", type: "Deep Learning (1D Conv)", train_accuracy: 93.26, test_accuracy: 82.63, precision: 84.34, recall: 82.62, f1_score: 82.30 },
  ];

  // Multi-metric comparison data for grouped bar chart
  const groupedChartData = [
    { metric: "Test Accuracy", SVM: 79.12, MLP: 79.75, CNN: 82.63 },
    { metric: "Precision", SVM: 81.54, MLP: 81.15, CNN: 84.34 },
    { metric: "Recall", SVM: 79.12, MLP: 79.75, CNN: 82.62 },
    { metric: "F1-Score", SVM: 79.68, MLP: 79.77, CNN: 82.30 },
  ];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl lg:text-3xl font-extrabold text-slate-900 tracking-tight">
          Model Performance & Evaluation Analytics
        </h1>
        <p className="text-slate-500 text-sm mt-1">
          Comparative analysis between Deep Learning (1D CNN) and Traditional ML (SVM, MLP) on 800 held-out speaker-independent test recordings.
        </p>
      </div>

      {/* 3 Model Metric Highlight Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {/* CNN Card */}
        <div className="bg-white p-6 rounded-2xl border-2 border-indigo-500 shadow-md relative overflow-hidden space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-indigo-700 bg-indigo-50 px-2.5 py-1 rounded-full border border-indigo-200">
              Deep Learning
            </span>
            <span className="text-xs font-bold bg-amber-100 text-amber-800 px-2 py-0.5 rounded-full flex items-center space-x-1">
              <Award className="w-3.5 h-3.5 text-amber-600" />
              <span>Best Overall</span>
            </span>
          </div>

          <div>
            <h3 className="text-xl font-extrabold text-slate-900">1D CNN Model</h3>
            <p className="text-xs text-slate-500 mt-0.5">3 Conv1D Blocks + Temporal Pooling</p>
          </div>

          <div className="grid grid-cols-2 gap-3 pt-2 border-t border-slate-100">
            <div>
              <span className="text-xs text-slate-400 block">Test Accuracy</span>
              <span className="text-2xl font-black text-indigo-600">82.63%</span>
            </div>
            <div>
              <span className="text-xs text-slate-400 block">Weighted F1</span>
              <span className="text-2xl font-black text-slate-800">82.30%</span>
            </div>
            <div>
              <span className="text-xs text-slate-400 block">Precision</span>
              <span className="text-sm font-bold text-slate-700">84.34%</span>
            </div>
            <div>
              <span className="text-xs text-slate-400 block">Training Acc</span>
              <span className="text-sm font-bold text-slate-700">93.26%</span>
            </div>
          </div>
        </div>

        {/* MLP Card */}
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-emerald-700 bg-emerald-50 px-2.5 py-1 rounded-full border border-emerald-200">
              Traditional ML
            </span>
            <span className="text-xs text-slate-400 font-medium">Rank #2</span>
          </div>

          <div>
            <h3 className="text-xl font-extrabold text-slate-900">Multi-Layer Perceptron</h3>
            <p className="text-xs text-slate-500 mt-0.5">300 Hidden Neurons, Adaptive LR</p>
          </div>

          <div className="grid grid-cols-2 gap-3 pt-2 border-t border-slate-100">
            <div>
              <span className="text-xs text-slate-400 block">Test Accuracy</span>
              <span className="text-2xl font-black text-emerald-600">79.75%</span>
            </div>
            <div>
              <span className="text-xs text-slate-400 block">Weighted F1</span>
              <span className="text-2xl font-black text-slate-800">79.77%</span>
            </div>
            <div>
              <span className="text-xs text-slate-400 block">Precision</span>
              <span className="text-sm font-bold text-slate-700">81.15%</span>
            </div>
            <div>
              <span className="text-xs text-slate-400 block">Training Acc</span>
              <span className="text-sm font-bold text-slate-700">99.19%</span>
            </div>
          </div>
        </div>

        {/* SVM Card */}
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-blue-700 bg-blue-50 px-2.5 py-1 rounded-full border border-blue-200">
              Traditional ML
            </span>
            <span className="text-xs text-slate-400 font-medium">Rank #3</span>
          </div>

          <div>
            <h3 className="text-xl font-extrabold text-slate-900">Support Vector Machine</h3>
            <p className="text-xs text-slate-500 mt-0.5">RBF Kernel, StandardScaler Pipeline</p>
          </div>

          <div className="grid grid-cols-2 gap-3 pt-2 border-t border-slate-100">
            <div>
              <span className="text-xs text-slate-400 block">Test Accuracy</span>
              <span className="text-2xl font-black text-blue-600">79.12%</span>
            </div>
            <div>
              <span className="text-xs text-slate-400 block">Weighted F1</span>
              <span className="text-2xl font-black text-slate-800">79.68%</span>
            </div>
            <div>
              <span className="text-xs text-slate-400 block">Precision</span>
              <span className="text-sm font-bold text-slate-700">81.54%</span>
            </div>
            <div>
              <span className="text-xs text-slate-400 block">Training Acc</span>
              <span className="text-sm font-bold text-slate-700">99.45%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Model Comparison Table */}
      <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
        <div>
          <h3 className="font-bold text-slate-900 text-base">Model Comparison Matrix</h3>
          <p className="text-xs text-slate-500">Evaluated on identical speaker-independent test split (no data leakage)</p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-slate-700">
            <thead className="bg-slate-50 text-xs uppercase text-slate-400 font-semibold border-b border-slate-200">
              <tr>
                <th className="py-3 px-4">Model</th>
                <th className="py-3 px-4">Architecture Type</th>
                <th className="py-3 px-4">Train Accuracy</th>
                <th className="py-3 px-4">Test Accuracy</th>
                <th className="py-3 px-4">Weighted Precision</th>
                <th className="py-3 px-4">Weighted Recall</th>
                <th className="py-3 px-4">Weighted F1-Score</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {comparisonTable.map((row: any) => {
                const isBest = row.model === "CNN";
                return (
                  <tr key={row.model} className={isBest ? "bg-indigo-50/40 font-semibold" : ""}>
                    <td className="py-3.5 px-4 font-bold text-slate-900 flex items-center space-x-2">
                      <span>{row.model}</span>
                      {isBest && (
                        <span className="text-[10px] bg-indigo-600 text-white font-bold px-1.5 py-0.5 rounded">
                          Winner
                        </span>
                      )}
                    </td>
                    <td className="py-3.5 px-4 text-xs text-slate-500">{row.type}</td>
                    <td className="py-3.5 px-4 font-mono">{row.train_accuracy}%</td>
                    <td className={`py-3.5 px-4 font-mono text-base ${isBest ? "text-indigo-600 font-black" : "font-bold text-slate-800"}`}>
                      {row.test_accuracy}%
                    </td>
                    <td className="py-3.5 px-4 font-mono">{row.precision}%</td>
                    <td className="py-3.5 px-4 font-mono">{row.recall}%</td>
                    <td className="py-3.5 px-4 font-mono font-bold text-slate-900">{row.f1_score}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Metrics Chart */}
      <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
        <div>
          <h3 className="font-bold text-slate-900 text-base">Cross-Model Performance Breakdown</h3>
          <p className="text-xs text-slate-500">Grouped metrics comparison across Accuracy, Precision, Recall, and F1-score</p>
        </div>

        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={groupedChartData} margin={{ top: 10, right: 30, left: 0, bottom: 5 }}>
              <XAxis dataKey="metric" tick={{ fontSize: 12, fill: "#334155", fontWeight: 600 }} />
              <YAxis domain={[70, 90]} tick={{ fontSize: 11, fill: "#64748b" }} unit="%" />
              <Tooltip formatter={(val: any) => [`${val}%`]} />
              <Legend wrapperStyle={{ paddingTop: "10px" }} />
              <Bar dataKey="CNN" fill="#4f46e5" radius={[6, 6, 0, 0]} barSize={24} name="1D CNN (Deep Learning)" />
              <Bar dataKey="MLP" fill="#10b981" radius={[6, 6, 0, 0]} barSize={24} name="MLP (300 Neurons)" />
              <Bar dataKey="SVM" fill="#3b82f6" radius={[6, 6, 0, 0]} barSize={24} name="SVM (RBF Kernel)" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Confusion Matrices Section */}
      <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h3 className="font-bold text-slate-900 text-base">Confusion Matrices</h3>
            <p className="text-xs text-slate-500">Actual vs. predicted emotion class distributions on held-out test audio</p>
          </div>

          {/* Matrix Tabs */}
          <div className="flex items-center space-x-1.5 p-1 bg-slate-100 rounded-xl self-start">
            {[
              { id: "comparison", label: "All 3 Models" },
              { id: "cnn", label: "CNN" },
              { id: "mlp", label: "MLP" },
              { id: "svm", label: "SVM" },
            ].map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveMatrixTab(tab.id as any)}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  activeMatrixTab === tab.id
                    ? "bg-white text-slate-900 shadow-sm"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Matrix Image Display */}
        <div className="p-4 bg-slate-50 rounded-2xl border border-slate-200/80 flex items-center justify-center">
          {activeMatrixTab === "comparison" && (
            <img
              src="http://localhost:8000/static/plots/confusion_matrices_comparison.png"
              alt="Confusion Matrices Comparison"
              className="rounded-xl max-w-full h-auto shadow-sm"
            />
          )}
          {activeMatrixTab === "cnn" && (
            <img
              src="http://localhost:8000/static/plots/confusion_matrix_cnn.png"
              alt="1D CNN Confusion Matrix"
              className="rounded-xl max-w-lg h-auto shadow-sm"
            />
          )}
          {activeMatrixTab === "mlp" && (
            <img
              src="http://localhost:8000/static/plots/confusion_matrix_mlp.png"
              alt="MLP Confusion Matrix"
              className="rounded-xl max-w-lg h-auto shadow-sm"
            />
          )}
          {activeMatrixTab === "svm" && (
            <img
              src="http://localhost:8000/static/plots/confusion_matrix_svm.png"
              alt="SVM Confusion Matrix"
              className="rounded-xl max-w-lg h-auto shadow-sm"
            />
          )}
        </div>
      </div>

      {/* CNN Training Curves */}
      <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
        <div>
          <h3 className="font-bold text-slate-900 text-base">Deep Learning Training Curves</h3>
          <p className="text-xs text-slate-500">1D CNN Training vs. Validation Accuracy and Loss across epochs</p>
        </div>

        <div className="p-4 bg-slate-50 rounded-2xl border border-slate-200/80 flex items-center justify-center">
          <img
            src="http://localhost:8000/static/plots/cnn_training_history.png"
            alt="CNN Training History"
            className="rounded-xl max-w-full h-auto shadow-sm"
          />
        </div>
      </div>
    </div>
  );
}

