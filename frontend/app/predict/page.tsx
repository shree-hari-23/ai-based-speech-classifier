"use client";

import React, { useState } from "react";
import {
  UploadCloud,
  Mic,
  Cpu,
  Sparkles,
  Loader2,
  AlertCircle,
  HelpCircle,
  Volume2
} from "lucide-react";
import AudioUploader from "@/components/AudioUploader";
import AudioRecorder from "@/components/AudioRecorder";
import EmotionCard from "@/components/EmotionCard";
import ConfidenceChart from "@/components/ConfidenceChart";
import Waveform from "@/components/Waveform";
import ModelComparison from "@/components/ModelComparison";
import AiPredictionExplanation from "@/components/AiPredictionExplanation";
import { predictEmotion, predictAllModels, ApiError } from "@/lib/api";

type InputTab = "upload" | "record";
type ModelChoice = "CNN" | "SVM" | "MLP" | "ALL";

export default function PredictPage() {
  const [activeTab, setActiveTab] = useState<InputTab>("upload");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelChoice>("CNN");

  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Prediction Results State
  const [singleResult, setSingleResult] = useState<any>(null);
  const [multiResult, setMultiResult] = useState<any>(null);

  const handlePredict = async () => {
    const audioPayload = activeTab === "upload" ? selectedFile : recordedBlob;
    if (!audioPayload) {
      setErrorMessage("Please upload an audio file or record speech before predicting.");
      return;
    }

    setIsLoading(true);
    setErrorMessage(null);

    try {
      if (selectedModel === "ALL") {
        const response = await predictAllModels(audioPayload);
        setMultiResult(response);
        setSingleResult(null);
      } else {
        const response = await predictEmotion(audioPayload, selectedModel);
        setSingleResult(response);
        setMultiResult(null);
      }
    } catch (err: any) {
      console.error("Prediction failed:", err);
      if (err instanceof ApiError) {
        setErrorMessage(err.message);
      } else {
        setErrorMessage("Prediction failed. Please ensure the backend is running at http://localhost:8000.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const hasAudio = activeTab === "upload" ? !!selectedFile : !!recordedBlob;

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl lg:text-3xl font-extrabold text-slate-900 tracking-tight">
          Acoustic Speech Emotion Prediction
        </h1>
        <p className="text-slate-500 text-sm mt-1">
          Upload human speech or record voice via microphone to predict emotional state from vocal acoustics.
        </p>
      </div>

      {/* Main Grid: Input Section & Controls */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Audio Input (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-5">
            {/* Tab Selection: Upload vs Record */}
            <div className="flex border-b border-slate-200">
              <button
                type="button"
                onClick={() => setActiveTab("upload")}
                className={`flex items-center space-x-2 py-3 px-5 text-sm font-semibold border-b-2 transition-all ${
                  activeTab === "upload"
                    ? "border-indigo-600 text-indigo-600"
                    : "border-transparent text-slate-500 hover:text-slate-800"
                }`}
              >
                <UploadCloud className="w-4 h-4" />
                <span>Upload Audio File</span>
              </button>

              <button
                type="button"
                onClick={() => setActiveTab("record")}
                className={`flex items-center space-x-2 py-3 px-5 text-sm font-semibold border-b-2 transition-all ${
                  activeTab === "record"
                    ? "border-indigo-600 text-indigo-600"
                    : "border-transparent text-slate-500 hover:text-slate-800"
                }`}
              >
                <Mic className="w-4 h-4" />
                <span>Record via Microphone</span>
              </button>
            </div>

            {/* Tab Content */}
            {activeTab === "upload" ? (
              <AudioUploader
                selectedFile={selectedFile}
                onFileSelect={(file) => {
                  setSelectedFile(file);
                  setErrorMessage(null);
                }}
                onClear={() => {
                  setSelectedFile(null);
                  setSingleResult(null);
                  setMultiResult(null);
                }}
              />
            ) : (
              <AudioRecorder
                recordedBlob={recordedBlob}
                onRecordingComplete={(blob) => {
                  setRecordedBlob(blob);
                  setErrorMessage(null);
                }}
                onClear={() => {
                  setRecordedBlob(null);
                  setSingleResult(null);
                  setMultiResult(null);
                }}
              />
            )}

            {/* Model Selector */}
            <div className="space-y-2.5 pt-2">
              <label className="text-xs font-bold text-slate-700 uppercase tracking-wider block">
                Select Model Architecture
              </label>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                {[
                  { id: "CNN", label: "1D CNN", sub: "Deep Learning (82.6%)" },
                  { id: "MLP", label: "MLP", sub: "300 Neurons (79.8%)" },
                  { id: "SVM", label: "SVM", sub: "RBF Kernel (79.1%)" },
                  { id: "ALL", label: "All Models", sub: "Compare 3 Models" },
                ].map((m) => (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => setSelectedModel(m.id as ModelChoice)}
                    className={`p-3 rounded-xl border text-left transition-all ${
                      selectedModel === m.id
                        ? "border-indigo-600 bg-indigo-50/60 ring-2 ring-indigo-600/20"
                        : "border-slate-200 hover:border-slate-300 bg-white"
                    }`}
                  >
                    <span className="block text-xs font-bold text-slate-900">{m.label}</span>
                    <span className="block text-[10px] text-slate-500 mt-0.5">{m.sub}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Error Message Alert */}
            {errorMessage && (
              <div className="flex items-center space-x-2.5 text-xs text-rose-700 bg-rose-50 border border-rose-200 p-3.5 rounded-xl">
                <AlertCircle className="w-4 h-4 shrink-0 text-rose-600" />
                <span>{errorMessage}</span>
              </div>
            )}

            {/* Predict Button */}
            <button
              type="button"
              disabled={!hasAudio || isLoading}
              onClick={handlePredict}
              className={`w-full py-3.5 rounded-xl text-sm font-bold flex items-center justify-center space-x-2 transition-all shadow-md ${
                !hasAudio || isLoading
                  ? "bg-slate-200 text-slate-400 cursor-not-allowed shadow-none"
                  : "bg-indigo-600 hover:bg-indigo-700 text-white shadow-indigo-200 hover:shadow-lg"
              }`}
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Extracting Features & Predicting...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-5 h-5" />
                  <span>
                    {selectedModel === "ALL" ? "Compare All 3 Models" : `Predict Emotion (${selectedModel})`}
                  </span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Right Column: Information & Guidelines (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          <div className="bg-slate-50 p-6 rounded-2xl border border-slate-200/80 space-y-4 text-xs text-slate-600 leading-relaxed">
            <h3 className="font-bold text-slate-900 text-sm flex items-center space-x-2">
              <HelpCircle className="w-4 h-4 text-indigo-600" />
              <span>Acoustic Pipeline Overview</span>
            </h3>

            <div className="space-y-3">
              <div className="p-3 bg-white rounded-xl border border-slate-200/60 shadow-sm">
                <span className="font-bold text-slate-800 block text-xs">1. Signal Preprocessing</span>
                <p className="text-[11px] text-slate-500 mt-0.5">
                  Resampled to 22.05 kHz mono, silence trimmed, amplitude normalized, center-padded to 3.0s.
                </p>
              </div>

              <div className="p-3 bg-white rounded-xl border border-slate-200/60 shadow-sm">
                <span className="font-bold text-slate-800 block text-xs">2. Acoustic Feature Extraction</span>
                <p className="text-[11px] text-slate-500 mt-0.5">
                  40 MFCCs + 12 Chroma STFT + 128 Mel bands. Mean pooled to 180 features for SVM/MLP; 130 sequential frames for Conv1D.
                </p>
              </div>

              <div className="p-3 bg-white rounded-xl border border-slate-200/60 shadow-sm">
                <span className="font-bold text-slate-800 block text-xs">3. Real Probability Estimation</span>
                <p className="text-[11px] text-slate-500 mt-0.5">
                  Softmax and Platt scaling generate calibrated confidence percentages across all 8 classes.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Prediction Output Section */}
      {(singleResult || multiResult) && (
        <div className="space-y-6 pt-4 border-t border-slate-200">
          <h2 className="text-xl font-extrabold text-slate-900">Prediction Results</h2>

          {/* Single Model Result */}
          {singleResult && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                <div className="lg:col-span-6">
                  <EmotionCard
                    emotion={singleResult.predicted_emotion}
                    confidence={singleResult.confidence}
                    model={singleResult.model}
                    disclaimer={singleResult.disclaimer}
                  />
                </div>
                <div className="lg:col-span-6">
                  <Waveform
                    points={singleResult.waveform_sample}
                    duration={singleResult.audio_metadata?.duration}
                  />
                </div>
              </div>

              <ConfidenceChart
                probabilities={singleResult.probabilities}
                predictedEmotion={singleResult.predicted_emotion}
              />
            </div>
          )}

          {/* Multi-Model Comparison Result */}
          {multiResult && (
            <div className="space-y-6">
              <ModelComparison
                modelsData={multiResult.models}
                consensusEmotion={multiResult.consensus_emotion}
              />

              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                <div className="lg:col-span-6">
                  <ConfidenceChart
                    probabilities={multiResult.models?.CNN?.probabilities || {}}
                    predictedEmotion={multiResult.consensus_emotion}
                  />
                </div>
                <div className="lg:col-span-6">
                  <Waveform points={multiResult.waveform_sample} />
                </div>
              </div>
            </div>
          )}

          {/* Automated AI In-Depth Explanations (Rendered directly after prediction) */}
          <div className="pt-2">
            <AiPredictionExplanation
              prediction={
                singleResult
                  ? {
                      predicted_emotion: singleResult.predicted_emotion,
                      confidence: singleResult.confidence,
                      model: singleResult.model,
                      filename: singleResult.filename || selectedFile?.name || "recording.wav",
                      probabilities: singleResult.probabilities,
                      acoustic_profile: singleResult.acoustic_profile,
                      timeline: singleResult.timeline,
                      model_comparison: singleResult.model_comparison,
                      ai_explanation: singleResult.ai_explanation,
                      six_line_report: singleResult.six_line_report || singleResult.ai_explanation?.six_line_report,
                    }
                  : {
                      predicted_emotion: multiResult.consensus_emotion,
                      confidence: multiResult.consensus_confidence,
                      model: "ALL_MODELS",
                      filename: multiResult.filename || selectedFile?.name || "recording.wav",
                      probabilities: multiResult.models?.CNN?.probabilities || {},
                      acoustic_profile: multiResult.acoustic_profile,
                      timeline: multiResult.timeline,
                      model_comparison: multiResult.models,
                      ai_explanation: multiResult.ai_explanation,
                      six_line_report: multiResult.six_line_report || multiResult.ai_explanation?.six_line_report,
                    }
              }
            />
          </div>
        </div>
      )}
    </div>
  );
}

