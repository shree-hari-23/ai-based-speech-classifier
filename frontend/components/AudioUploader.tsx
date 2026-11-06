"use client";

import React, { useState, useRef } from "react";
import { UploadCloud, Music, X, Play, Pause, AlertCircle } from "lucide-react";

interface AudioUploaderProps {
  onFileSelect: (file: File) => void;
  selectedFile: File | null;
  onClear: () => void;
}

export default function AudioUploader({
  onFileSelect,
  selectedFile,
  onClear
}: AudioUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const validateAndSelect = (file: File) => {
    setErrorMsg(null);
    const validExtensions = [".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"];
    const ext = "." + file.name.split(".").pop()?.toLowerCase();

    if (!validExtensions.includes(ext)) {
      setErrorMsg(`Invalid file type (${ext}). Please select an audio file (${validExtensions.join(", ")})`);
      return;
    }

    if (file.size > 25 * 1024 * 1024) {
      setErrorMsg("File exceeds 25 MB limit.");
      return;
    }

    onFileSelect(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSelect(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      validateAndSelect(e.target.files[0]);
    }
  };

  const togglePlayback = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play();
      setIsPlaying(true);
    }
  };

  return (
    <div className="space-y-3">
      {!selectedFile ? (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all flex flex-col items-center justify-center space-y-3 ${
            isDragging
              ? "border-indigo-500 bg-indigo-50/50 scale-[1.01]"
              : "border-slate-300 hover:border-indigo-400 bg-slate-50/60 hover:bg-slate-50"
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*,.wav,.mp3,.ogg,.flac,.m4a"
            className="hidden"
            onChange={handleFileInput}
          />
          <div className="w-12 h-12 rounded-2xl bg-indigo-100 text-indigo-600 flex items-center justify-center shadow-sm">
            <UploadCloud className="w-6 h-6" />
          </div>
          <div>
            <p className="text-sm font-semibold text-slate-800">
              Drag & drop human speech audio here
            </p>
            <p className="text-xs text-slate-500 mt-1">
              Supports WAV, MP3, OGG, FLAC, M4A (Max 25 MB)
            </p>
          </div>
          <button
            type="button"
            className="px-4 py-2 rounded-xl text-xs font-semibold bg-white border border-slate-200 text-slate-700 shadow-sm hover:bg-slate-50 hover:text-indigo-600 transition-colors"
          >
            Browse Audio Files
          </button>
        </div>
      ) : (
        /* Selected File Card */
        <div className="p-4 rounded-2xl border border-indigo-200 bg-indigo-50/30 flex items-center justify-between">
          <div className="flex items-center space-x-3.5">
            <button
              type="button"
              onClick={togglePlayback}
              className="w-10 h-10 rounded-xl bg-indigo-600 text-white flex items-center justify-center shadow-md hover:bg-indigo-700 transition-colors"
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
            </button>
            <div>
              <p className="text-sm font-bold text-slate-900 truncate max-w-xs">
                {selectedFile.name}
              </p>
              <p className="text-xs text-slate-500">
                {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • Audio Ready
              </p>
            </div>
            {/* Audio element for local playback */}
            <audio
              ref={audioRef}
              src={URL.createObjectURL(selectedFile)}
              onEnded={() => setIsPlaying(false)}
              className="hidden"
            />
          </div>

          <button
            type="button"
            onClick={() => {
              if (audioRef.current) audioRef.current.pause();
              setIsPlaying(false);
              onClear();
            }}
            className="p-1.5 rounded-lg text-slate-400 hover:text-rose-600 hover:bg-rose-50 transition-colors"
            title="Remove audio file"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
      )}

      {errorMsg && (
        <div className="flex items-center space-x-2 text-xs text-rose-600 bg-rose-50 border border-rose-200 p-2.5 rounded-xl">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}
    </div>
  );
}

