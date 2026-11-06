"use client";

import React, { useState, useRef, useEffect } from "react";
import { Mic, Square, Play, Pause, RotateCcw, AlertCircle, CheckCircle } from "lucide-react";

interface AudioRecorderProps {
  onRecordingComplete: (blob: Blob) => void;
  onClear: () => void;
  recordedBlob: Blob | null;
}

export default function AudioRecorder({
  onRecordingComplete,
  onClear,
  recordedBlob
}: AudioRecorderProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    return () => {
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
    };
  }, []);

  const startRecording = async () => {
    setErrorMsg(null);
    audioChunksRef.current = [];
    setRecordingTime(0);

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Microphone API not supported in this browser environment.");
      }

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/wav" });
        onRecordingComplete(audioBlob);
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start(250); // Emit chunk every 250ms
      setIsRecording(true);

      timerIntervalRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
    } catch (err: any) {
      console.error("Microphone access error:", err);
      setErrorMsg(err.message || "Could not access microphone. Please allow microphone permissions in your browser.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
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

  const formatSeconds = (sec: number) => {
    const mins = Math.floor(sec / 60);
    const secs = sec % 60;
    return `${mins}:${secs < 10 ? "0" : ""}${secs}`;
  };

  return (
    <div className="space-y-4">
      {!recordedBlob ? (
        <div className="border border-slate-200 rounded-2xl p-8 bg-slate-50/50 flex flex-col items-center justify-center space-y-4 text-center">
          {/* Animated Mic Button */}
          <button
            type="button"
            onClick={isRecording ? stopRecording : startRecording}
            className={`w-20 h-20 rounded-full flex items-center justify-center transition-all shadow-lg ${
              isRecording
                ? "bg-rose-600 text-white animate-pulse shadow-rose-200 scale-105"
                : "bg-indigo-600 text-white hover:bg-indigo-700 shadow-indigo-200 hover:scale-105"
            }`}
          >
            {isRecording ? <Square className="w-8 h-8 fill-current" /> : <Mic className="w-9 h-9" />}
          </button>

          <div>
            <p className="text-sm font-bold text-slate-800">
              {isRecording ? "Recording Speech in Real Time..." : "Click to Record Voice"}
            </p>
            <p className="text-xs text-slate-500 mt-1">
              {isRecording
                ? `Elapsed Time: ${formatSeconds(recordingTime)} (Click again to stop)`
                : "Speak a short sentence (2 to 5 seconds recommended)"}
            </p>
          </div>

          {isRecording && (
            <div className="flex items-center space-x-1.5 text-xs font-semibold text-rose-600 bg-rose-50 px-3 py-1 rounded-full border border-rose-200">
              <span className="w-2 h-2 rounded-full bg-rose-600 animate-ping" />
              <span>Microphone Active</span>
            </div>
          )}
        </div>
      ) : (
        /* Completed Recording Card */
        <div className="p-4 rounded-2xl border border-emerald-200 bg-emerald-50/40 flex items-center justify-between">
          <div className="flex items-center space-x-3.5">
            <button
              type="button"
              onClick={togglePlayback}
              className="w-10 h-10 rounded-xl bg-emerald-600 text-white flex items-center justify-center shadow-md hover:bg-emerald-700 transition-colors"
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
            </button>
            <div>
              <p className="text-sm font-bold text-slate-900 flex items-center space-x-1.5">
                <CheckCircle className="w-4 h-4 text-emerald-600" />
                <span>Microphone Recording Ready</span>
              </p>
              <p className="text-xs text-slate-500">
                Duration: {formatSeconds(recordingTime)} • Ready for Inference
              </p>
            </div>
            <audio
              ref={audioRef}
              src={URL.createObjectURL(recordedBlob)}
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
            className="flex items-center space-x-1 px-3 py-1.5 rounded-lg text-xs font-semibold text-slate-600 hover:text-slate-900 bg-white border border-slate-200 hover:bg-slate-50 shadow-sm transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Retake</span>
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

