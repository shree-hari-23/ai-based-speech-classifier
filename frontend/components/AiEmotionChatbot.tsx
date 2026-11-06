"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  Sparkles,
  Bot,
  User,
  Send,
  Loader2,
  Copy,
  Check,
  ChevronDown,
  ChevronUp,
  Brain,
  Clock,
  Scale,
  FileText,
  Volume2,
  ShieldCheck,
  RefreshCw,
  Lightbulb
} from "lucide-react";
import { sendAiChat, AiChatMessage } from "@/lib/api";

export interface PredictionContext {
  filename?: string;
  predicted_emotion?: string;
  confidence?: number;
  model?: string;
  probabilities?: Record<string, number>;
  acoustic_profile?: {
    pitch_mean_hz?: number;
    pitch_std_hz?: number;
    rms_energy?: number;
    rms_db?: number;
    spectral_centroid_hz?: number;
    zero_crossing_rate?: number;
    snr_db?: number;
    clipping_ratio?: number;
    volume_level?: string;
    overall_quality?: string;
    quality_issues?: string[];
  };
  timeline?: Array<{
    segment_index?: number;
    name?: string;
    time_range?: string;
    emotion?: string;
    confidence?: number;
    energy_level?: string;
    pitch_hz?: number;
  }>;
  model_comparison?: Record<string, any>;
}

interface AiEmotionChatbotProps {
  context?: PredictionContext | null;
  className?: string;
}

const CAPABILITIES = [
  {
    id: "emotion_explainer",
    title: "Emotion Explainer",
    icon: Brain,
    badgeColor: "bg-indigo-50 text-indigo-700 border-indigo-200",
    buttonBg: "hover:bg-indigo-50 hover:border-indigo-300",
    description: "Explain why model classified this emotion",
    promptText: "Explain why the model classified this recording as the predicted emotion and what acoustic features supported it.",
  },
  {
    id: "voice_analyst",
    title: "Voice & Timeline Analyst",
    icon: Clock,
    badgeColor: "bg-emerald-50 text-emerald-700 border-emerald-200",
    buttonBg: "hover:bg-emerald-50 hover:border-emerald-300",
    description: "Timeline shifts & recording quality audit",
    promptText: "Explain how the predicted emotion changes throughout the recording timeline and evaluate the audio quality (noise, volume, clipping).",
  },
  {
    id: "model_comparison",
    title: "ML vs DL Comparison",
    icon: Scale,
    badgeColor: "bg-purple-50 text-purple-700 border-purple-200",
    buttonBg: "hover:bg-purple-50 hover:border-purple-300",
    description: "Compare SVM, MLP and 1D CNN models",
    promptText: "Explain why SVM, MLP and CNN produced their respective predictions and how their mathematical architectures differ.",
  },
  {
    id: "report_generator",
    title: "Personalized Report",
    icon: FileText,
    badgeColor: "bg-blue-50 text-blue-700 border-blue-200",
    buttonBg: "hover:bg-blue-50 hover:border-blue-300",
    description: "Generate full personalized analysis report",
    promptText: "Generate a complete personalized audio analysis report summarizing the predicted emotions, confidence scores, acoustic characteristics, and model comparison.",
  },
];

export default function AiEmotionChatbot({ context, className = "" }: AiEmotionChatbotProps) {
  const [messages, setMessages] = useState<AiChatMessage[]>([
    {
      role: "model",
      content:
        "👋 Welcome! I am your **EmotiSound AI Acoustic Intelligence Assistant**.\n\n" +
        "I can explain why your audio was classified into its predicted emotion, trace how emotions evolved throughout the timeline, evaluate recording quality (noise, volume, clipping), compare Traditional ML vs Deep Learning models, or generate a full personalized analysis report.\n\n" +
        "Click one of the four quick capabilities below or ask me any question about your speech recording!",
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    },
  ]);

  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  // Handle triggering one of the 4 core capabilities
  const handleCapabilityClick = async (capabilityId: string, promptText: string) => {
    if (isLoading) return;

    const userMsg: AiChatMessage = {
      role: "user",
      content: promptText,
      capability: capabilityId,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setIsLoading(true);

    try {
      const res = await sendAiChat({
        message: promptText,
        capability: capabilityId,
        context: context || {},
        history: newMessages.map((m) => ({ role: m.role, content: m.content })),
      });

      setMessages((prev) => [
        ...prev,
        {
          role: "model",
          content: res.reply,
          capability: capabilityId,
          timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        },
      ]);
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        {
          role: "model",
          content: `⚠️ Failed to reach AI service: ${err.message || "Please check that the backend is running."}`,
          timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle free-form custom user message
  const handleSendMessage = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const text = inputMessage.trim();
    if (!text || isLoading) return;

    setInputMessage("");

    const userMsg: AiChatMessage = {
      role: "user",
      content: text,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setIsLoading(true);

    try {
      const res = await sendAiChat({
        message: text,
        capability: "emotion_explainer",
        context: context || {},
        history: newMessages.map((m) => ({ role: m.role, content: m.content })),
      });

      setMessages((prev) => [
        ...prev,
        {
          role: "model",
          content: res.reply,
          capability: res.capability,
          timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        },
      ]);
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        {
          role: "model",
          content: `⚠️ AI explanation error: ${err.message || "Please ensure the backend is active."}`,
          timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const copyToClipboard = (text: string, index: number) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  // Simple markdown renderer for clean bullet points, bold text, headers
  const renderFormattedContent = (content: string) => {
    const lines = content.split("\n");
    return lines.map((line, idx) => {
      // H1 / H2 / H3
      if (line.startsWith("# ")) {
        return (
          <h2 key={idx} className="text-base font-extrabold text-slate-900 mt-2 mb-1">
            {line.replace("# ", "")}
          </h2>
        );
      }
      if (line.startsWith("## ")) {
        return (
          <h3 key={idx} className="text-sm font-bold text-slate-800 mt-2 mb-1">
            {line.replace("## ", "")}
          </h3>
        );
      }
      if (line.startsWith("### ")) {
        return (
          <h4 key={idx} className="text-xs font-bold text-slate-700 mt-1.5 mb-0.5">
            {line.replace("### ", "")}
          </h4>
        );
      }
      // Blockquote
      if (line.startsWith("> ")) {
        return (
          <blockquote
            key={idx}
            className="border-l-2 border-indigo-400 pl-3 my-1 text-slate-500 italic text-[11px]"
          >
            {line.replace("> ", "")}
          </blockquote>
        );
      }
      // Bullet items
      if (line.trim().startsWith("- ") || line.trim().startsWith("* ")) {
        const itemText = line.trim().replace(/^[-*]\s+/, "");
        return (
          <li key={idx} className="ml-4 list-disc text-xs text-slate-700 my-0.5 leading-relaxed">
            <span dangerouslySetInnerHTML={{ __html: formatInline(itemText) }} />
          </li>
        );
      }
      // Empty line
      if (!line.trim()) {
        return <div key={idx} className="h-1.5" />;
      }
      // Normal paragraph
      return (
        <p
          key={idx}
          className="text-xs text-slate-700 leading-relaxed my-0.5"
          dangerouslySetInnerHTML={{ __html: formatInline(line) }}
        />
      );
    });
  };

  const formatInline = (text: string) => {
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong class="font-bold text-slate-900">$1</strong>')
      .replace(/\*(.*?)\*/g, '<em class="italic">$1</em>')
      .replace(/`([^`]+)`/g, '<code class="bg-slate-100 text-indigo-700 px-1 py-0.5 rounded text-[11px] font-mono">$1</code>');
  };

  return (
    <div
      className={`bg-white rounded-3xl border border-indigo-100/90 shadow-xl shadow-indigo-900/5 overflow-hidden flex flex-col transition-all ${className}`}
    >
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-900 via-indigo-800 to-slate-900 p-4 sm:p-5 text-white flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-2xl bg-indigo-500/30 border border-indigo-400/40 flex items-center justify-center text-indigo-200 shrink-0 shadow-inner">
            <Sparkles className="w-5 h-5 text-indigo-300 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h3 className="font-extrabold text-base tracking-tight text-white">
                AI Acoustic Intelligence Explainer
              </h3>
              <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-200 border border-indigo-400/30">
                Gemini 2.5 Flash
              </span>
            </div>
            <p className="text-xs text-indigo-200/80 mt-0.5">
              One unified AI assistant powered by 4 acoustic intelligence capabilities
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <button
            type="button"
            onClick={() =>
              setMessages([
                {
                  role: "model",
                  content:
                    "Chat history cleared. Choose any capability below or ask a question regarding your speech emotion prediction!",
                  timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
                },
              ])
            }
            title="Reset Conversation"
            className="p-1.5 rounded-lg bg-white/10 hover:bg-white/20 text-indigo-200 transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
          <button
            type="button"
            onClick={() => setIsCollapsed(!isCollapsed)}
            className="p-1.5 rounded-lg bg-white/10 hover:bg-white/20 text-indigo-200 transition-colors"
          >
            {isCollapsed ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {!isCollapsed && (
        <>
          {/* Active Audio Context Bar */}
          {context && (
            <div className="bg-indigo-50/70 border-b border-indigo-100 px-4 py-2.5 flex flex-wrap items-center justify-between gap-2 text-xs">
              <div className="flex items-center space-x-2 text-slate-700">
                <Volume2 className="w-3.5 h-3.5 text-indigo-600 shrink-0" />
                <span className="font-semibold text-slate-800">
                  {context.filename || "Active Audio"}
                </span>
                <span className="text-slate-400">&bull;</span>
                <span className="font-bold text-indigo-700 capitalize">
                  {context.predicted_emotion} ({context.confidence}%)
                </span>
                {context.acoustic_profile?.snr_db !== undefined && (
                  <>
                    <span className="text-slate-400">&bull;</span>
                    <span className="text-slate-600">
                      SNR: <b>{context.acoustic_profile.snr_db} dB</b>
                    </span>
                  </>
                )}
                {context.acoustic_profile?.pitch_mean_hz !== undefined && (
                  <>
                    <span className="text-slate-400">&bull;</span>
                    <span className="text-slate-600">
                      Pitch: <b>{context.acoustic_profile.pitch_mean_hz} Hz</b>
                    </span>
                  </>
                )}
              </div>

              <div className="flex items-center space-x-1.5 text-[11px] text-emerald-700 bg-emerald-100/70 border border-emerald-200 px-2 py-0.5 rounded-full font-medium">
                <ShieldCheck className="w-3 h-3 text-emerald-600" />
                <span>Audio Context Synced</span>
              </div>
            </div>
          )}

          {/* 4 Capability Trigger Buttons */}
          <div className="p-4 bg-slate-50/70 border-b border-slate-200/80">
            <div className="flex items-center space-x-1.5 mb-2.5">
              <Lightbulb className="w-3.5 h-3.5 text-amber-500" />
              <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
                Choose an AI Capability:
              </span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
              {CAPABILITIES.map((cap) => {
                const IconComponent = cap.icon;
                return (
                  <button
                    key={cap.id}
                    type="button"
                    disabled={isLoading}
                    onClick={() => handleCapabilityClick(cap.id, cap.promptText)}
                    className={`text-left p-3 rounded-2xl bg-white border border-slate-200 shadow-2xs transition-all ${cap.buttonBg} group disabled:opacity-50`}
                  >
                    <div className="flex items-center space-x-2">
                      <div
                        className={`w-7 h-7 rounded-xl flex items-center justify-center border shrink-0 ${cap.badgeColor}`}
                      >
                        <IconComponent className="w-3.5 h-3.5" />
                      </div>
                      <span className="text-xs font-bold text-slate-800 group-hover:text-indigo-600 transition-colors">
                        {cap.title}
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-500 mt-1.5 leading-snug line-clamp-2">
                      {cap.description}
                    </p>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Messages Thread */}
          <div className="flex-1 p-4 sm:p-5 space-y-4 max-h-[460px] overflow-y-auto bg-white/50">
            {messages.map((msg, index) => {
              const isUser = msg.role === "user";
              return (
                <div
                  key={index}
                  className={`flex items-start space-x-3 ${isUser ? "flex-row-reverse space-x-reverse" : ""}`}
                >
                  {/* Avatar */}
                  <div
                    className={`w-8 h-8 rounded-xl flex items-center justify-center shrink-0 text-xs font-bold shadow-xs ${
                      isUser
                        ? "bg-indigo-600 text-white"
                        : "bg-gradient-to-br from-indigo-500 to-purple-600 text-white"
                    }`}
                  >
                    {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
                  </div>

                  {/* Message Bubble */}
                  <div className={`max-w-[85%] sm:max-w-[78%] space-y-1`}>
                    <div
                      className={`p-4 rounded-2xl text-xs shadow-xs relative group ${
                        isUser
                          ? "bg-indigo-600 text-white rounded-tr-none"
                          : "bg-slate-50 text-slate-800 border border-slate-200/80 rounded-tl-none"
                      }`}
                    >
                      {/* Capability Tag if AI output */}
                      {!isUser && msg.capability && (
                        <div className="mb-2">
                          <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-indigo-100 text-indigo-700 border border-indigo-200 uppercase tracking-wide">
                            {msg.capability.replace("_", " ")}
                          </span>
                        </div>
                      )}

                      {/* Content */}
                      <div className="space-y-1">
                        {isUser ? (
                          <p className="leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                        ) : (
                          renderFormattedContent(msg.content)
                        )}
                      </div>

                      {/* Copy button for AI replies */}
                      {!isUser && (
                        <button
                          type="button"
                          onClick={() => copyToClipboard(msg.content, index)}
                          className="absolute top-2 right-2 p-1.5 rounded-lg bg-white border border-slate-200 text-slate-400 hover:text-indigo-600 opacity-0 group-hover:opacity-100 transition-opacity"
                          title="Copy explanation"
                        >
                          {copiedIndex === index ? (
                            <Check className="w-3.5 h-3.5 text-emerald-600" />
                          ) : (
                            <Copy className="w-3.5 h-3.5" />
                          )}
                        </button>
                      )}
                    </div>

                    <span
                      className={`block text-[10px] text-slate-400 px-1 ${isUser ? "text-right" : "text-left"}`}
                    >
                      {msg.timestamp}
                    </span>
                  </div>
                </div>
              );
            })}

            {/* Loading Indicator */}
            {isLoading && (
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 text-white flex items-center justify-center shrink-0">
                  <Bot className="w-4 h-4 animate-spin" />
                </div>
                <div className="bg-slate-50 border border-slate-200/80 rounded-2xl rounded-tl-none p-3.5 flex items-center space-x-2 text-xs text-slate-600">
                  <Loader2 className="w-4 h-4 animate-spin text-indigo-600" />
                  <span>Gemini is analyzing acoustic frequencies & drafting explanation...</span>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input Form */}
          <form
            onSubmit={handleSendMessage}
            className="p-3.5 sm:p-4 bg-white border-t border-slate-200 flex items-center space-x-2"
          >
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="Ask a question about this speech emotion prediction..."
              disabled={isLoading}
              className="flex-1 px-4 py-2.5 text-xs bg-slate-50 border border-slate-200 rounded-xl text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:bg-white transition-all"
            />
            <button
              type="submit"
              disabled={isLoading || !inputMessage.trim()}
              className="px-4 py-2.5 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white font-semibold text-xs rounded-xl transition-all shadow-sm shadow-indigo-200 flex items-center space-x-1.5 shrink-0"
            >
              <span>Ask AI</span>
              <Send className="w-3.5 h-3.5" />
            </button>
          </form>
        </>
      )}
    </div>
  );
}

