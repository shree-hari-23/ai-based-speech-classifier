"use client";

import React, { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";
import {
  Lock,
  Mail,
  ArrowRight,
  Sparkles,
  AlertCircle,
  CheckCircle2,
  ShieldCheck,
  Radio,
  Send,
  Loader2
} from "lucide-react";

export default function LoginPage() {
  const router = useRouter();
  const { login, smtpStatus } = useAuth();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [notificationInfo, setNotificationInfo] = useState<any | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsSubmitting(true);

    try {
      const res = await login(email, password);
      if (res && res.email_notification) {
        setNotificationInfo(res.email_notification);
      }
      // Redirect after brief delay so user can see notification feedback
      setTimeout(() => {
        router.push("/dashboard");
      }, 1200);
    } catch (err: any) {
      setError(err.message || "Failed to sign in. Please verify your email and password.");
      setIsSubmitting(false);
    }
  };

  const handleFillDemo = () => {
    setEmail("analyst@emotisound.ai");
    setPassword("Password123!");
    setError(null);
  };

  return (
    <div className="min-h-[calc(100vh-8rem)] flex items-center justify-center py-6 px-4">
      <div className="w-full max-w-md">
        {/* Main Card */}
        <div className="bg-white rounded-3xl shadow-xl shadow-slate-200/60 border border-slate-200/80 overflow-hidden">
          
          {/* Neon Logo & Header */}
          <div className="bg-gradient-to-b from-slate-950 via-slate-900 to-indigo-950 p-8 text-center text-white relative">
            <div className="relative inline-block mb-4">
              <div className="absolute -inset-1 rounded-2xl bg-indigo-500 opacity-75 blur-md animate-pulse"></div>
              {/* Project Logo Image */}
              <img
                src="/images/logo.png"
                alt="EmotiSound AI Neon Logo"
                className="relative w-24 h-24 rounded-2xl object-cover border-2 border-indigo-400/50 shadow-2xl mx-auto"
              />
            </div>

            <div className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full bg-indigo-500/20 border border-indigo-400/30 text-indigo-300 text-xs font-semibold uppercase tracking-wider mb-2">
              <ShieldCheck className="w-3.5 h-3.5 text-indigo-400" />
              <span>Secure Authentication</span>
            </div>

            <h1 className="text-2xl font-black tracking-tight text-white">
              EmotiSound AI
            </h1>
            <p className="text-xs text-slate-400 mt-1">
              Acoustic Speech Emotion Recognition
            </p>
          </div>

          {/* Form Area */}
          <div className="p-8 space-y-6">
            <div>
              <h2 className="text-xl font-bold text-slate-900">Sign In to Your Account</h2>
              <p className="text-xs text-slate-500 mt-1">
                Enter your credentials to access deep learning models & prediction logs.
              </p>
            </div>

            {/* Error Message */}
            {error && (
              <div className="p-3.5 rounded-xl bg-rose-50 border border-rose-200 flex items-start space-x-3 text-rose-700 text-xs animate-shake">
                <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-rose-600" />
                <span>{error}</span>
              </div>
            )}

            {/* Notification Delivery Feedback */}
            {notificationInfo && (
              <div className="p-3.5 rounded-xl bg-emerald-50 border border-emerald-200 flex items-start space-x-3 text-emerald-800 text-xs">
                <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5 text-emerald-600" />
                <div>
                  <p className="font-semibold">Sign in successful!</p>
                  <p className="text-emerald-700 mt-0.5">
                    {notificationInfo.sent
                      ? "A security notification with the EmotiSound logo was sent to your email."
                      : notificationInfo.message}
                  </p>
                </div>
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                  Email Address
                </label>
                <div className="relative">
                  <Mail className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
                  <input
                    type="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="analyst@emotisound.ai"
                    className="w-full pl-10 pr-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:bg-white transition-all text-slate-900 placeholder:text-slate-400"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                  Password
                </label>
                <div className="relative">
                  <Lock className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
                  <input
                    type="password"
                    required
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••••••"
                    className="w-full pl-10 pr-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:bg-white transition-all text-slate-900 placeholder:text-slate-400"
                  />
                </div>
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                disabled={isSubmitting}
                className="w-full py-3 px-4 rounded-xl font-bold text-sm text-white bg-indigo-600 hover:bg-indigo-700 transition-all shadow-md shadow-indigo-200 flex items-center justify-center space-x-2 disabled:opacity-60"
              >
                {isSubmitting ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Signing in...</span>
                  </>
                ) : (
                  <>
                    <span>Sign In</span>
                    <ArrowRight className="w-4 h-4" />
                  </>
                )}
              </button>
            </form>

            {/* Demo Account Shortcut */}
            <div className="pt-2">
              <button
                type="button"
                onClick={handleFillDemo}
                className="w-full py-2 px-3 rounded-lg border border-dashed border-indigo-300 bg-indigo-50/50 hover:bg-indigo-50 text-indigo-700 text-xs font-semibold transition-colors flex items-center justify-center space-x-1.5"
              >
                <Sparkles className="w-3.5 h-3.5 text-indigo-500" />
                <span>Auto-fill Demo Analyst Account</span>
              </button>
            </div>

            {/* SMTP Notification Notice */}
            <div className="p-3 bg-slate-50 rounded-xl border border-slate-200/80 text-[11px] text-slate-600 space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-slate-700 flex items-center space-x-1">
                  <Send className="w-3 h-3 text-indigo-500" />
                  <span>Email Notification Dispatch</span>
                </span>
                <span
                  className={`px-1.5 py-0.5 rounded-full text-[10px] font-bold ${
                    smtpStatus?.configured
                      ? "bg-emerald-100 text-emerald-800"
                      : "bg-amber-100 text-amber-800"
                  }`}
                >
                  {smtpStatus?.configured ? "SMTP Active" : "Config in .env"}
                </span>
              </div>
              <p className="text-slate-500 leading-relaxed">
                When you log in, an automated security notification featuring the official{" "}
                <b>EmotiSound AI Neon Logo</b> is generated and dispatched to your email via SMTP.
              </p>
            </div>

            {/* Footer switch to Register */}
            <div className="text-center text-xs text-slate-500 pt-2 border-t border-slate-100">
              Don&apos;t have an account yet?{" "}
              <Link href="/register" className="font-bold text-indigo-600 hover:text-indigo-800">
                Create an account
              </Link>
            </div>

          </div>
        </div>
      </div>
    </div>
  );
}

