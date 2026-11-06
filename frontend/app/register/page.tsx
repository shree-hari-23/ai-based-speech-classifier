"use client";

import React, { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";
import {
  User,
  Mail,
  Lock,
  ArrowRight,
  ShieldCheck,
  AlertCircle,
  CheckCircle2,
  Sparkles,
  Send,
  Loader2
} from "lucide-react";

export default function RegisterPage() {
  const router = useRouter();
  const { register, smtpStatus } = useAuth();

  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [notificationInfo, setNotificationInfo] = useState<any | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (password !== confirmPassword) {
      setError("Passwords do not match. Please ensure both fields are identical.");
      return;
    }

    if (password.length < 6) {
      setError("Password must be at least 6 characters long.");
      return;
    }

    setIsSubmitting(true);

    try {
      const res = await register(email, password, fullName);
      if (res && res.email_notification) {
        setNotificationInfo(res.email_notification);
      }
      setTimeout(() => {
        router.push("/dashboard");
      }, 1500);
    } catch (err: any) {
      setError(err.message || "Failed to create account. Please check your information.");
      setIsSubmitting(false);
    }
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
              <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
              <span>Create Free Account</span>
            </div>

            <h1 className="text-2xl font-black tracking-tight text-white">
              Join EmotiSound AI
            </h1>
            <p className="text-xs text-slate-400 mt-1">
              Speech Emotion Recognition Platform (CNN, SVM, MLP)
            </p>
          </div>

          {/* Form Area */}
          <div className="p-8 space-y-5">
            <div>
              <h2 className="text-xl font-bold text-slate-900">User Registration</h2>
              <p className="text-xs text-slate-500 mt-1">
                Create your analyst credentials to test models & track speech predictions.
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
                  <p className="font-semibold">Account created successfully!</p>
                  <p className="text-emerald-700 mt-0.5">
                    {notificationInfo.sent
                      ? "A welcome email featuring the EmotiSound AI logo has been dispatched."
                      : notificationInfo.message}
                  </p>
                </div>
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-3.5">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Full Name
                </label>
                <div className="relative">
                  <User className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
                  <input
                    type="text"
                    required
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    placeholder="e.g. Alex Morgan"
                    className="w-full pl-10 pr-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:bg-white transition-all text-slate-900 placeholder:text-slate-400"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Email Address
                </label>
                <div className="relative">
                  <Mail className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
                  <input
                    type="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="alex@example.com"
                    className="w-full pl-10 pr-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:bg-white transition-all text-slate-900 placeholder:text-slate-400"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Password (min 6 characters)
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

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Confirm Password
                </label>
                <div className="relative">
                  <Lock className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
                  <input
                    type="password"
                    required
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    placeholder="••••••••••••"
                    className="w-full pl-10 pr-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:bg-white transition-all text-slate-900 placeholder:text-slate-400"
                  />
                </div>
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                disabled={isSubmitting}
                className="w-full py-3 px-4 mt-2 rounded-xl font-bold text-sm text-white bg-indigo-600 hover:bg-indigo-700 transition-all shadow-md shadow-indigo-200 flex items-center justify-center space-x-2 disabled:opacity-60"
              >
                {isSubmitting ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Creating Account & Sending Notification...</span>
                  </>
                ) : (
                  <>
                    <span>Create Account</span>
                    <ArrowRight className="w-4 h-4" />
                  </>
                )}
              </button>
            </form>

            {/* SMTP Notification Notice */}
            <div className="p-3 bg-slate-50 rounded-xl border border-slate-200/80 text-[11px] text-slate-600 space-y-1">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-slate-700 flex items-center space-x-1">
                  <Send className="w-3 h-3 text-indigo-500" />
                  <span>Welcome Email with Project Logo</span>
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
                A welcome notification embedding the <b>EmotiSound AI Neon Logo</b> will be sent to your email.
              </p>
            </div>

            {/* Footer switch to Login */}
            <div className="text-center text-xs text-slate-500 pt-2 border-t border-slate-100">
              Already have an account?{" "}
              <Link href="/login" className="font-bold text-indigo-600 hover:text-indigo-800">
                Sign In
              </Link>
            </div>

          </div>
        </div>
      </div>
    </div>
  );
}

