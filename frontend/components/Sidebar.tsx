"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Mic,
  BarChart3,
  History,
  Layers,
  Sparkles,
  Info,
  LogIn,
  LogOut
} from "lucide-react";
import { useAuth } from "@/lib/auth-context";

const NAV_ITEMS = [
  {
    name: "Dashboard",
    href: "/dashboard",
    icon: LayoutDashboard,
    description: "System overview & stats"
  },
  {
    name: "Predict Emotion",
    href: "/predict",
    icon: Mic,
    description: "Audio inference & mic"
  },
  {
    name: "Model Analytics",
    href: "/analytics",
    icon: BarChart3,
    description: "Model comparisons & curves"
  },
  {
    name: "Prediction History",
    href: "/history",
    icon: History,
    description: "SQLite prediction logs"
  }
];

export default function Sidebar() {
  const pathname = usePathname();
  const { user, logout } = useAuth();

  return (
    <aside className="w-64 border-r border-slate-200 bg-white flex flex-col justify-between shrink-0 min-h-[calc(100vh-4rem)]">
      <div className="p-4 space-y-6">
        <div>
          <p className="px-3 text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
            Navigation
          </p>
          <nav className="space-y-1">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href || (item.href === "/dashboard" && pathname === "/");

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`flex items-center space-x-3 px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all ${
                    isActive
                      ? "bg-indigo-50 text-indigo-700 shadow-sm shadow-indigo-100 font-semibold"
                      : "text-slate-600 hover:text-slate-900 hover:bg-slate-50"
                  }`}
                >
                  <Icon
                    className={`w-4 h-4 ${
                      isActive ? "text-indigo-600" : "text-slate-400"
                    }`}
                  />
                  <span>{item.name}</span>
                </Link>
              );
            })}
          </nav>
        </div>

        {/* Model Status Card */}
        <div className="p-3.5 bg-slate-50 rounded-2xl border border-slate-200/80 space-y-2.5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-700 flex items-center space-x-1.5">
              <Layers className="w-3.5 h-3.5 text-indigo-600" />
              <span>Active Models</span>
            </span>
            <span className="text-[10px] bg-emerald-100 text-emerald-800 font-bold px-1.5 py-0.5 rounded-full">
              3 Ready
            </span>
          </div>

          <div className="space-y-1.5 text-xs text-slate-600">
            <div className="flex items-center justify-between bg-white px-2 py-1 rounded-lg border border-slate-100">
              <span className="font-medium">1D CNN (Deep Learning)</span>
              <span className="font-bold text-indigo-600">82.6%</span>
            </div>
            <div className="flex items-center justify-between bg-white px-2 py-1 rounded-lg border border-slate-100">
              <span className="font-medium">MLP (300 Neurons)</span>
              <span className="font-medium text-slate-500">79.8%</span>
            </div>
            <div className="flex items-center justify-between bg-white px-2 py-1 rounded-lg border border-slate-100">
              <span className="font-medium">SVM (RBF Kernel)</span>
              <span className="font-medium text-slate-500">79.1%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Footer Info & Auth Card */}
      <div className="p-4 border-t border-slate-100 space-y-3">
        {user ? (
          <div className="p-3 bg-indigo-50/60 rounded-xl border border-indigo-100 flex items-center justify-between">
            <div className="flex items-center space-x-2.5 truncate">
              <div className="w-7 h-7 rounded-lg bg-indigo-600 text-white font-bold flex items-center justify-center text-xs shrink-0">
                {user.full_name.charAt(0).toUpperCase()}
              </div>
              <div className="truncate">
                <p className="text-xs font-bold text-slate-800 truncate">{user.full_name}</p>
                <p className="text-[10px] text-slate-500 truncate">{user.email}</p>
              </div>
            </div>
            <button
              onClick={() => logout()}
              title="Sign Out"
              className="text-slate-400 hover:text-rose-600 p-1 rounded-md hover:bg-rose-50 transition-colors"
            >
              <LogOut className="w-3.5 h-3.5" />
            </button>
          </div>
        ) : (
          <Link
            href="/login"
            className="w-full py-2 px-3 bg-slate-50 hover:bg-indigo-50 border border-slate-200 hover:border-indigo-200 rounded-xl text-xs font-semibold text-slate-700 hover:text-indigo-700 flex items-center justify-center space-x-2 transition-colors"
          >
            <LogIn className="w-3.5 h-3.5 text-indigo-500" />
            <span>Sign In / Register</span>
          </Link>
        )}

        <div className="text-[11px] text-slate-400 space-y-0.5">
          <p className="font-semibold text-slate-600">EmotiSound AI Platform</p>
          <p>4,240 RAVDESS & TESS audio files.</p>
        </div>
      </div>
    </aside>
  );
}

