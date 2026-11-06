"use client";

import React from "react";
import Link from "next/link";
import {
  ExternalLink,
  LogOut,
  LogIn,
  UserPlus,
} from "lucide-react";
import { useAuth } from "@/lib/auth-context";

export default function Navbar() {
  const { user, logout } = useAuth();

  return (
    <header className="h-16 border-b border-slate-200 bg-white/90 backdrop-blur-md sticky top-0 z-30 px-6 flex items-center justify-between">
      {/* Brand & Project Logo */}
      <Link href="/dashboard" className="flex items-center space-x-3 group">
        <div className="relative">
          <div className="absolute -inset-0.5 rounded-xl bg-indigo-500 opacity-40 group-hover:opacity-75 blur-sm transition-opacity"></div>
          {/* Project Logo from images folder */}
          <img
            src="/images/logo.png"
            alt="EmotiSound AI Logo"
            className="relative w-9 h-9 rounded-xl object-cover shadow-sm border border-indigo-200"
          />
        </div>
        <div>
          <span className="font-extrabold text-slate-900 tracking-tight text-lg group-hover:text-indigo-600 transition-colors">
            EmotiSound AI
          </span>
          <span className="ml-2 text-xs font-semibold px-2 py-0.5 rounded-full bg-indigo-50 text-indigo-700 border border-indigo-200">
            Acoustic SER
          </span>
        </div>
      </Link>

      <div className="flex items-center space-x-3">
        {/* Swagger API Docs Link */}
        <a
          href="http://localhost:8000/docs"
          target="_blank"
          rel="noopener noreferrer"
          className="hidden lg:flex items-center space-x-1.5 text-xs font-medium text-slate-600 hover:text-indigo-600 px-3 py-1.5 rounded-lg hover:bg-slate-50 transition-colors border border-transparent hover:border-slate-200"
        >
          <span>Swagger Docs</span>
          <ExternalLink className="w-3.5 h-3.5" />
        </a>

        {/* User Authentication Status */}
        {user ? (
          <div className="flex items-center space-x-2 pl-2 border-l border-slate-200">
            <div className="flex items-center space-x-2 bg-slate-50 hover:bg-indigo-50/50 border border-slate-200 rounded-xl px-2.5 py-1.5 transition-colors">
              <div className="w-6 h-6 rounded-lg bg-indigo-600 text-white flex items-center justify-center text-xs font-bold shadow-xs">
                {user.full_name.charAt(0).toUpperCase()}
              </div>
              <div className="hidden sm:block text-left">
                <p className="text-xs font-bold text-slate-800 leading-none">{user.full_name}</p>
                <p className="text-[10px] text-slate-500 leading-tight mt-0.5 truncate max-w-[120px]">
                  {user.email}
                </p>
              </div>
            </div>

            <button
              onClick={() => logout()}
              title="Sign Out"
              className="p-1.5 text-slate-400 hover:text-rose-600 hover:bg-rose-50 rounded-lg transition-colors border border-transparent hover:border-rose-200"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        ) : (
          <div className="flex items-center space-x-2 pl-2 border-l border-slate-200">
            <Link
              href="/login"
              className="flex items-center space-x-1.5 text-xs font-semibold text-slate-700 hover:text-indigo-600 px-3 py-1.5 rounded-lg hover:bg-slate-50 transition-colors border border-slate-200"
            >
              <LogIn className="w-3.5 h-3.5 text-slate-400" />
              <span>Sign In</span>
            </Link>
            <Link
              href="/register"
              className="flex items-center space-x-1.5 text-xs font-bold text-white bg-indigo-600 hover:bg-indigo-700 px-3 py-1.5 rounded-lg transition-colors shadow-sm shadow-indigo-200"
            >
              <UserPlus className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Register</span>
            </Link>
          </div>
        )}
      </div>
    </header>
  );
}
