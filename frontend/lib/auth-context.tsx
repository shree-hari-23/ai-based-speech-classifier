"use client";

import React, { createContext, useContext, useEffect, useState, useCallback } from "react";
import {
  getCurrentUser,
  loginUser,
  registerUser,
  logoutUser,
  getSmtpStatus,
  sendTestEmail,
  getFluxbaseStatus,
} from "./api";

export interface User {
  id: number;
  email: string;
  full_name: string;
  created_at: string;
  last_login?: string;
}

export interface SmtpStatus {
  configured: boolean;
  host: string;
  port: number;
  user_configured: boolean;
  user_masked: string;
  tls: boolean;
  ssl: boolean;
  logo_found: boolean;
  logo_filename: string;
}

export interface FluxbaseStatus {
  provider: string;
  configured: boolean;
  endpoint: string;
  project_id: string | null;
  api_key_configured: boolean;
  api_key_masked: string;
  mode: string;
  connection_test: boolean;
  users_count?: number;
  users_in_fluxbase?: string[];
  message: string;
  error: string | null;
}

interface AuthContextType {
  user: User | null;
  token: string | null;
  isLoading: boolean;
  smtpStatus: SmtpStatus | null;
  fluxbaseStatus: FluxbaseStatus | null;
  login: (email: string, password: string) => Promise<any>;
  register: (email: string, password: string, fullName: string) => Promise<any>;
  logout: () => Promise<void>;
  refreshSmtpStatus: () => Promise<void>;
  refreshFluxbaseStatus: () => Promise<void>;
  sendTestNotification: (email: string) => Promise<any>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [smtpStatus, setSmtpStatus] = useState<SmtpStatus | null>(null);
  const [fluxbaseStatus, setFluxbaseStatus] = useState<FluxbaseStatus | null>(null);

  // Fetch Fluxbase status
  const refreshFluxbaseStatus = useCallback(async () => {
    try {
      const data = await getFluxbaseStatus();
      setFluxbaseStatus(data);
    } catch (e) {
      console.warn("Could not load Fluxbase status:", e);
    }
  }, []);

  // Fetch SMTP status
  const refreshSmtpStatus = useCallback(async () => {
    try {
      const data = await getSmtpStatus();
      setSmtpStatus(data);
    } catch (e) {
      console.warn("Could not load SMTP status:", e);
    }
  }, []);

  // Hydrate user session from localStorage
  useEffect(() => {
    async function initAuth() {
      setIsLoading(true);
      if (typeof window !== "undefined") {
        const storedToken = localStorage.getItem("emotisound_token");
        if (storedToken) {
          setToken(storedToken);
          try {
            const userData = await getCurrentUser();
            setUser(userData);
          } catch (err) {
            console.warn("Stored token expired or invalid:", err);
            localStorage.removeItem("emotisound_token");
            localStorage.removeItem("emotisound_user");
            setToken(null);
            setUser(null);
          }
        }
      }
      await Promise.all([refreshSmtpStatus(), refreshFluxbaseStatus()]);
      setIsLoading(false);
    }

    initAuth();
  }, [refreshSmtpStatus, refreshFluxbaseStatus]);

  const login = async (email: string, password: string) => {
    setIsLoading(true);
    try {
      const res = await loginUser(email, password);
      if (res.token) {
        setToken(res.token);
        setUser(res.user);
        if (typeof window !== "undefined") {
          localStorage.setItem("emotisound_token", res.token);
          localStorage.setItem("emotisound_user", JSON.stringify(res.user));
        }
      }
      return res;
    } finally {
      setIsLoading(false);
    }
  };

  const register = async (email: string, password: string, fullName: string) => {
    setIsLoading(true);
    try {
      const res = await registerUser(email, password, fullName);
      if (res.token) {
        setToken(res.token);
        setUser(res.user);
        if (typeof window !== "undefined") {
          localStorage.setItem("emotisound_token", res.token);
          localStorage.setItem("emotisound_user", JSON.stringify(res.user));
        }
      }
      return res;
    } finally {
      setIsLoading(false);
    }
  };

  const logout = async () => {
    setIsLoading(true);
    try {
      await logoutUser();
      setToken(null);
      setUser(null);
    } finally {
      setIsLoading(false);
    }
  };

  const sendTestNotification = async (email: string) => {
    return await sendTestEmail(email);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isLoading,
        smtpStatus,
        fluxbaseStatus,
        login,
        register,
        logout,
        refreshSmtpStatus,
        refreshFluxbaseStatus,
        sendTestNotification,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}

