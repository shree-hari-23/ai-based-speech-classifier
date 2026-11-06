/**
 * Central API Client for Speech Emotion Recognition Backend.
 * All API requests are encapsulated here to prevent URL duplication.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function getAuthHeader(): Record<string, string> {
  if (typeof window !== "undefined") {
    const token = localStorage.getItem("emotisound_token");
    if (token) {
      return { Authorization: `Bearer ${token}` };
    }
  }
  return {};
}

export class ApiError extends Error {
  status: number;
  data: any;

  constructor(message: string, status: number = 500, data: any = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.data = data;
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errorDetail = `Request failed with status ${res.status}`;
    try {
      const errJson = await res.json();
      if (errJson && errJson.detail) {
        errorDetail = errJson.detail;
      }
    } catch {
      // Non-JSON response
    }
    throw new ApiError(errorDetail, res.status);
  }
  return res.json();
}

/** Check backend health */
export async function getHealthStatus() {
  try {
    const res = await fetch(`${API_BASE}/health`, {
      cache: "no-store",
      headers: { ...getAuthHeader() }
    });
    return await handleResponse<any>(res);
  } catch (err: any) {
    throw new ApiError(
      err.message || "Cannot connect to speech emotion recognition backend. Ensure backend is running at http://localhost:8000.",
      err.status || 0
    );
  }
}

/** Check Fluxbase Cloud Database status */
export async function getFluxbaseStatus() {
  try {
    const res = await fetch(`${API_BASE}/fluxbase-status`, {
      cache: "no-store",
      headers: { ...getAuthHeader() }
    });
    return await handleResponse<{
      provider: string;
      configured: boolean;
      endpoint: string;
      project_id: string | null;
      api_key_configured: boolean;
      api_key_masked: string;
      mode: string;
      connection_test: boolean;
      message: string;
      users_count?: number;
      users_in_fluxbase?: string[];
      error: string | null;
    }>(res);
  } catch (err: any) {
    throw new ApiError(err.message || "Failed to fetch Fluxbase status", err.status || 0);
  }
}

/** Fetch all users stored in Fluxbase users table */
export async function getFluxbaseUsers() {
  try {
    const res = await fetch(`${API_BASE}/fluxbase/users`, {
      cache: "no-store",
      headers: { ...getAuthHeader() }
    });
    return await handleResponse<{
      provider: string;
      project_id: string;
      table: string;
      count: number;
      users: Array<{ id: string; email: string; full_name: string; created_at: string }>;
    }>(res);
  } catch (err: any) {
    throw new ApiError(err.message || "Failed to fetch users from Fluxbase", err.status || 0);
  }
}

/** Check SMTP Configuration and Logo availability */
export async function getSmtpStatus() {
  try {
    const res = await fetch(`${API_BASE}/auth/smtp-status`, {
      cache: "no-store",
      headers: { ...getAuthHeader() }
    });
    return await handleResponse<{
      configured: boolean;
      host: string;
      port: number;
      user_configured: boolean;
      user_masked: string;
      tls: boolean;
      ssl: boolean;
      logo_found: boolean;
      logo_filename: string;
    }>(res);
  } catch (err: any) {
    throw new ApiError(err.message || "Failed to fetch SMTP status", err.status || 0);
  }
}

/** Send test notification email with project logo */
export async function sendTestEmail(email: string) {
  try {
    const res = await fetch(`${API_BASE}/auth/test-email`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...getAuthHeader()
      },
      body: JSON.stringify({ email })
    });
    return await handleResponse<any>(res);
  } catch (err: any) {
    throw new ApiError(err.message || "Failed to dispatch test notification email", err.status || 0);
  }
}

/** Register new user */
export async function registerUser(email: string, password: string, fullName: string) {
  try {
    const res = await fetch(`${API_BASE}/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password, full_name: fullName })
    });
    return await handleResponse<{
      user: { id: number; email: string; full_name: string; created_at: string };
      token: string;
      token_type: string;
      email_notification: any;
    }>(res);
  } catch (err: any) {
    throw new ApiError(err.message || "Registration failed", err.status || 0);
  }
}

/** Log in existing user */
export async function loginUser(email: string, password: string) {
  try {
    const res = await fetch(`${API_BASE}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password })
    });
    return await handleResponse<{
      user: { id: number; email: string; full_name: string; created_at: string; last_login: string };
      token: string;
      token_type: string;
      email_notification: any;
    }>(res);
  } catch (err: any) {
    throw new ApiError(err.message || "Login failed", err.status || 0);
  }
}

/** Get currently authenticated user profile */
export async function getCurrentUser() {
  try {
    const res = await fetch(`${API_BASE}/auth/me`, {
      cache: "no-store",
      headers: { ...getAuthHeader() }
    });
    return await handleResponse<{
      id: number;
      email: string;
      full_name: string;
      created_at: string;
      last_login: string;
    }>(res);
  } catch (err: any) {
    throw new ApiError(err.message || "Authentication token invalid or expired", err.status || 0);
  }
}

/** Log out current user */
export async function logoutUser() {
  try {
    await fetch(`${API_BASE}/auth/logout`, {
      method: "POST",
      headers: { ...getAuthHeader() }
    });
  } catch {
    // Ignore error
  }
  if (typeof window !== "undefined") {
    localStorage.removeItem("emotisound_token");
    localStorage.removeItem("emotisound_user");
  }
}

/** Available ML/DL models */
export async function getModels() {
  try {
    const res = await fetch(`${API_BASE}/models`, { cache: "no-store" });
    return await handleResponse<{ models: any[]; default_model: string }>(res);
  } catch (err: any) {
    throw new ApiError(err.message || "Failed to load available models", err.status || 0);
  }
}

/** Evaluation metrics, confusion matrices, and model comparison */
export async function getAnalytics() {
  try {
    const res = await fetch(`${API_BASE}/analytics`, { cache: "no-store" });
    return await handleResponse<any>(res);
  } catch (err: any) {
    throw new ApiError(err.message || "Failed to load model evaluation metrics", err.status || 0);
  }
}

/** Predict emotion using selected model */
export async function predictEmotion(
  file: File | Blob,
  model: string = "CNN",
  customFilename?: string
) {
  try {
    const formData = new FormData();
    const filename = customFilename || (file instanceof File ? file.name : "recording.wav");
    formData.append("file", file, filename);
    formData.append("model", model);

    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { ...getAuthHeader() },
      body: formData,
    });
    return await handleResponse<any>(res);
  } catch (err: any) {
    throw new ApiError(err.message || "Prediction request failed", err.status || 0);
  }
}

/** Predict emotion using ALL models simultaneously for side-by-side comparison */
export async function predictAllModels(file: File | Blob, customFilename?: string) {
  try {
    const formData = new FormData();
    const filename = customFilename || (file instanceof File ? file.name : "recording.wav");
    formData.append("file", file, filename);

    const res = await fetch(`${API_BASE}/predict-all`, {
      method: "POST",
      headers: { ...getAuthHeader() },
      body: formData,
    });
    return await handleResponse<any>(res);
  } catch (err: any) {
    throw new ApiError(err.message || "Multi-model comparison request failed", err.status || 0);
  }
}

/** Submit live microphone speech recording */
export async function submitMicrophoneRecording(blob: Blob, model: string = "CNN") {
  try {
    const formData = new FormData();
    formData.append("file", blob, "microphone_recording.wav");
    formData.append("model", model);

    const res = await fetch(`${API_BASE}/recording`, {
      method: "POST",
      headers: { ...getAuthHeader() },
      body: formData,
    });
    return await handleResponse<any>(res);
  } catch (err: any) {
    throw new ApiError(err.message || "Microphone recording processing failed", err.status || 0);
  }
}

/** Fetch prediction history with filtering and pagination */
export async function getHistory(params?: {
  search?: string;
  emotion?: string;
  model?: string;
  limit?: number;
  offset?: number;
}) {
  try {
    const query = new URLSearchParams();
    if (params?.search) query.append("search", params.search);
    if (params?.emotion && params.emotion !== "all") query.append("emotion", params.emotion);
    if (params?.model && params.model !== "all") query.append("model", params.model);
    if (params?.limit) query.append("limit", params.limit.toString());
    if (params?.offset) query.append("offset", params.offset.toString());

    const res = await fetch(`${API_BASE}/history?${query.toString()}`, {
      cache: "no-store",
      headers: { ...getAuthHeader() }
    });
    return await handleResponse<{ total: number; limit: number; offset: number; items: any[] }>(res);
  } catch (err: any) {
    throw new ApiError(err.message || "Failed to load prediction history", err.status || 0);
  }
}

/** Delete history item */
export async function deleteHistoryItem(id: number) {
  try {
    const res = await fetch(`${API_BASE}/history/${id}`, {
      method: "DELETE",
      headers: { ...getAuthHeader() }
    });
    return await handleResponse<any>(res);
  } catch (err: any) {
    throw new ApiError(err.message || `Failed to delete history record #${id}`, err.status || 0);
  }
}

/** AI Explainer & Chatbot API */

export interface AiChatMessage {
  role: "user" | "model" | "assistant";
  content: string;
  capability?: string;
  timestamp?: string;
}

export async function sendAiChat(payload: {
  message?: string;
  capability?: string;
  context?: any;
  history?: any[];
}) {
  try {
    const res = await fetch(`${API_BASE}/ai/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...getAuthHeader()
      },
      body: JSON.stringify(payload)
    });
    return await handleResponse<{
      capability: string;
      engine: string;
      model: string;
      reply: string;
      success: boolean;
      notice?: string;
    }>(res);
  } catch (err: any) {
    throw new ApiError(err.message || "AI Explainer chat request failed", err.status || 0);
  }
}

export async function getAiExplanation(payload: {
  capability: string;
  message?: string;
  context?: any;
}) {
  try {
    const res = await fetch(`${API_BASE}/ai/explain`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...getAuthHeader()
      },
      body: JSON.stringify(payload)
    });
    return await handleResponse<{
      capability: string;
      engine: string;
      model: string;
      reply: string;
      success: boolean;
    }>(res);
  } catch (err: any) {
    throw new ApiError(err.message || "Failed to generate AI explanation", err.status || 0);
  }
}

export async function getAiStatus() {
  try {
    const res = await fetch(`${API_BASE}/ai/status`, {
      cache: "no-store",
      headers: { ...getAuthHeader() }
    });
    return await handleResponse<{
      provider: string;
      configured: boolean;
      api_key_masked: string;
      model: string;
      capabilities: Array<{ id: string; name: string; description: string }>;
    }>(res);
  } catch (err: any) {
    throw new ApiError(err.message || "Failed to load AI Explainer status", err.status || 0);
  }
}

/** Dispatches the 6-line formatted speech emotion analysis report via SMTP with project logo */
export async function sendPredictionReport(payload: {
  email: string;
  predicted_emotion: string;
  confidence: number;
  model: string;
  filename?: string;
  report_lines?: string[];
  acoustic_profile?: any;
  timeline?: any;
  model_comparison?: any;
}) {
  try {
    const res = await fetch(`${API_BASE}/predict/send-report`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...getAuthHeader(),
      },
      body: JSON.stringify(payload),
    });
    return await handleResponse<{
      status: string;
      message: string;
      recipient: string;
      report_lines: string[];
    }>(res);
  } catch (err: any) {
    throw new ApiError(err.message || "Failed to dispatch speech emotion report to email", err.status || 0);
  }
}

