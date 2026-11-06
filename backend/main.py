"""
FastAPI Backend for Acoustic Speech Emotion Recognition.
Provides endpoints for audio upload/recording prediction, multi-model comparison,
model evaluation analytics, and SQLite-backed prediction history.
Programmatically configured to run via `python main.py`.
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import uvicorn
import librosa
from pydantic import BaseModel, EmailStr
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    Query,
    status,
    Depends,
    Header,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# Add backend directory to Python path
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

from config import (
    HOST,
    PORT,
    CORS_ORIGINS,
    MODELS_DIR,
    METRICS_PATH,
    UPLOADS_DIR,
    SUPPORTED_EMOTIONS,
    LOGO_IMAGE_PATH,
    FRONTEND_URL,
    FLUXBASE_URL,
    FLUXBASE_PROJECT_ID,
    FLUXBASE_API_KEY,
    SAMPLE_RATE
)
from services.fluxbase_client import (
    get_fluxbase_status,
    init_fluxbase_schema,
    is_fluxbase_configured,
    get_all_fluxbase_users,
    store_user_email_in_fluxbase,
)
from services.prediction_service import engine
from services.audio_service import save_temp_upload, cleanup_file, extract_audio_metadata
from services.acoustic_analyzer import compute_acoustic_profile, compute_emotion_timeline
from services.ai_explainer_service import (
    explain_speech_emotion,
    get_ai_service_status,
    generate_complete_prediction_explanation,
    build_six_line_report,
)
from services.history_service import (
    init_db,
    save_prediction,
    get_history,
    delete_history_item,
    clear_all_history
)
from services.auth_service import (
    init_auth_db,
    register_user,
    authenticate_user,
    verify_access_token,
    get_user_by_id,
)
from services.email_service import (
    get_smtp_status,
    send_test_notification,
    send_prediction_report_email,
    get_logo_image_path
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SER_Backend")

# Initialize FastAPI App
app = FastAPI(
    title="Acoustic Speech Emotion Recognition API",
    description="Machine Learning & Deep Learning Speech Emotion Recognition System (SVM, MLP, CNN)",
    version="1.0.0"
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for visualization charts and plots
MODELS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static/plots", StaticFiles(directory=str(MODELS_DIR)), name="plots")


@app.on_event("startup")
async def startup_event():
    """Initializes Fluxbase cloud database schema, syncs admin Gmail to Fluxbase users table, and preloads models."""
    logger.info("Initializing backend services with Fluxbase Cloud Database...")
    try:
        init_auth_db()
    except Exception as e:
        logger.warning(f"Fluxbase initialization notice: {e}")
    try:
        engine.load_models()
        logger.info("Models preloaded successfully.")
    except Exception as e:
        logger.warning(f"Could not preload all models on startup: {e}. Models will load on first request.")


# ==============================================================================
# Authentication Schemas & Dependencies
# ==============================================================================

class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: str


class LoginRequest(BaseModel):
    email: str
    password: str


class TestEmailRequest(BaseModel):
    email: str


class AiChatRequest(BaseModel):
    message: Optional[str] = ""
    capability: Optional[str] = "emotion_explainer"
    context: Optional[Dict[str, Any]] = None
    history: Optional[List[Dict[str, Any]]] = None


class AiExplainRequest(BaseModel):
    capability: Optional[str] = "emotion_explainer"
    message: Optional[str] = ""
    context: Optional[Dict[str, Any]] = None


class SendPredictionReportRequest(BaseModel):
    email: EmailStr
    predicted_emotion: str
    confidence: float
    model: str = "CNN"
    filename: Optional[str] = "recording.wav"
    report_lines: Optional[List[str]] = None
    acoustic_profile: Optional[Dict[str, Any]] = None
    timeline: Optional[List[Dict[str, Any]]] = None
    model_comparison: Optional[Dict[str, Any]] = None


async def get_current_user_optional(authorization: Optional[str] = Header(None)) -> Optional[Dict[str, Any]]:
    """Extracts and verifies JWT token from Authorization header if present."""
    if not authorization:
        return None
    try:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]
            payload = verify_access_token(token)
            if payload and "user_id" in payload:
                user = get_user_by_id(payload["user_id"])
                return user
    except Exception as e:
        logger.debug(f"Optional token check failed: {e}")
    return None


async def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Requires valid JWT token in Authorization header."""
    user = await get_current_user_optional(authorization)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in to proceed.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


# ==============================================================================
# Authentication Endpoints
# ==============================================================================

@app.post("/auth/register", tags=["Authentication"])
async def register(req: RegisterRequest):
    """
    Registers a new user, hashes password with bcrypt, persists to SQLite,
    sends a welcome email containing the project logo via SMTP, and returns a JWT token.
    """
    try:
        result = register_user(
            email=req.email,
            password=req.password,
            full_name=req.full_name
        )
        return JSONResponse(status_code=status.HTTP_201_CREATED, content=result)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Registration exception: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Registration failed: {str(e)}")


@app.post("/auth/login", tags=["Authentication"])
async def login(req: LoginRequest, request: Request):
    """
    Authenticates user, updates last_login, sends security login notification email
    containing the project logo via SMTP, and returns a JWT token.
    """
    try:
        client_ip = request.client.host if request.client else "127.0.0.1"
        user_agent = request.headers.get("user-agent", "Web Browser")
        result = authenticate_user(
            email=req.email,
            password=req.password,
            client_ip=client_ip,
            user_agent=user_agent
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=result)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
    except Exception as e:
        logger.error(f"Login exception: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Login failed: {str(e)}")


@app.get("/auth/me", tags=["Authentication"])
async def get_me(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Returns currently authenticated user profile."""
    return {
        "id": current_user["id"],
        "email": current_user["email"],
        "full_name": current_user["full_name"],
        "created_at": current_user["created_at"],
        "last_login": current_user.get("last_login")
    }


@app.post("/auth/logout", tags=["Authentication"])
async def logout():
    """Logs out user (stateless JWT client cleanup confirmation)."""
    return {"status": "success", "message": "Logged out successfully"}


@app.get("/auth/smtp-status", tags=["Authentication"])
async def check_smtp_status():
    """Returns SMTP configuration status and whether the project logo was located."""
    return get_smtp_status()


@app.post("/auth/test-email", tags=["Authentication"])
async def test_email_dispatch(req: TestEmailRequest):
    """Dispatches a test notification email with the project logo to verify SMTP settings."""
    if not req.email or "@" not in req.email:
        raise HTTPException(status_code=400, detail="A valid email address is required.")
    result = send_test_notification(req.email.strip().lower())
    return result


@app.post("/predict/send-report", tags=["Prediction"])
@app.post("/api/send-report", tags=["Prediction"])
async def send_prediction_report_endpoint(req: SendPredictionReportRequest):
    """
    Dispatches the official 6-line formatted speech emotion analysis report
    to the recipient email address embedding the project logo via SMTP.
    """
    clean_email = req.email.strip().lower()
    if not clean_email or "@" not in clean_email:
        raise HTTPException(status_code=400, detail="A valid email address is required.")

    pred_dict = {
        "predicted_emotion": req.predicted_emotion,
        "confidence": req.confidence,
        "model": req.model,
        "filename": req.filename or "recording.wav",
        "acoustic_profile": req.acoustic_profile or {},
        "timeline": req.timeline or [],
        "model_comparison": req.model_comparison or {}
    }

    # Ensure 6 lines exist
    lines = req.report_lines or []
    if len(lines) != 6:
        lines = build_six_line_report(pred_dict)

    result = send_prediction_report_email(
        to_email=clean_email,
        prediction_data=pred_dict,
        report_lines=lines
    )

    if not result.get("sent"):
        raise HTTPException(
            status_code=500,
            detail=result.get("message") or "Failed to deliver speech emotion report via SMTP."
        )

    return {
        "status": "success",
        "message": f"Report successfully dispatched to {clean_email} with project logo.",
        "recipient": clean_email,
        "report_lines": lines
    }



# ==============================================================================
# API Endpoints
# ==============================================================================

@app.get("/", tags=["Status"])
def root():
    """Backend status and service info."""
    return {
        "status": "online",
        "service": "Acoustic Speech Emotion Recognition API",
        "version": "1.0.0",
        "supported_emotions": SUPPORTED_EMOTIONS,
        "available_models": ["CNN", "SVM", "MLP"],
        "docs_url": f"http://localhost:{PORT}/docs"
    }


@app.get("/health", tags=["Status"])
def health_check():
    """Health check verifying Fluxbase database and model states."""
    models_info = engine.get_available_models()
    flux_status = get_fluxbase_status()

    all_ready = any(m["is_loaded"] for m in models_info)

    return {
        "status": "healthy" if all_ready else "degraded",
        "database": {
            "provider": "Fluxbase Cloud Database",
            "configured": flux_status["configured"],
            "mode": flux_status["mode"],
            "endpoint": flux_status["endpoint"]
        },
        "models": models_info
    }


@app.get("/fluxbase-status", tags=["Status"])
def fluxbase_status_endpoint():
    """Returns detailed Fluxbase Cloud Database configuration and connectivity status."""
    return get_fluxbase_status()


@app.get("/fluxbase/users", tags=["Database"])
def get_fluxbase_users_endpoint():
    """Returns all user Gmail accounts stored in the Fluxbase users table."""
    users = get_all_fluxbase_users()
    return {
        "provider": "Fluxbase Cloud Database",
        "project_id": FLUXBASE_PROJECT_ID,
        "table": "users",
        "count": len(users),
        "users": users
    }


@app.get("/api/users", tags=["Database"])
def get_api_users_endpoint():
    """Alias for retrieving users stored in Fluxbase Cloud Database users table."""
    return get_fluxbase_users_endpoint()


@app.get("/models", tags=["Models"])
def get_models():
    """Lists all available ML and DL models with descriptions."""
    return {
        "models": engine.get_available_models(),
        "default_model": "CNN"
    }


@app.post("/predict", tags=["Prediction"])
async def predict_emotion(
    file: UploadFile = File(...),
    model: str = Form("CNN"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
):
    """
    Predicts emotion from an uploaded audio file using the selected model (SVM, MLP, or CNN).
    Computes acoustic features, audio quality diagnostics, timeline segments, and saves to database.
    """
    temp_path = None
    try:
        temp_path, file_size = await save_temp_upload(file)
        audio_meta = extract_audio_metadata(temp_path)

        # Load audio array for acoustic analysis & timeline
        y, sr = librosa.load(str(temp_path), sr=SAMPLE_RATE, mono=True)
        acoustic_profile = compute_acoustic_profile(y, sr)
        timeline = compute_emotion_timeline(y, sr, engine=engine)

        # Run inference
        result = engine.predict(temp_path, model_name=model)

        # Multi-model comparative context for AI Explainer
        try:
            comp_result = engine.predict_all(temp_path)
            result["model_comparison"] = comp_result.get("models", {})
        except Exception:
            result["model_comparison"] = {}

        result["acoustic_profile"] = acoustic_profile
        result["timeline"] = timeline

        # Persist to database history
        record = save_prediction(
            filename=file.filename or "uploaded_audio.wav",
            predicted_emotion=result["predicted_emotion"],
            confidence=result["confidence"],
            model_used=result["model"],
            probabilities=result["probabilities"],
            file_size=file_size,
            duration=audio_meta["duration"],
            waveform_sample=result.get("waveform_sample"),
            user_id=current_user["id"] if current_user else None
        )

        result["id"] = record["id"]
        result["filename"] = file.filename
        result["audio_metadata"] = audio_meta

        # Automatically generate complete AI explanations after predicting
        try:
            result["ai_explanation"] = generate_complete_prediction_explanation(result)
            result["six_line_report"] = result["ai_explanation"].get("six_line_report") or build_six_line_report(result)
        except Exception as e_ai:
            logger.warning(f"AI explanation generation notice: {e_ai}")
            result["ai_explanation"] = None
            result["six_line_report"] = build_six_line_report(result)

        return JSONResponse(status_code=status.HTTP_200_OK, content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    finally:
        cleanup_file(temp_path)


@app.post("/predict-all", tags=["Prediction"])
async def predict_all_models(
    file: UploadFile = File(...),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
):
    """
    Runs emotion prediction across all three models (SVM, MLP, and CNN) simultaneously
    for direct comparative analysis, including full acoustic profile and timeline.
    """
    temp_path = None
    try:
        temp_path, file_size = await save_temp_upload(file)
        audio_meta = extract_audio_metadata(temp_path)

        y, sr = librosa.load(str(temp_path), sr=SAMPLE_RATE, mono=True)
        acoustic_profile = compute_acoustic_profile(y, sr)
        timeline = compute_emotion_timeline(y, sr, engine=engine)

        # Run inference across all models
        result = engine.predict_all(temp_path)
        result["model_comparison"] = result.get("models", {})
        result["acoustic_profile"] = acoustic_profile
        result["timeline"] = timeline
        result["predicted_emotion"] = result["consensus_emotion"]
        result["confidence"] = result["consensus_confidence"]
        result["model"] = "CNN"
        result["probabilities"] = result["models"]["CNN"]["probabilities"]

        # Persist consensus/best model prediction
        record = save_prediction(
            filename=file.filename or "uploaded_audio.wav",
            predicted_emotion=result["consensus_emotion"],
            confidence=result["consensus_confidence"],
            model_used="ALL_MODELS",
            probabilities=result["models"]["CNN"]["probabilities"],
            file_size=file_size,
            duration=audio_meta["duration"],
            waveform_sample=result.get("waveform_sample"),
            user_id=current_user["id"] if current_user else None
        )

        result["id"] = record["id"]
        result["filename"] = file.filename
        result["audio_metadata"] = audio_meta

        # Automatically generate complete AI explanations after predicting
        try:
            result["ai_explanation"] = generate_complete_prediction_explanation(result)
            result["six_line_report"] = result["ai_explanation"].get("six_line_report") or build_six_line_report(result)
        except Exception as e_ai:
            logger.warning(f"AI explanation generation notice: {e_ai}")
            result["ai_explanation"] = None
            result["six_line_report"] = build_six_line_report(result)

        return JSONResponse(status_code=status.HTTP_200_OK, content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multi-model prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Multi-model prediction error: {str(e)}")
    finally:
        cleanup_file(temp_path)


@app.post("/recording", tags=["Prediction"])
async def process_microphone_recording(
    file: UploadFile = File(...),
    model: str = Form("CNN"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
):
    """
    Dedicated endpoint for processing live microphone speech recordings.
    """
    temp_path = None
    try:
        temp_path, file_size = await save_temp_upload(file)
        audio_meta = extract_audio_metadata(temp_path)

        y, sr = librosa.load(str(temp_path), sr=SAMPLE_RATE, mono=True)
        acoustic_profile = compute_acoustic_profile(y, sr)
        timeline = compute_emotion_timeline(y, sr, engine=engine)

        result = engine.predict(temp_path, model_name=model)
        try:
            comp_result = engine.predict_all(temp_path)
            result["model_comparison"] = comp_result.get("models", {})
        except Exception:
            result["model_comparison"] = {}

        result["acoustic_profile"] = acoustic_profile
        result["timeline"] = timeline

        record = save_prediction(
            filename="microphone_recording.wav",
            predicted_emotion=result["predicted_emotion"],
            confidence=result["confidence"],
            model_used=result["model"],
            probabilities=result["probabilities"],
            file_size=file_size,
            duration=audio_meta["duration"],
            waveform_sample=result.get("waveform_sample"),
            user_id=current_user["id"] if current_user else None
        )

        result["id"] = record["id"]
        result["filename"] = "microphone_recording.wav"
        result["audio_metadata"] = audio_meta

        # Automatically generate complete AI explanations after predicting
        try:
            result["ai_explanation"] = generate_complete_prediction_explanation(result)
            result["six_line_report"] = result["ai_explanation"].get("six_line_report") or build_six_line_report(result)
        except Exception as e_ai:
            logger.warning(f"AI explanation generation notice: {e_ai}")
            result["ai_explanation"] = None
            result["six_line_report"] = build_six_line_report(result)

        return JSONResponse(status_code=status.HTTP_200_OK, content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recording processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Recording processing error: {str(e)}")
    finally:
        cleanup_file(temp_path)


# ==============================================================================
# AI Explainer & Chatbot Endpoints (Powered by Google Gemini)
# ==============================================================================

@app.post("/ai/chat", tags=["AI Explainer"])
async def ai_chat_endpoint(req: AiChatRequest):
    """
    Interactive AI Acoustic Chatbot supporting four capabilities:
    1. emotion_explainer: Explains why the model classified audio as Happy, Sad, Angry, etc.
    2. voice_analyst: Explains emotional changes across timeline segments & evaluates audio quality.
    3. model_comparison: Explains why SVM, MLP, and 1D CNN agree or produce different predictions.
    4. report_generator: Generates a complete personalized audio analysis summary report.
    """
    try:
        capability = req.capability or "emotion_explainer"
        context = req.context or {}
        history = req.history or []
        message = req.message or ""

        result = explain_speech_emotion(
            capability=capability,
            user_message=message,
            context=context,
            history=history
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=result)
    except Exception as e:
        logger.error(f"AI chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI chat error: {str(e)}")


@app.post("/ai/explain", tags=["AI Explainer"])
async def ai_explain_endpoint(req: AiExplainRequest):
    """
    One-click AI explanation generation for a specified capability.
    """
    try:
        capability = req.capability or "emotion_explainer"
        context = req.context or {}
        message = req.message or ""

        result = explain_speech_emotion(
            capability=capability,
            user_message=message,
            context=context
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=result)
    except Exception as e:
        logger.error(f"AI explanation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI explanation error: {str(e)}")


@app.get("/ai/status", tags=["AI Explainer"])
def ai_status_endpoint():
    """
    Returns Google Gemini AI integration and capability status.
    """
    return get_ai_service_status()


@app.get("/analytics", tags=["Analytics"])
def get_analytics():
    """
    Returns actual evaluation metrics, comparison tables, and static plot URLs.
    """
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Evaluation metrics not found. Please run training and evaluation first."
        )

    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    # Attach visualization static image URLs
    base_plot_url = f"http://localhost:{PORT}/static/plots"
    metrics["plot_urls"] = {
        "confusion_matrix_svm": f"{base_plot_url}/confusion_matrix_svm.png",
        "confusion_matrix_mlp": f"{base_plot_url}/confusion_matrix_mlp.png",
        "confusion_matrix_cnn": f"{base_plot_url}/confusion_matrix_cnn.png",
        "confusion_matrices_comparison": f"{base_plot_url}/confusion_matrices_comparison.png",
        "model_comparison_chart": f"{base_plot_url}/model_comparison_chart.png",
        "cnn_accuracy": f"{base_plot_url}/cnn_accuracy.png",
        "cnn_loss": f"{base_plot_url}/cnn_loss.png",
        "cnn_training_history": f"{base_plot_url}/cnn_training_history.png",
        "dataset_exploration": f"{base_plot_url}/dataset_exploration.png"
    }

    return metrics


@app.get("/history", tags=["History"])
def list_prediction_history(
    search: Optional[str] = Query(None, description="Search by filename or emotion"),
    emotion: Optional[str] = Query(None, description="Filter by predicted emotion"),
    model: Optional[str] = Query(None, description="Filter by model used"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0)
):
    """Retrieves SQLite-persisted prediction history with filtering and pagination."""
    return get_history(search=search, emotion=emotion, model=model, limit=limit, offset=offset)


@app.delete("/history/{record_id}", tags=["History"])
def delete_history_record(record_id: int):
    """Deletes a prediction record by its ID."""
    success = delete_history_item(record_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"History record #{record_id} not found.")
    return {"status": "success", "message": f"Record #{record_id} successfully deleted."}


@app.delete("/history", tags=["History"])
def clear_all_history_records():
    """Clears all prediction history records."""
    deleted_count = clear_all_history()
    return {"status": "success", "deleted_count": deleted_count}


# ==============================================================================
# Programmatic Server Execution
# ==============================================================================
if __name__ == "__main__":
    print(f"\n=======================================================")
    print(f" Starting Acoustic Speech Emotion Recognition API")
    print(f" URL: http://localhost:{PORT}")
    print(f" Swagger UI: http://localhost:{PORT}/docs")
    print(f"=======================================================\n")
    uvicorn.run(app, host=HOST, port=PORT)
