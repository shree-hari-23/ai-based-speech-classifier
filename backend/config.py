import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Load environment variables from backend/.env first, then root .env
load_dotenv(BASE_DIR / ".env")
load_dotenv(PROJECT_ROOT / ".env")

DATASETS_DIR = BASE_DIR / "datasets"
RAVDESS_DIR = DATASETS_DIR / "RAVDESS"
TESS_DIR = DATASETS_DIR / "TESS"

MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"
IMAGES_DIR = PROJECT_ROOT / "images"
LOGO_IMAGE_PATH = IMAGES_DIR / "EmotiSound AI Neon Logo.png"

# Fluxbase Cloud Database Configuration (as specified in Fluxbase Integration Guide)
FLUXBASE_URL = os.getenv("FLUXBASE_URL", "https://fluxbase.vercel.app/api/execute-sql").strip()
FLUXBASE_PROJECT_ID = os.getenv("FLUXBASE_PROJECT_ID", "").strip()
FLUXBASE_API_KEY = os.getenv("FLUXBASE_API_KEY", "").strip()

# Create required directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RAVDESS_DIR.mkdir(parents=True, exist_ok=True)
TESS_DIR.mkdir(parents=True, exist_ok=True)

# Model file paths
SVM_MODEL_PATH = MODELS_DIR / "svm_model.pkl"
MLP_MODEL_PATH = MODELS_DIR / "mlp_model.pkl"
CNN_MODEL_PATH = MODELS_DIR / "cnn_model.keras"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"

# Supported target emotions (8 classes)
SUPPORTED_EMOTIONS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised"
]

# RAVDESS emotion code mapping
# Format: 03-01-XX-01-01-01-01.wav -> XX is the 3rd token (1-indexed)
RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# TESS emotion mapping (from directory or filename keywords)
# Note: 'ps' represents 'pleasant surprise' -> mapped to 'surprised'
# 'fear' -> mapped to 'fearful'
TESS_EMOTION_MAP = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fearful",
    "happy": "happy",
    "neutral": "neutral",
    "ps": "surprised",
    "pleasant_surprise": "surprised",
    "pleasant_surprised": "surprised",
    "sad": "sad"
}

# Audio Preprocessing & Feature Extraction configuration
SAMPLE_RATE = 22050
DURATION = 3.0  # seconds
TARGET_LENGTH = int(SAMPLE_RATE * DURATION)  # 66,150 samples

# Feature Extraction specifications:
# 40 MFCC + 12 Chroma STFT + 128 Mel Spectrogram = 180 tabular features (mean pooled)
N_MFCC = 40
N_CHROMA = 12
N_MELS = 128
FEATURE_DIM = N_MFCC + N_CHROMA + N_MELS  # 180

# Sequential features configuration for 1D CNN:
# Mel Spectrogram sequence or MFCC sequence over time
# With 3.0s at SR=22050 and hop_length=512 -> ~130 time frames
HOP_LENGTH = 512
N_FFT = 2048

# Server configuration
HOST = "0.0.0.0"
PORT = 8000
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Authentication (JWT) Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "emotisound-acoustic-speech-emotion-recognition-secret-key-2026")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "168"))  # 7 days

# SMTP Email Configuration
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL", "").strip() or SMTP_USER
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "EmotiSound AI - Acoustic SER").strip()
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "True").lower() in ("true", "1", "yes")
SMTP_USE_SSL = os.getenv("SMTP_USE_SSL", "False").lower() in ("true", "1", "yes")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# Gemini AI Integration Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()


