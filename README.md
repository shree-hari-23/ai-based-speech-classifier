# Acoustic Speech Emotion Recognition Using Machine Learning and Deep Learning

An intelligent full-stack AI/ML/DL application that analyzes acoustic properties of human speech recordings and predicts the expressed emotional state using both Traditional Machine Learning (**Support Vector Machine**, **Multi-Layer Perceptron**) and Deep Learning (**1D Convolutional Neural Network**).

---

## 1. Project Overview & Architecture

Human speech carries emotional signals embedded across vocal frequency contours, pitch dynamics, harmonic spectra, and rhythmic cadences. This project extracts engineered acoustic features and sequential time-frequency representations to classify human speech into **8 canonical emotion classes**:
`Neutral`, `Calm`, `Happy`, `Sad`, `Angry`, `Fearful`, `Disgust`, and `Surprised`.

### Architecture Flow:
```text
Raw Audio Recording (Upload or Microphone)
   │
   ▼
Audio Preprocessing Pipeline (Librosa)
   • 22,050 Hz Mono Resampling
   • Top-dB Silence Trimming (28 dB)
   • Dynamic Amplitude Peak Normalization (0.95)
   • Pre-emphasis High-Pass Filter (0.97)
   • Fixed Duration Enforcement (3.0s = 66,150 samples)
   │
   ├─────────────────────────────────────────────────┐
   ▼                                                 ▼
Traditional ML Feature Extraction        Deep Learning Sequential Matrix
   • 40 MFCCs (Time-Averaged)                • 130 Time Frames x 180 Channels
   • 12 Chroma STFT (Time-Averaged)          • Time-Aligned Spectrogram Sequence
   • 128 Mel Bands (Time-Averaged)           • Preserves Temporal Dynamics
   • Shape: (180,)                           • Shape: (130, 180)
   │                                                 │
   ├──► Support Vector Machine (RBF Kernel)          ▼
   │    • Platt Scaled Probabilities          1D Convolutional Neural Network
   │                                                 • Conv1D (64) + MaxPool + Dropout
   └──► Multi-Layer Perceptron (300 Neurons)         • Conv1D (128) + MaxPool + Dropout
        • Adaptive Learning Rate                     • Conv1D (256) + Dropout
                                                     • Dense (8, Softmax)
   │                                                 │
   └───────────────────────┬─────────────────────────┘
                           ▼
             Calibrated Class Probabilities
                           │
                           ▼
          Interactive Next.js Dashboard UI
   • Waveform Visualization
   • Confidence Percentages
   • Cross-Model Comparison Table
   • SQLite Persistent Prediction History
```

---

## 2. Actual Model Evaluation Results

Evaluated on **800 held-out test recordings** using a **speaker-independent split strategy** (test actors never appear in the training set to prevent vocal timbre data leakage):

| Model | Architecture | Training Accuracy | Testing Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score |
|---|---|---|---|---|---|---|
| **1D CNN** | 3-Block Conv1D + MaxPool + Dropout | **93.26%** | **82.63%** | **84.34%** | **82.62%** | **82.30%** |
| **MLP** | MLPClassifier (300 neurons, adaptive LR) | 99.19% | 79.75% | 81.15% | 79.75% | 79.77% |
| **SVM** | Scikit-learn SVC (StandardScaler + RBF) | 99.45% | 79.12% | 81.54% | 79.12% | 79.68% |

> **Key Finding**: The **1D CNN** achieves the highest testing accuracy (**82.63%**), confirming that capturing temporal shifts and intonation over time yields superior accuracy compared to static time-averaged features.

---

## 3. Technology Stack

### Backend
- **Python 3.12**
- **FastAPI**: REST API with asynchronous endpoints and CORS
- **Uvicorn**: Programmatic ASGI server execution
- **Fluxbase Cloud Database**: Managed PostgreSQL via REST API (`/api/execute-sql`)
- **Librosa & SoundFile**: Acoustic preprocessing, resampling, and feature extraction
- **TensorFlow / Keras**: 1D Convolutional Neural Network
- **Scikit-learn**: Support Vector Machine, Multi-Layer Perceptron, and Label Encoding
- **Matplotlib & Seaborn**: Diagnostic plots and confusion matrix generation

### Frontend
- **Next.js 14** (App Router)
- **React 18** & **TypeScript**
- **Tailwind CSS**: Modern responsive interface
- **Recharts**: Interactive probability bar charts and distribution graphs
- **Lucide React**: Clean iconography

---

## 4. Folder Structure

```text
speech/
│
├── frontend/
│   ├── app/
│   │   ├── layout.tsx              # Root app layout (Navbar + Sidebar)
│   │   ├── page.tsx                # Home / Dashboard route
│   │   ├── globals.css             # Tailwind & theme styles
│   │   ├── dashboard/page.tsx      # Overview metrics & stats
│   │   ├── predict/page.tsx        # Upload, record & emotion prediction
│   │   ├── analytics/page.tsx      # Cross-model evaluation & curves
│   │   └── history/page.tsx        # SQLite prediction history table
│   ├── components/
│   │   ├── Navbar.tsx              # Header with live backend health
│   │   ├── Sidebar.tsx             # Navigation drawer
│   │   ├── AudioUploader.tsx       # Drag-and-drop file upload & playback
│   │   ├── AudioRecorder.tsx       # Live microphone recorder
│   │   ├── EmotionCard.tsx         # Predicted emotion badge & gauge
│   │   ├── ConfidenceChart.tsx     # Recharts emotion probability chart
│   │   ├── Waveform.tsx            # SVG acoustic waveform visualizer
│   │   └── ModelComparison.tsx     # 3-model side-by-side comparison
│   ├── lib/
│   │   └── api.ts                  # Centralized API client
│   ├── package.json
│   └── .env.local
│
├── backend/
│   ├── datasets/
│   │   ├── RAVDESS/                # 1,440 recordings (Actors 01-24)
│   │   └── TESS/                   # 2,800 recordings (OAF & YAF)
│   ├── models/
│   │   ├── svm_model.pkl           # Trained SVM pipeline
│   │   ├── mlp_model.pkl           # Trained MLP pipeline
│   │   ├── cnn_model.keras         # Checkpointed 1D CNN model
│   │   ├── cnn_scaler.npz          # Feature normalization tensors
│   │   ├── label_encoder.pkl       # Fitted 8-class label encoder
│   │   ├── evaluation_metrics.json # Full evaluation results
│   │   └── *.png                   # Confusion matrices & loss curves
│   ├── training/
│   │   ├── download_datasets.py    # Dataset acquisition script
│   │   ├── dataset_loader.py       # Audio validation & speaker split
│   │   ├── preprocessing.py        # Librosa audio conditioning
│   │   ├── feature_extraction.py   # MFCC, Chroma, Mel extraction
│   │   ├── train_svm.py            # SVM trainer
│   │   ├── train_mlp.py            # MLP trainer
│   │   ├── train_cnn.py            # 1D CNN trainer
│   │   ├── evaluate_models.py      # Cross-model evaluator
│   │   └── visualize_results.py    # Plot & matrix generator
│   ├── services/
│   │   ├── prediction_service.py   # Model inference engine
│   │   ├── audio_service.py        # Validation & waveform downsampling
│   │   └── history_service.py      # SQLite CRUD operations
│   ├── speech_emotion.db           # SQLite database
│   ├── config.py                   # Central hyperparameters
│   ├── requirements.txt            # Python dependencies
│   └── main.py                     # FastAPI application & Uvicorn runner
│
└── README.md
```

---

## 5. Execution Instructions

### Step 1: Start the Backend

Open a terminal:
```bash
cd backend
pip install -r requirements.txt
python main.py
```
- The backend starts programmatically at: `http://localhost:8000`
- Interactive Swagger API Documentation: `http://localhost:8000/docs`

### Step 2: Start the Frontend

Open a second terminal:
```bash
cd frontend
npm install
npm run dev
```
- Open your browser at: `http://localhost:3000`

---

## 6. REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API status and service metadata |
| `GET` | `/health` | Health check verifying model and SQLite status |
| `GET` | `/models` | Returns available models (1D CNN, MLP, SVM) |
| `POST` | `/predict` | Predicts emotion from uploaded audio file |
| `POST` | `/predict-all` | Evaluates audio simultaneously across SVM, MLP, and CNN |
| `POST` | `/recording` | Dedicated endpoint for live microphone speech |
| `GET` | `/analytics` | Serves evaluation metrics and static plot URLs |
| `GET` | `/history` | Filterable and paginated prediction history |
| `DELETE` | `/history/{id}` | Deletes a specific prediction record from SQLite |
| `POST` | `/auth/register` | Registers new user, sends welcome email with project logo |
| `POST` | `/auth/login` | Authenticates user, sends security login alert with logo |
| `GET` | `/auth/me` | Retrieves profile of currently authenticated user |
| `POST` | `/auth/logout` | Client-side logout confirmation |
| `GET` | `/auth/smtp-status` | Verifies SMTP server configuration and logo file |
| `POST` | `/auth/test-email` | Dispatches test email embedding the project logo |
| `GET` | `/fluxbase-status` | Verifies Fluxbase Cloud Database configuration & health |

---

## 7. Fluxbase Cloud Database Integration

This project integrates with **Fluxbase** as its primary cloud relational database, replacing all local database files in strict compliance with the **Fluxbase Backend Integration Guide v4.0**.

### Architecture & API Specifications:
- **API Endpoint**: `POST https://fluxbase.vercel.app/api/execute-sql`
- **Authentication**: `Bearer <FLUXBASE_API_KEY>`
- **Request Format**: `{"projectId": "YOUR_PROJECT_ID", "query": "SQL STATEMENT"}` *(uses `query`, not `sql`)*
- **Response Format**: `data.result.rows` *(accesses `result.rows`, not `data`)*
- **Managed Tables**:
  - `users`: Stores user authentication records, bcrypt password hashes, and timestamps.
  - `prediction_history`: Stores acoustic audio predictions, confidence scores, and waveforms.

### How to Configure Fluxbase in `.env`:
Open `backend/.env` (or root `.env`) and add your Fluxbase credentials:
```env
FLUXBASE_URL=https://fluxbase.vercel.app/api/execute-sql
FLUXBASE_PROJECT_ID=your_fluxbase_project_id
FLUXBASE_API_KEY=your_fluxbase_api_key
```

> **Where to get your credentials**:
> 1. Log in to [Fluxbase Dashboard](https://fluxbase.vercel.app).
> 2. Go to **Settings** ➔ Copy your **Project ID**.
> 3. Go to **Settings** ➔ **API Keys** ➔ **Create Key** ➔ Copy your **API Key**.

---

## 8. Authentication & SMTP Email Notifications

The application features full user authentication backed by bcrypt password hashing and JWT tokens, coupled with automated real-time SMTP notifications embedding the official **EmotiSound AI Neon Logo** located at `images/EmotiSound AI Neon Logo.png`.

### SMTP Configuration in `.env`:
Paste your SMTP credentials in `backend/.env` (or root `.env`):
```env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
SMTP_FROM_EMAIL=your_email@gmail.com
SMTP_FROM_NAME=EmotiSound AI - Speech Emotion Recognition
SMTP_USE_TLS=True
SMTP_USE_SSL=False
```

> **Gmail Tip**: When using Gmail, generate an **App Password** from your Google Account:
> `Security` ➔ `2-Step Verification` ➔ `App Passwords` ➔ generate password for "Mail".

### Email Notification Features:
- **Welcome Notification**: Sent on `/auth/register` with embedded project logo (`cid:project_logo`), account verification details, and active model performance summary.
- **Login Security Alert**: Sent on `/auth/login` with timestamp, client IP, device user agent, and quick access link.
- **Embedded Project Logo**: Inline MIME image attached directly from `images/EmotiSound AI Neon Logo.png` so images render across Gmail, Outlook, Apple Mail, and mobile clients.
- **Test Dispatch Tool**: Test email delivery directly from the dashboard UI or Swagger docs.

---

## 8. Ethical Considerations & Acoustic Scope

Speech emotion recognition systems classify **vocal acoustic features** (fundamental frequency, cadence, harmonic energy, and timbre). These predictions represent **acoustic emotion estimates**, not verified psychological states or internal mental experiences. The application displays this disclaimer across all prediction views.

