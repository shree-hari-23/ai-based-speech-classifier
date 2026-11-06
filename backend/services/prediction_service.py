"""
Prediction Service for Speech Emotion Recognition.
Loads serialized models (SVM, MLP, CNN), runs preprocessing and feature extraction,
and computes calibrated class probabilities and confidence scores.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

import joblib
import numpy as np
import tensorflow as tf

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    SVM_MODEL_PATH,
    MLP_MODEL_PATH,
    CNN_MODEL_PATH,
    LABEL_ENCODER_PATH,
    MODELS_DIR,
    SUPPORTED_EMOTIONS,
    SAMPLE_RATE
)
from training.preprocessing import preprocess_audio
from training.feature_extraction import extract_features_from_audio
from services.audio_service import extract_waveform_points

logger = logging.getLogger(__name__)

DISCLAIMER_TEXT = "Predictions represent acoustic emotion estimates derived from vocal frequency and cadence, not verified psychological states."


class EmotionPredictionEngine:
    """Manages loaded models and executes multi-modal inference."""

    def __init__(self):
        self.svm_pipeline = None
        self.mlp_pipeline = None
        self.cnn_model = None
        self.label_encoder = None
        self.cnn_scaler = None
        self.classes = SUPPORTED_EMOTIONS
        self.models_loaded = False

    def load_models(self):
        """Loads all model weights and encoders into memory."""
        if self.models_loaded:
            return

        # 1. Label Encoder
        if LABEL_ENCODER_PATH.exists():
            self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
            self.classes = [str(c) for c in self.label_encoder.classes_]
            logger.info(f"Loaded LabelEncoder with classes: {self.classes}")
        else:
            logger.warning(f"Label encoder not found at {LABEL_ENCODER_PATH}")

        # 2. SVM Model
        if SVM_MODEL_PATH.exists():
            self.svm_pipeline = joblib.load(SVM_MODEL_PATH)
            logger.info("Loaded SVM pipeline successfully.")
        else:
            logger.warning(f"SVM model not found at {SVM_MODEL_PATH}")

        # 3. MLP Model
        if MLP_MODEL_PATH.exists():
            self.mlp_pipeline = joblib.load(MLP_MODEL_PATH)
            logger.info("Loaded MLP pipeline successfully.")
        else:
            logger.warning(f"MLP model not found at {MLP_MODEL_PATH}")

        # 4. CNN Model & Scaler
        scaler_path = MODELS_DIR / "cnn_scaler.npz"
        if CNN_MODEL_PATH.exists():
            self.cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
            logger.info("Loaded 1D CNN model successfully.")
        else:
            logger.warning(f"CNN model not found at {CNN_MODEL_PATH}")

        if scaler_path.exists():
            self.cnn_scaler = np.load(scaler_path)
            logger.info("Loaded CNN sequence normalization parameters.")
        else:
            logger.warning(f"CNN scaler not found at {scaler_path}")

        self.models_loaded = True

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Returns metadata regarding all currently loaded and available models."""
        self.load_models()
        available = []

        available.append({
            "id": "CNN",
            "name": "1D Convolutional Neural Network",
            "type": "Deep Learning",
            "is_loaded": self.cnn_model is not None,
            "description": "Temporal sequence model using 3 Conv1D blocks with dropout and maxpooling over 130 acoustic frames."
        })
        available.append({
            "id": "SVM",
            "name": "Support Vector Machine",
            "type": "Traditional Machine Learning",
            "is_loaded": self.svm_pipeline is not None,
            "description": "StandardScaler + RBF kernel SVC with Platt probability scaling on 180 acoustic features."
        })
        available.append({
            "id": "MLP",
            "name": "Multi-Layer Perceptron",
            "type": "Traditional Machine Learning",
            "is_loaded": self.mlp_pipeline is not None,
            "description": "Feed-forward neural network with 300 hidden neurons and adaptive learning rate on 180 features."
        })

        return available

    def _predict_svm(self, ml_features: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        if self.svm_pipeline is None:
            raise RuntimeError("SVM model is not loaded. Training required.")

        feat_2d = ml_features.reshape(1, -1)
        probs = self.svm_pipeline.predict_proba(feat_2d)[0]
        pred_idx = int(np.argmax(probs))
        pred_emotion = self.classes[pred_idx]
        confidence = round(float(probs[pred_idx]) * 100, 2)

        probabilities = {
            self.classes[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(self.classes))
        }
        # Sort probabilities descending
        probabilities = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True))

        return pred_emotion, confidence, probabilities

    def _predict_mlp(self, ml_features: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        if self.mlp_pipeline is None:
            raise RuntimeError("MLP model is not loaded. Training required.")

        feat_2d = ml_features.reshape(1, -1)
        probs = self.mlp_pipeline.predict_proba(feat_2d)[0]
        pred_idx = int(np.argmax(probs))
        pred_emotion = self.classes[pred_idx]
        confidence = round(float(probs[pred_idx]) * 100, 2)

        probabilities = {
            self.classes[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(self.classes))
        }
        probabilities = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True))

        return pred_emotion, confidence, probabilities

    def _predict_cnn(self, cnn_sequence: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        if self.cnn_model is None:
            raise RuntimeError("CNN model is not loaded. Training required.")

        seq_3d = cnn_sequence[np.newaxis, ...]  # shape: (1, 130, 180)

        # Standardize using trained mean and std
        if self.cnn_scaler is not None:
            mean = self.cnn_scaler["mean"]
            std = self.cnn_scaler["std"]
            seq_3d = (seq_3d - mean) / std

        probs = self.cnn_model.predict(seq_3d, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_emotion = self.classes[pred_idx]
        confidence = round(float(probs[pred_idx]) * 100, 2)

        probabilities = {
            self.classes[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(self.classes))
        }
        probabilities = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True))

        return pred_emotion, confidence, probabilities

    def predict(
        self,
        audio_source: Union[str, Path, bytes],
        model_name: str = "CNN"
    ) -> Dict[str, Any]:
        """
        Executes single model emotion prediction.
        Preprocesses audio, extracts acoustic features, runs model, and returns response.
        """
        self.load_models()
        model_key = model_name.upper().strip()

        # 1. Preprocess audio
        y = preprocess_audio(audio_source, target_sr=SAMPLE_RATE)
        waveform_points = extract_waveform_points(y, num_points=120)

        # 2. Extract features
        ml_features, cnn_sequence = extract_features_from_audio(y, sr=SAMPLE_RATE)

        # 3. Model inference
        if model_key == "SVM":
            pred_emo, conf, probs = self._predict_svm(ml_features)
        elif model_key == "MLP":
            pred_emo, conf, probs = self._predict_mlp(ml_features)
        elif model_key in ["CNN", "DEEP_LEARNING"]:
            pred_emo, conf, probs = self._predict_cnn(cnn_sequence)
        else:
            raise ValueError(f"Unknown model '{model_name}'. Choose from: SVM, MLP, CNN.")

        return {
            "predicted_emotion": pred_emo,
            "confidence": conf,
            "model": model_key,
            "probabilities": probs,
            "waveform_sample": waveform_points,
            "disclaimer": DISCLAIMER_TEXT
        }

    def predict_all(
        self,
        audio_source: Union[str, Path, bytes]
    ) -> Dict[str, Any]:
        """
        Executes emotion prediction across ALL models simultaneously
        to enable direct side-by-side model comparison.
        """
        self.load_models()

        # 1. Preprocess audio
        y = preprocess_audio(audio_source, target_sr=SAMPLE_RATE)
        waveform_points = extract_waveform_points(y, num_points=120)

        # 2. Extract features
        ml_features, cnn_sequence = extract_features_from_audio(y, sr=SAMPLE_RATE)

        # 3. Run predictions
        svm_emo, svm_conf, svm_probs = self._predict_svm(ml_features)
        mlp_emo, mlp_conf, mlp_probs = self._predict_mlp(ml_features)
        cnn_emo, cnn_conf, cnn_probs = self._predict_cnn(cnn_sequence)

        # Consensus or best prediction (CNN preferred due to highest empirical accuracy)
        return {
            "best_model": "CNN",
            "consensus_emotion": cnn_emo,
            "consensus_confidence": cnn_conf,
            "waveform_sample": waveform_points,
            "models": {
                "CNN": {
                    "model": "CNN",
                    "type": "Deep Learning (1D Conv)",
                    "predicted_emotion": cnn_emo,
                    "confidence": cnn_conf,
                    "probabilities": cnn_probs
                },
                "SVM": {
                    "model": "SVM",
                    "type": "Traditional Machine Learning",
                    "predicted_emotion": svm_emo,
                    "confidence": svm_conf,
                    "probabilities": svm_probs
                },
                "MLP": {
                    "model": "MLP",
                    "type": "Traditional Machine Learning",
                    "predicted_emotion": mlp_emo,
                    "confidence": mlp_conf,
                    "probabilities": mlp_probs
                }
            },
            "disclaimer": DISCLAIMER_TEXT
        }


# Singleton engine instance
engine = EmotionPredictionEngine()
