"""
Model Evaluation and Comparison Module for Speech Emotion Recognition.
Loads trained SVM, MLP, and CNN models, evaluates their performance on the held-out test set,
computes metrics (Accuracy, Precision, Recall, F1, Confusion Matrices),
and exports the unified evaluation summary to backend/models/evaluation_metrics.json.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    MODELS_DIR,
    SVM_MODEL_PATH,
    MLP_MODEL_PATH,
    CNN_MODEL_PATH,
    LABEL_ENCODER_PATH,
    METRICS_PATH,
    SUPPORTED_EMOTIONS
)
from training.feature_extraction import load_cached_features

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def evaluate_all_models() -> Dict[str, Any]:
    """
    Evaluates SVM, MLP, and CNN models on identical test data.
    Generates comparison tables, classification reports, and saves evaluation_metrics.json.
    """
    logger.info("Loading test dataset and encoders...")
    data = load_cached_features()
    if data is None:
        raise FileNotFoundError("Feature cache not found. Please run feature_extraction.py first.")

    X_train_ml = data["X_train_ml"]
    y_train = data["y_train"]
    X_test_ml = data["X_test_ml"]
    y_test = data["y_test"]

    X_train_cnn = data["X_train_cnn"]
    X_test_cnn = data["X_test_cnn"]

    encoder = data["encoder"] or joblib.load(LABEL_ENCODER_PATH)
    class_names = [str(c) for c in encoder.classes_]

    # Verify models exist
    for p in [SVM_MODEL_PATH, MLP_MODEL_PATH, CNN_MODEL_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}. Please train all models first.")

    # 1. Evaluate SVM
    logger.info("Evaluating SVM Model...")
    svm_model = joblib.load(SVM_MODEL_PATH)
    y_train_svm = svm_model.predict(X_train_ml)
    y_test_svm = svm_model.predict(X_test_ml)

    svm_train_acc = float(accuracy_score(y_train, y_train_svm))
    svm_test_acc = float(accuracy_score(y_test, y_test_svm))
    svm_precision = float(precision_score(y_test, y_test_svm, average="weighted", zero_division=0))
    svm_recall = float(recall_score(y_test, y_test_svm, average="weighted", zero_division=0))
    svm_f1 = float(f1_score(y_test, y_test_svm, average="weighted", zero_division=0))
    svm_cm = confusion_matrix(y_test, y_test_svm).tolist()
    svm_report = classification_report(y_test, y_test_svm, target_names=class_names, output_dict=True)

    # 2. Evaluate MLP
    logger.info("Evaluating MLP Model...")
    mlp_model = joblib.load(MLP_MODEL_PATH)
    y_train_mlp = mlp_model.predict(X_train_ml)
    y_test_mlp = mlp_model.predict(X_test_ml)

    mlp_train_acc = float(accuracy_score(y_train, y_train_mlp))
    mlp_test_acc = float(accuracy_score(y_test, y_test_mlp))
    mlp_precision = float(precision_score(y_test, y_test_mlp, average="weighted", zero_division=0))
    mlp_recall = float(recall_score(y_test, y_test_mlp, average="weighted", zero_division=0))
    mlp_f1 = float(f1_score(y_test, y_test_mlp, average="weighted", zero_division=0))
    mlp_cm = confusion_matrix(y_test, y_test_mlp).tolist()
    mlp_report = classification_report(y_test, y_test_mlp, target_names=class_names, output_dict=True)

    # 3. Evaluate CNN
    logger.info("Evaluating 1D CNN Model...")
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
    cnn_scaler = np.load(MODELS_DIR / "cnn_scaler.npz")
    feat_mean = cnn_scaler["mean"]
    feat_std = cnn_scaler["std"]

    X_train_cnn_norm = (X_train_cnn - feat_mean) / feat_std
    X_test_cnn_norm = (X_test_cnn - feat_mean) / feat_std

    y_train_cnn_cat = tf.keras.utils.to_categorical(y_train, num_classes=len(class_names))
    y_test_cnn_cat = tf.keras.utils.to_categorical(y_test, num_classes=len(class_names))

    cnn_train_loss, cnn_train_acc = cnn_model.evaluate(X_train_cnn_norm, y_train_cnn_cat, verbose=0)
    cnn_test_loss, cnn_test_acc = cnn_model.evaluate(X_test_cnn_norm, y_test_cnn_cat, verbose=0)

    cnn_probs = cnn_model.predict(X_test_cnn_norm, verbose=0)
    y_test_cnn = np.argmax(cnn_probs, axis=1)

    cnn_precision = float(precision_score(y_test, y_test_cnn, average="weighted", zero_division=0))
    cnn_recall = float(recall_score(y_test, y_test_cnn, average="weighted", zero_division=0))
    cnn_f1 = float(f1_score(y_test, y_test_cnn, average="weighted", zero_division=0))
    cnn_cm = confusion_matrix(y_test, y_test_cnn).tolist()
    cnn_report = classification_report(y_test, y_test_cnn, target_names=class_names, output_dict=True)

    # Load CNN history if available
    cnn_history_file = MODELS_DIR / "cnn_history.json"
    cnn_history = {}
    if cnn_history_file.exists():
        with open(cnn_history_file, "r") as f:
            cnn_history = json.load(f)

    # Determine best model
    model_scores = {
        "SVM": svm_test_acc,
        "MLP": mlp_test_acc,
        "CNN": float(cnn_test_acc)
    }
    best_model_name = max(model_scores, key=model_scores.get)

    # Assemble unified evaluation metrics dictionary
    evaluation_results = {
        "classes": class_names,
        "test_sample_count": int(len(y_test)),
        "train_sample_count": int(len(y_train)),
        "best_model": best_model_name,
        "best_accuracy": round(model_scores[best_model_name] * 100, 2),
        "comparison_table": [
            {
                "model": "SVM",
                "type": "Traditional Machine Learning",
                "train_accuracy": round(svm_train_acc * 100, 2),
                "test_accuracy": round(svm_test_acc * 100, 2),
                "precision": round(svm_precision * 100, 2),
                "recall": round(svm_recall * 100, 2),
                "f1_score": round(svm_f1 * 100, 2)
            },
            {
                "model": "MLP",
                "type": "Traditional Machine Learning",
                "train_accuracy": round(mlp_train_acc * 100, 2),
                "test_accuracy": round(mlp_test_acc * 100, 2),
                "precision": round(mlp_precision * 100, 2),
                "recall": round(mlp_recall * 100, 2),
                "f1_score": round(mlp_f1 * 100, 2)
            },
            {
                "model": "CNN",
                "type": "Deep Learning (1D Conv)",
                "train_accuracy": round(float(cnn_train_acc) * 100, 2),
                "test_accuracy": round(float(cnn_test_acc) * 100, 2),
                "precision": round(cnn_precision * 100, 2),
                "recall": round(cnn_recall * 100, 2),
                "f1_score": round(cnn_f1 * 100, 2)
            }
        ],
        "models": {
            "SVM": {
                "train_accuracy": round(svm_train_acc * 100, 2),
                "test_accuracy": round(svm_test_acc * 100, 2),
                "precision": round(svm_precision * 100, 2),
                "recall": round(svm_recall * 100, 2),
                "f1_score": round(svm_f1 * 100, 2),
                "confusion_matrix": svm_cm,
                "classification_report": svm_report
            },
            "MLP": {
                "train_accuracy": round(mlp_train_acc * 100, 2),
                "test_accuracy": round(mlp_test_acc * 100, 2),
                "precision": round(mlp_precision * 100, 2),
                "recall": round(mlp_recall * 100, 2),
                "f1_score": round(mlp_f1 * 100, 2),
                "confusion_matrix": mlp_cm,
                "classification_report": mlp_report
            },
            "CNN": {
                "train_accuracy": round(float(cnn_train_acc) * 100, 2),
                "test_accuracy": round(float(cnn_test_acc) * 100, 2),
                "precision": round(cnn_precision * 100, 2),
                "recall": round(cnn_recall * 100, 2),
                "f1_score": round(cnn_f1 * 100, 2),
                "test_loss": round(float(cnn_test_loss), 4),
                "train_loss": round(float(cnn_train_loss), 4),
                "confusion_matrix": cnn_cm,
                "classification_report": cnn_report,
                "history": cnn_history
            }
        }
    }

    # Save to JSON
    with open(METRICS_PATH, "w") as f:
        json.dump(evaluation_results, f, indent=2)
    logger.info(f"[SUCCESS] Unified evaluation metrics saved to: {METRICS_PATH}")

    # Print Table
    print("\n" + "=" * 80)
    print("           ACOUSTIC SPEECH EMOTION RECOGNITION - MODEL COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Model':<8} | {'Type':<26} | {'Test Acc':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9}")
    print("-" * 80)
    for row in evaluation_results["comparison_table"]:
        print(f"{row['model']:<8} | {row['type']:<26} | {row['test_accuracy']:>7.2f}% | {row['precision']:>7.2f}% | {row['recall']:>7.2f}% | {row['f1_score']:>7.2f}%")
    print("=" * 80)
    print(f"Best Performing Model: {best_model_name} with {evaluation_results['best_accuracy']}% Test Accuracy\n")

    return evaluation_results


if __name__ == "__main__":
    evaluate_all_models()

