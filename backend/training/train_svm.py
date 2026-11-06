"""
Support Vector Machine (SVM) Training Module for Speech Emotion Recognition.
Trains an RBF-kernel SVC inside a scikit-learn Pipeline with StandardScaler.
Enables Platt scaling (probability=True) for confidence estimation.
Evaluates model on held-out test set and saves the serialized pipeline.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import GridSearchCV

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import SVM_MODEL_PATH, MODELS_DIR, LABEL_ENCODER_PATH
from training.feature_extraction import load_cached_features, prepare_dataset_features
from training.dataset_loader import load_combined_dataset, speaker_independent_train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def train_svm(tune_hyperparameters: bool = False) -> Dict[str, Any]:
    """
    Trains the SVM Classifier on the 180-dimensional acoustic feature vectors:
      1. Loads train/test features and label encoder.
      2. Constructs Pipeline([('scaler', StandardScaler()), ('svm', SVC(...))]).
      3. Trains the pipeline (optionally runs GridSearchCV for C and gamma).
      4. Evaluates training and testing metrics (Accuracy, Precision, Recall, F1).
      5. Computes confusion matrix and detailed classification report.
      6. Serializes model to backend/models/svm_model.pkl.
    """
    logger.info("Loading cached features for SVM training...")
    data = load_cached_features()

    if data is None:
        logger.info("Cache not found. Extracting features from dataset...")
        df = load_combined_dataset()
        train_df, test_df = speaker_independent_train_test_split(df)
        data = prepare_dataset_features(train_df, test_df)

    X_train = data["X_train_ml"]
    y_train = data["y_train"]
    X_test = data["X_test_ml"]
    y_test = data["y_test"]
    encoder = data["encoder"] or joblib.load(LABEL_ENCODER_PATH)

    logger.info(f"Loaded training samples: {X_train.shape}, testing samples: {X_test.shape}")

    if tune_hyperparameters:
        logger.info("Running Hyperparameter Grid Search (C and gamma)...")
        base_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True, random_state=42))
        ])
        param_grid = {
            "svm__C": [1.0, 5.0, 10.0],
            "svm__gamma": ["scale", "auto", 0.01]
        }
        grid = GridSearchCV(base_pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        best_pipeline = grid.best_estimator_
        logger.info(f"Best SVM Parameters: {grid.best_params_}")
    else:
        # High-performing defaults for speech acoustic features
        logger.info("Configuring SVM Pipeline with StandardScaler and RBF kernel...")
        best_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                C=10.0,
                kernel="rbf",
                gamma="scale",
                probability=True,  # Enables probability estimation for prediction confidence
                random_state=42
            ))
        ])
        logger.info("Fitting SVM pipeline on training data...")
        best_pipeline.fit(X_train, y_train)

    # Predictions
    y_train_pred = best_pipeline.predict(X_train)
    y_test_pred = best_pipeline.predict(X_test)

    # Metrics computation
    train_acc = float(accuracy_score(y_train, y_train_pred))
    test_acc = float(accuracy_score(y_test, y_test_pred))
    precision = float(precision_score(y_test, y_test_pred, average="weighted", zero_division=0))
    recall = float(recall_score(y_test, y_test_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_test_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_test, y_test_pred).tolist()

    class_names = [str(c) for c in encoder.classes_]
    clf_report = classification_report(y_test, y_test_pred, target_names=class_names, output_dict=True)

    metrics = {
        "model": "SVM",
        "train_accuracy": round(train_acc * 100, 2),
        "test_accuracy": round(test_acc * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "confusion_matrix": cm,
        "classes": class_names,
        "classification_report": clf_report
    }

    # Save trained pipeline
    joblib.dump(best_pipeline, SVM_MODEL_PATH)
    logger.info(f"[SUCCESS] Trained SVM pipeline saved to: {SVM_MODEL_PATH}")

    # Save metrics JSON
    metrics_path = MODELS_DIR / "svm_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Display evaluation summary
    print("\n" + "=" * 65)
    print("      SUPPORT VECTOR MACHINE (SVM) - EVALUATION RESULTS")
    print("=" * 65)
    print(f"Training Accuracy:   {metrics['train_accuracy']}%")
    print(f"Testing Accuracy:    {metrics['test_accuracy']}%")
    print(f"Weighted Precision:  {metrics['precision']}%")
    print(f"Weighted Recall:     {metrics['recall']}%")
    print(f"Weighted F1-Score:   {metrics['f1_score']}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=class_names))
    print("=" * 65)

    return metrics


if __name__ == "__main__":
    train_svm(tune_hyperparameters=False)

