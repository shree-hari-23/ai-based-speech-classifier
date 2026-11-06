"""
Multi-Layer Perceptron (MLP) Training Module for Speech Emotion Recognition.
Trains an MLPClassifier with exact required specifications:
  - alpha = 0.01
  - batch_size = 256
  - hidden_layer_sizes = (300,)
  - learning_rate = 'adaptive'
  - max_iter = 500
Uses StandardScaler inside a scikit-learn Pipeline.
Evaluates model on held-out test set and saves the serialized model.
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
from sklearn.neural_network import MLPClassifier
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
from config import MLP_MODEL_PATH, MODELS_DIR, LABEL_ENCODER_PATH
from training.feature_extraction import load_cached_features, prepare_dataset_features
from training.dataset_loader import load_combined_dataset, speaker_independent_train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def train_mlp() -> Dict[str, Any]:
    """
    Trains the Multi-Layer Perceptron (MLP) Classifier on the 180-dimensional acoustic features:
      1. Loads train/test features and label encoder.
      2. Configures StandardScaler and MLPClassifier with mandatory specifications.
      3. Trains the neural network pipeline.
      4. Evaluates training and testing metrics (Accuracy, Precision, Recall, F1).
      5. Computes confusion matrix and classification report.
      6. Serializes model to backend/models/mlp_model.pkl.
    """
    logger.info("Loading cached features for MLP training...")
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

    # Build Pipeline with mandatory hyperparameters
    logger.info("Configuring MLP Classifier with mandatory specifications...")
    mlp_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            alpha=0.01,
            batch_size=256,
            hidden_layer_sizes=(300,),
            learning_rate="adaptive",
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=25,
            random_state=42,
            verbose=False
        ))
    ])

    logger.info("Fitting MLP pipeline on training data...")
    mlp_pipeline.fit(X_train, y_train)

    # Predictions
    y_train_pred = mlp_pipeline.predict(X_train)
    y_test_pred = mlp_pipeline.predict(X_test)

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
        "model": "MLP",
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
    joblib.dump(mlp_pipeline, MLP_MODEL_PATH)
    logger.info(f"[SUCCESS] Trained MLP pipeline saved to: {MLP_MODEL_PATH}")

    # Save metrics JSON
    metrics_path = MODELS_DIR / "mlp_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Display evaluation summary
    print("\n" + "=" * 65)
    print("      MULTI-LAYER PERCEPTRON (MLP) - EVALUATION RESULTS")
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
    train_mlp()

