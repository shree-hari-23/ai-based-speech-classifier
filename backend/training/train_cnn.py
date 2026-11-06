"""
Deep Learning Model (1D CNN) Training Module for Speech Emotion Recognition.
Implements the exact mandatory TensorFlow/Keras Conv1D architecture:
  Input Layer
  ↓
  Conv1D – 64 filters, ReLU
  ↓
  Dropout
  ↓
  MaxPooling1D
  ↓
  Conv1D – 128 filters, ReLU
  ↓
  Dropout
  ↓
  MaxPooling1D
  ↓
  Conv1D – 256 filters, ReLU
  ↓
  Dropout
  ↓
  Flatten
  ↓
  Dense – 8 neurons, Softmax

Uses categorical cross-entropy, Adam optimizer, EarlyStopping, and ModelCheckpoint.
Generates training vs validation accuracy and loss curves.
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
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
    CNN_MODEL_PATH,
    MODELS_DIR,
    LABEL_ENCODER_PATH,
    SUPPORTED_EMOTIONS
)
from training.feature_extraction import load_cached_features, prepare_dataset_features
from training.dataset_loader import load_combined_dataset, speaker_independent_train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def build_cnn_model(input_shape: Tuple[int, int] = (130, 180), num_classes: int = 8) -> tf.keras.Model:
    """
    Builds the 1D CNN model matching the exact architectural specification:
      Input Layer -> Conv1D(64) -> ReLU -> Dropout -> MaxPooling1D
                  -> Conv1D(128) -> ReLU -> Dropout -> MaxPooling1D
                  -> Conv1D(256) -> ReLU -> Dropout
                  -> Flatten -> Dense(8, Softmax)
    """
    model = models.Sequential([
        # Input Layer
        layers.Input(shape=input_shape, name="audio_sequence_input"),

        # Conv Block 1
        layers.Conv1D(filters=64, kernel_size=5, padding="same", activation="relu", name="conv1d_64"),
        layers.Dropout(0.25, name="dropout_1"),
        layers.MaxPooling1D(pool_size=2, name="maxpool_1"),

        # Conv Block 2
        layers.Conv1D(filters=128, kernel_size=5, padding="same", activation="relu", name="conv1d_128"),
        layers.Dropout(0.3, name="dropout_2"),
        layers.MaxPooling1D(pool_size=2, name="maxpool_2"),

        # Conv Block 3
        layers.Conv1D(filters=256, kernel_size=5, padding="same", activation="relu", name="conv1d_256"),
        layers.Dropout(0.3, name="dropout_3"),

        # Output Head
        layers.Flatten(name="flatten"),
        layers.Dense(num_classes, activation="softmax", name="emotion_softmax")
    ], name="Speech_Emotion_CNN")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def plot_training_curves(history: tf.keras.callbacks.History, save_dir: Path = MODELS_DIR):
    """
    Plots training & validation accuracy and loss graphs as required:
      1. cnn_training_accuracy.png
      2. cnn_training_loss.png
      3. cnn_training_history.png (combined)
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history.history["accuracy"]) + 1)

    # 1. Accuracy Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history["accuracy"], "b-o", linewidth=2, label="Training Accuracy")
    plt.plot(epochs, history.history["val_accuracy"], "g--s", linewidth=2, label="Validation Accuracy")
    plt.title("CNN Training and Validation Accuracy", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    acc_path = save_dir / "cnn_accuracy.png"
    plt.savefig(acc_path, dpi=200)
    plt.close()
    logger.info(f"Accuracy plot saved to: {acc_path}")

    # 2. Loss Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history["loss"], "r-o", linewidth=2, label="Training Loss")
    plt.plot(epochs, history.history["val_loss"], "m--s", linewidth=2, label="Validation Loss")
    plt.title("CNN Training and Validation Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Categorical Crossentropy Loss", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    loss_path = save_dir / "cnn_loss.png"
    plt.savefig(loss_path, dpi=200)
    plt.close()
    logger.info(f"Loss plot saved to: {loss_path}")

    # 3. Combined Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(epochs, history.history["accuracy"], label="Train Acc", color="#2563eb", linewidth=2)
    ax1.plot(epochs, history.history["val_accuracy"], label="Val Acc", color="#16a34a", linewidth=2, linestyle="--")
    ax1.set_title("CNN Accuracy Curve", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(epochs, history.history["loss"], label="Train Loss", color="#dc2626", linewidth=2)
    ax2.plot(epochs, history.history["val_loss"], label="Val Loss", color="#9333ea", linewidth=2, linestyle="--")
    ax2.set_title("CNN Loss Curve", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    combined_path = save_dir / "cnn_training_history.png"
    plt.savefig(combined_path, dpi=200)
    plt.close()


def train_cnn(epochs: int = 50, batch_size: int = 64) -> Dict[str, Any]:
    """
    Full training pipeline for the Deep Learning 1D CNN model.
    """
    logger.info("Loading cached sequential features for CNN training...")
    data = load_cached_features()

    if data is None:
        logger.info("Cache not found. Extracting features from dataset...")
        df = load_combined_dataset()
        train_df, test_df = speaker_independent_train_test_split(df)
        data = prepare_dataset_features(train_df, test_df)

    X_train = data["X_train_cnn"]  # shape: (3440, 130, 180)
    y_train = data["y_train"]      # shape: (3440,)
    X_test = data["X_test_cnn"]    # shape: (800, 130, 180)
    y_test = data["y_test"]        # shape: (800,)
    encoder = data["encoder"] or joblib.load(LABEL_ENCODER_PATH)

    logger.info(f"CNN Train input shape: {X_train.shape}, Test input shape: {X_test.shape}")

    # Standardize sequential features across time and channels
    feat_mean = np.mean(X_train, axis=(0, 1), keepdims=True)
    feat_std = np.std(X_train, axis=(0, 1), keepdims=True) + 1e-7

    X_train_scaled = (X_train - feat_mean) / feat_std
    X_test_scaled = (X_test - feat_mean) / feat_std

    # Save scaler parameters for live inference
    scaler_path = MODELS_DIR / "cnn_scaler.npz"
    np.savez(scaler_path, mean=feat_mean, std=feat_std)
    logger.info(f"CNN sequence normalization parameters saved to: {scaler_path}")

    # One-hot encode targets
    num_classes = len(SUPPORTED_EMOTIONS)
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

    # Build CNN architecture
    model = build_cnn_model(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=num_classes)
    model.summary(print_fn=logger.info)

    # Callbacks: EarlyStopping and ModelCheckpoint saving best model
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    checkpoint = callbacks.ModelCheckpoint(
        filepath=str(CNN_MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-5,
        verbose=1
    )

    logger.info("Beginning CNN training with EarlyStopping and ModelCheckpoint...")
    history = model.fit(
        X_train_scaled,
        y_train_cat,
        validation_data=(X_test_scaled, y_test_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, checkpoint, reduce_lr],
        verbose=1
    )

    # Load best checkpointed model
    if CNN_MODEL_PATH.exists():
        logger.info(f"Loading best checkpoint from {CNN_MODEL_PATH}...")
        model = models.load_model(CNN_MODEL_PATH)
    else:
        model.save(CNN_MODEL_PATH)

    # Plot training curves
    plot_training_curves(history, save_dir=MODELS_DIR)

    # Evaluate model
    train_loss, train_acc = model.evaluate(X_train_scaled, y_train_cat, verbose=0)
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test_cat, verbose=0)

    # Predict test probabilities and classes
    y_test_probs = model.predict(X_test_scaled, verbose=0)
    y_test_pred = np.argmax(y_test_probs, axis=1)

    precision = float(precision_score(y_test, y_test_pred, average="weighted", zero_division=0))
    recall = float(recall_score(y_test, y_test_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_test_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_test, y_test_pred).tolist()

    class_names = [str(c) for c in encoder.classes_]
    clf_report = classification_report(y_test, y_test_pred, target_names=class_names, output_dict=True)

    metrics = {
        "model": "CNN",
        "train_accuracy": round(float(train_acc) * 100, 2),
        "test_accuracy": round(float(test_acc) * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "test_loss": round(float(test_loss), 4),
        "train_loss": round(float(train_loss), 4),
        "confusion_matrix": cm,
        "classes": class_names,
        "classification_report": clf_report
    }

    # Save metrics JSON
    metrics_path = MODELS_DIR / "cnn_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save training history JSON
    history_path = MODELS_DIR / "cnn_history.json"
    with open(history_path, "w") as f:
        json.dump({
            "accuracy": [float(v) for v in history.history["accuracy"]],
            "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
            "loss": [float(v) for v in history.history["loss"]],
            "val_loss": [float(v) for v in history.history["val_loss"]],
        }, f, indent=2)

    # Display evaluation summary
    print("\n" + "=" * 65)
    print("      DEEP LEARNING 1D CNN - EVALUATION RESULTS")
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
    train_cnn(epochs=50, batch_size=64)

