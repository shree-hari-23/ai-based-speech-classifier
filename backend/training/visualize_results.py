"""
Visualization Module for Speech Emotion Recognition.
Generates publication-quality charts and plots:
  1. Confusion Matrix for SVM
  2. Confusion Matrix for MLP
  3. Confusion Matrix for CNN
  4. Combined Confusion Matrices
  5. Model Comparison Bar Chart (Accuracy, Precision, Recall, F1)
  6. Sample Audio Waveform and Mel Spectrogram visualization
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MODELS_DIR, METRICS_PATH, SUPPORTED_EMOTIONS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def plot_single_confusion_matrix(
    cm: List[List[int]],
    classes: List[str],
    model_name: str,
    save_path: Path
):
    """Generates an individual confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    cm_arr = np.array(cm)
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True
    )
    plt.title(f"{model_name} - Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Emotion", fontsize=11)
    plt.ylabel("Actual Emotion", fontsize=11)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.info(f"Saved confusion matrix: {save_path.name}")


def plot_all_confusion_matrices(metrics: Dict[str, Any], save_dir: Path = MODELS_DIR):
    """
    Generates confusion matrices for each individual model, plus a combined 3-panel figure.
    """
    classes = metrics["classes"]

    # 1. Individual plots
    for model_name in ["SVM", "MLP", "CNN"]:
        cm = metrics["models"][model_name]["confusion_matrix"]
        plot_path = save_dir / f"confusion_matrix_{model_name.lower()}.png"
        plot_single_confusion_matrix(cm, classes, model_name, plot_path)

    # 2. Combined 3-panel comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    colormaps = ["Blues", "Greens", "Purples"]

    for idx, model_name in enumerate(["SVM", "MLP", "CNN"]):
        cm_arr = np.array(metrics["models"][model_name]["confusion_matrix"])
        sns.heatmap(
            cm_arr,
            annot=True,
            fmt="d",
            cmap=colormaps[idx],
            xticklabels=classes,
            yticklabels=classes,
            cbar=False,
            ax=axes[idx]
        )
        acc = metrics["models"][model_name]["test_accuracy"]
        axes[idx].set_title(f"{model_name} (Acc: {acc}%)", fontsize=13, fontweight="bold")
        axes[idx].set_xlabel("Predicted Emotion", fontsize=10)
        if idx == 0:
            axes[idx].set_ylabel("Actual Emotion", fontsize=10)
        else:
            axes[idx].set_ylabel("")
        axes[idx].tick_params(axis="x", rotation=45)

    plt.suptitle("Speech Emotion Recognition - Confusion Matrix Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    combined_path = save_dir / "confusion_matrices_comparison.png"
    plt.savefig(combined_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved combined confusion matrices: {combined_path.name}")


def plot_model_comparison_chart(metrics: Dict[str, Any], save_dir: Path = MODELS_DIR):
    """
    Generates a grouped bar chart comparing Accuracy, Precision, Recall, and F1 across all models.
    """
    models = ["SVM", "MLP", "CNN"]
    metric_keys = ["test_accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

    data = {m: [metrics["models"][m][k] for k in metric_keys] for m in models}

    x = np.arange(len(metric_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#3b82f6", "#10b981", "#8b5cf6"]  # Blue, Green, Purple
    for i, m in enumerate(models):
        offset = (i - 1) * width
        rects = ax.bar(x + offset, data[m], width, label=m, color=colors[i], edgecolor="black", alpha=0.85)
        # Add value labels on top of bars
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Performance Metrics Comparison Across Speech Emotion Models", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(60, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)

    plt.tight_layout()
    chart_path = save_dir / "model_comparison_chart.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    logger.info(f"Saved model comparison chart: {chart_path.name}")


def generate_all_visualizations():
    """Generates all required plots and confusion matrices."""
    if not METRICS_PATH.exists():
        logger.info("Evaluation metrics file not found. Running evaluate_models first...")
        from training.evaluate_models import evaluate_all_models
        metrics = evaluate_all_models()
    else:
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)

    logger.info("Generating confusion matrices and comparison charts...")
    plot_all_confusion_matrices(metrics, save_dir=MODELS_DIR)
    plot_model_comparison_chart(metrics, save_dir=MODELS_DIR)
    logger.info("All visualizations generated successfully in backend/models/!")


if __name__ == "__main__":
    generate_all_visualizations()

