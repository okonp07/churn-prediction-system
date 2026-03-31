from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from src.models.target_manager import TargetManager

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - exercised indirectly when dependency is absent
    plt = None


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    min_rating = int(min(np.min(y_true), np.min(y_pred)))
    max_rating = int(max(np.max(y_true), np.max(y_pred)))
    num_ratings = max_rating - min_rating + 1
    confusion = confusion_matrix(y_true, y_pred, labels=np.arange(min_rating, max_rating + 1))
    num_scored_items = float(len(y_true))

    hist_true = np.bincount(y_true - min_rating, minlength=num_ratings)
    hist_pred = np.bincount(y_pred - min_rating, minlength=num_ratings)

    expected = np.outer(hist_true, hist_pred) / num_scored_items
    weights = np.zeros((num_ratings, num_ratings), dtype=float)
    for i in range(num_ratings):
        for j in range(num_ratings):
            weights[i, j] = ((i - j) ** 2) / ((num_ratings - 1) ** 2 if num_ratings > 1 else 1)

    observed = (weights * confusion).sum() / num_scored_items
    expected_score = (weights * expected).sum() / num_scored_items
    return float(1.0 - observed / expected_score) if expected_score else 1.0


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    target_manager: TargetManager,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if target_manager.task_type == "ordinal_multiclass_classification":
        metrics["quadratic_weighted_kappa"] = quadratic_weighted_kappa(y_true, y_pred)
        metrics["ordinal_mae"] = float(mean_absolute_error(y_true, y_pred))

    if y_proba is not None:
        try:
            metrics["log_loss"] = float(log_loss(y_true, y_proba))
        except ValueError:
            metrics["log_loss"] = None

        if len(np.unique(y_true)) == 2:
            positive_proba = y_proba[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_true, positive_proba))
            metrics["pr_auc"] = float(average_precision_score(y_true, positive_proba))
        else:
            classes = np.arange(y_proba.shape[1])
            y_true_bin = label_binarize(y_true, classes=classes)
            metrics["roc_auc_ovr_weighted"] = float(
                roc_auc_score(y_true_bin, y_proba, average="weighted", multi_class="ovr")
            )
            metrics["pr_auc_ovr_weighted"] = float(
                average_precision_score(y_true_bin, y_proba, average="weighted")
            )
    return metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    target_manager: TargetManager,
) -> dict[str, Any]:
    if target_manager.task_type == "regression":
        return regression_metrics(y_true, y_pred)
    return classification_metrics(y_true, y_pred, y_proba, target_manager)


def plot_confusion_matrix(
    matrix: list[list[int]],
    class_labels: list[str],
    output_path: str | Path,
) -> None:
    if plt is None:
        return
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(class_labels)))
    axis.set_yticks(range(len(class_labels)))
    axis.set_xticklabels(class_labels, rotation=45, ha="right")
    axis.set_yticklabels(class_labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("Validation Confusion Matrix")

    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            axis.text(j, i, str(value), ha="center", va="center", color="black")

    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_target_distribution(target: pd.Series, output_path: str | Path) -> None:
    if plt is None:
        return
    figure, axis = plt.subplots(figsize=(8, 4))
    target.value_counts().sort_index().plot(kind="bar", color="#28536B", ax=axis)
    axis.set_title("Target Distribution")
    axis.set_xlabel("Churn Risk Score")
    axis.set_ylabel("Count")
    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_calibration_curve(
    y_true_binary: np.ndarray,
    y_score: np.ndarray,
    output_path: str | Path,
) -> None:
    if plt is None:
        return
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true_binary, y_score, n_bins=10, strategy="uniform"
    )
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot(mean_predicted_value, fraction_of_positives, marker="o", label="Model")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    axis.set_xlabel("Mean Predicted Probability")
    axis.set_ylabel("Fraction of Positives")
    axis.set_title("Calibration Curve for High-Risk Probability")
    axis.legend()
    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_json(payload: dict[str, Any], output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)
