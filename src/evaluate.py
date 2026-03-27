"""
evaluate.py
-----------
Metric computation and plotting helpers for AML model evaluation.
"""

from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _get_scores(model, X_test: np.ndarray) -> np.ndarray:
    """Return a continuous anomaly/probability score for any model type."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        # Isolation Forest / LOF: higher score = more normal → flip sign
        return -scores
    raise ValueError(f"Model {type(model)} has no predict_proba or decision_function.")


def _threshold_predictions(model, X_test: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Binary predictions via threshold on continuous scores (for unsupervised)."""
    if hasattr(model, "predict_proba"):
        return (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    # Unsupervised: -1 → 1 (anomaly), 1 → 0 (inlier)
    raw = model.predict(X_test)
    return (raw == -1).astype(int)


def evaluate_supervised(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    name: str,
) -> Dict:
    """Evaluate a supervised classifier.

    Returns
    -------
    dict with keys: name, precision, recall, f1, roc_auc, pr_auc
    """
    scores = _get_scores(model, X_test)
    preds = (scores >= 0.5).astype(int)
    return {
        "name": name,
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, scores),
        "pr_auc": average_precision_score(y_test, scores),
    }


def evaluate_unsupervised(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    name: str,
) -> Dict:
    """Evaluate an unsupervised anomaly detector.

    Anomaly score is inverted (higher = more anomalous) so PR-AUC is meaningful.

    Returns
    -------
    dict with keys: name, precision, recall, f1, roc_auc, pr_auc
    """
    scores = _get_scores(model, X_test)
    preds = _threshold_predictions(model, X_test)
    return {
        "name": name,
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, scores),
        "pr_auc": average_precision_score(y_test, scores),
    }


def summary_table(results: List[Dict]) -> pd.DataFrame:
    """Build a formatted summary DataFrame from a list of evaluation dicts."""
    df = pd.DataFrame(results).set_index("name")
    df = df[["precision", "recall", "f1", "roc_auc", "pr_auc"]]
    df.columns = ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    return df.round(4).sort_values("PR-AUC", ascending=False)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_pr_curves(
    models_dict: Dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> plt.Figure:
    """Precision-Recall curves for multiple models on a single axes.

    Parameters
    ----------
    models_dict : {name: fitted_model}
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in models_dict.items():
        scores = _get_scores(model, X_test)
        prec, rec, _ = precision_recall_curve(y_test, scores)
        ap = average_precision_score(y_test, scores)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_roc_curves(
    models_dict: Dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> plt.Figure:
    """ROC curves for multiple models on a single axes."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    for name, model in models_dict.items():
        scores = _get_scores(model, X_test)
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc_val = roc_auc_score(y_test, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_confusion_matrices(
    models_dict: Dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> plt.Figure:
    """Grid of confusion matrices (one per model)."""
    n = len(models_dict)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for ax, (name, model) in zip(axes, models_dict.items()):
        preds = _threshold_predictions(model, X_test)
        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Clean", "Laundering"])
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(name)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Confusion Matrices", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig
