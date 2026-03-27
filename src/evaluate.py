# evaluate.py
# metric computation and plotting helpers for comparing AML models

from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _get_scores(model, X_test: np.ndarray) -> np.ndarray:
    """Get continuous anomaly/probability scores from any model type."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        # IForest/LOF: higher score = more normal, so flip the sign
        return -scores
    raise ValueError(f"Model {type(model)} has no predict_proba or decision_function.")


def _threshold_predictions(model, X_test: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert model output to binary predictions."""
    if hasattr(model, "predict_proba"):
        return (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    # unsupervised: predict() returns -1 for anomaly, 1 for inlier
    raw = model.predict(X_test)
    return (raw == -1).astype(int)


def evaluate_supervised(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    name: str,
) -> Dict:
    """Compute precision/recall/F1/ROC-AUC/PR-AUC for a supervised model."""
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
    """Same metrics as evaluate_supervised but works with anomaly detectors.

    Anomaly score is inverted so higher = more suspicious (needed for PR-AUC to be meaningful).
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
    """Turn a list of evaluate_* dicts into a sorted comparison table."""
    df = pd.DataFrame(results).set_index("name")
    df = df[["precision", "recall", "f1", "roc_auc", "pr_auc"]]
    df.columns = ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    return df.round(4).sort_values("PR-AUC", ascending=False)


def plot_pr_curves(
    models_dict: Dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> plt.Figure:
    """Plot PR curves for a dict of {name: fitted_model}."""
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
    """Plot ROC curves for a dict of {name: fitted_model}."""
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
    """Grid of confusion matrices, one per model."""
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


_PALETTE = ["#e74c3c", "#f39c12", "#3498db", "#9b59b6", "#2ecc71"]


def plot_pr_roc_curves(
    scores_dict: Dict[str, np.ndarray],
    y_true: np.ndarray,
) -> plt.Figure:
    """Side-by-side PR and ROC curves from raw score arrays.

    scores_dict maps model name -> anomaly score array (higher = more anomalous).
    Useful for unsupervised models where we already have the scores.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax_idx, (curve_fn, xlabel, ylabel, title) in enumerate([
        (precision_recall_curve, "Recall",  "Precision", "Precision-Recall"),
        (roc_curve,              "FPR",     "TPR",       "ROC"),
    ]):
        ax = axes[ax_idx]
        for (name, scores), color in zip(scores_dict.items(), _PALETTE):
            if curve_fn == precision_recall_curve:
                p, r, _ = curve_fn(y_true, scores)
                ap = average_precision_score(y_true, scores)
                ax.plot(r, p, color=color, linewidth=2.5, label=f"{name} (AP={ap:.3f})")
            else:
                fpr, tpr, _ = curve_fn(y_true, scores)
                a = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=color, linewidth=2.5, label=f"{name} (AUC={a:.3f})")
        if curve_fn == roc_curve:
            ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_title(f"{title} Curve", fontsize=15, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=10)

    fig.tight_layout()
    return fig


def plot_pca_projection(
    X_scaled: np.ndarray,
    y_true: np.ndarray,
    ensemble_scores: np.ndarray,
    random_state: int = 42,
) -> plt.Figure:
    """2-panel PCA scatter: true labels (left) vs ensemble score heatmap (right)."""
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    ax.scatter(X_pca[y_true == 0, 0], X_pca[y_true == 0, 1],
               c="#2ecc71", alpha=0.1, s=3, label="Clean")
    ax.scatter(X_pca[y_true == 1, 0], X_pca[y_true == 1, 1],
               c="#e74c3c", alpha=0.4, s=8, label="Laundering")
    ax.set_title("PCA — True Labels", fontsize=14, fontweight="bold")
    ax.legend(markerscale=5)

    ax = axes[1]
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                    c=ensemble_scores, cmap="RdYlGn_r", alpha=0.3, s=5)
    plt.colorbar(sc, ax=ax, label="Ensemble Score")
    ax.set_title("PCA — Ensemble Anomaly Score", fontsize=14, fontweight="bold")

    fig.tight_layout()
    return fig
