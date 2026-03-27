# test_evaluate.py
# tests for src/evaluate.py - uses synthetic scores/models, no real data

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.evaluate import (
    evaluate_supervised,
    evaluate_unsupervised,
    plot_pca_projection,
    plot_pr_roc_curves,
    summary_table,
)
from src.models import train_isolation_forest, train_logistic


# ---
# Helpers
# ---

@pytest.fixture
def fitted_logistic(X_y):
    X, y = X_y
    return train_logistic(X, y), X, y


@pytest.fixture
def fitted_iforest(X_y):
    X, y = X_y
    model, scores = train_isolation_forest(X)
    return model, scores, X, y


@pytest.fixture
def score_arrays(X_y):
    """Four synthetic score arrays to mimic unsupervised model outputs."""
    X, y = X_y
    rng = np.random.default_rng(99)
    return {
        "ModelA": rng.random(len(y)),
        "ModelB": rng.random(len(y)),
        "ModelC": rng.random(len(y)),
        "Ensemble": rng.random(len(y)),
    }, y


# ---
# evaluate_supervised
# ---

METRIC_KEYS = {"name", "precision", "recall", "f1", "roc_auc", "pr_auc"}


def test_evaluate_supervised_keys(fitted_logistic):
    model, X, y = fitted_logistic
    result = evaluate_supervised(model, X, y, "Logistic")
    assert METRIC_KEYS == set(result.keys()), f"Unexpected keys: {set(result.keys())}"


def test_evaluate_supervised_metric_range(fitted_logistic):
    model, X, y = fitted_logistic
    result = evaluate_supervised(model, X, y, "Logistic")
    for key in ("precision", "recall", "f1", "roc_auc", "pr_auc"):
        assert 0.0 <= result[key] <= 1.0, f"{key}={result[key]} out of [0, 1]"


def test_evaluate_supervised_name_preserved(fitted_logistic):
    model, X, y = fitted_logistic
    result = evaluate_supervised(model, X, y, "MyModel")
    assert result["name"] == "MyModel"


# ---
# evaluate_unsupervised
# ---

def test_evaluate_unsupervised_keys(fitted_iforest):
    model, _, X, y = fitted_iforest
    result = evaluate_unsupervised(model, X, y, "IForest")
    assert METRIC_KEYS == set(result.keys())


def test_evaluate_unsupervised_metric_range(fitted_iforest):
    model, _, X, y = fitted_iforest
    result = evaluate_unsupervised(model, X, y, "IForest")
    for key in ("precision", "recall", "f1", "roc_auc", "pr_auc"):
        assert 0.0 <= result[key] <= 1.0, f"{key}={result[key]} out of [0, 1]"


# ---
# summary_table
# ---

def test_summary_table_shape(X_y):
    X, y = X_y
    results = [
        evaluate_supervised(train_logistic(X, y), X, y, "Logistic"),
        evaluate_unsupervised(train_isolation_forest(X)[0], X, y, "IForest"),
    ]
    df = summary_table(results)
    assert df.shape == (2, 5), f"Expected (2, 5), got {df.shape}"


def test_summary_table_columns(X_y):
    X, y = X_y
    results = [evaluate_supervised(train_logistic(X, y), X, y, "LR")]
    df = summary_table(results)
    expected = {"Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"}
    assert expected == set(df.columns)


def test_summary_table_sorted_by_pr_auc(X_y):
    X, y = X_y
    results = [
        evaluate_supervised(train_logistic(X, y), X, y, "LR"),
        evaluate_unsupervised(train_isolation_forest(X)[0], X, y, "IF"),
    ]
    df = summary_table(results)
    pr_aucs = df["PR-AUC"].tolist()
    assert pr_aucs == sorted(pr_aucs, reverse=True), "summary_table should be sorted descending by PR-AUC"


# ---
# plot_pr_roc_curves
# ---

def test_plot_pr_roc_curves_returns_figure(score_arrays):
    scores, y = score_arrays
    fig = plot_pr_roc_curves(scores, y)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_pr_roc_curves_has_two_axes(score_arrays):
    scores, y = score_arrays
    fig = plot_pr_roc_curves(scores, y)
    assert len(fig.axes) == 2, "Should produce exactly 2 axes (PR + ROC)"
    plt.close(fig)


def test_plot_pr_roc_curves_single_model(X_y):
    _, y = X_y
    rng = np.random.default_rng(1)
    scores = {"OnlyModel": rng.random(len(y))}
    fig = plot_pr_roc_curves(scores, y)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ---
# plot_pca_projection
# ---

def test_plot_pca_projection_returns_figure(X_y):
    X, y = X_y
    rng = np.random.default_rng(2)
    ens = rng.random(len(y))
    fig = plot_pca_projection(X, y, ens)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_pca_projection_has_two_axes(X_y):
    X, y = X_y
    rng = np.random.default_rng(3)
    ens = rng.random(len(y))
    fig = plot_pca_projection(X, y, ens)
    assert len(fig.axes) >= 2, "Should produce 2 scatter panels"
    plt.close(fig)
