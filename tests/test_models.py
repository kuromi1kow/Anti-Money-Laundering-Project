# test_models.py
# tests for src/models.py - uses small synthetic arrays so these run fast

import time

import numpy as np
import pytest

from src.models import (
    Autoencoder,
    _TORCH_AVAILABLE,
    _normalize,
    compute_ensemble_scores,
    train_autoencoder,
    train_isolation_forest,
    train_kmeans,
    train_logistic,
    train_random_forest,
)

_needs_torch = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")


# ---
# _normalize
# ---

def test_normalize_range(X_y):
    X, _ = X_y
    scores = X[:, 0]
    out = _normalize(scores)
    assert out.min() >= 0.0 and out.max() <= 1.0, "Output must be in [0, 1]"


def test_normalize_constant():
    s = np.ones(50)
    out = _normalize(s)
    assert (out == 0.0).all(), "Constant array should normalize to all zeros"


def test_normalize_preserves_shape(X_y):
    X, _ = X_y
    s = X[:, 0]
    assert _normalize(s).shape == s.shape


# ---
# Supervised models
# ---

def test_train_logistic_predict_proba(X_y):
    X, y = X_y
    model = train_logistic(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2), "predict_proba shape should be (n, 2)"
    assert (proba >= 0).all() and (proba <= 1).all(), "Probabilities must be in [0, 1]"


def test_train_logistic_binary_output(X_y):
    X, y = X_y
    model = train_logistic(X, y)
    preds = model.predict(X)
    assert set(preds).issubset({0, 1}), "Predictions must be binary"


def test_train_random_forest_importances(X_y):
    X, y = X_y
    model = train_random_forest(X, y, n_estimators=10)
    importances = model.feature_importances_
    assert importances.shape == (X.shape[1],)
    assert abs(importances.sum() - 1.0) < 1e-6, "Feature importances should sum to 1"


def test_train_random_forest_predict_proba(X_y):
    X, y = X_y
    model = train_random_forest(X, y, n_estimators=10)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)


# ---
# Unsupervised — Isolation Forest
# ---

def test_train_isolation_forest_returns_tuple(X_y):
    X, _ = X_y
    result = train_isolation_forest(X)
    assert isinstance(result, tuple) and len(result) == 2


def test_train_isolation_forest_scores_range(X_y):
    X, _ = X_y
    _, scores = train_isolation_forest(X)
    assert scores.shape == (len(X),)
    assert scores.min() >= 0.0 and scores.max() <= 1.0


# ---
# Unsupervised — K-Means
# ---

def test_train_kmeans_returns_tuple(X_y):
    X, _ = X_y
    result = train_kmeans(X, n_clusters=5)
    assert isinstance(result, tuple) and len(result) == 2


def test_train_kmeans_scores_range(X_y):
    X, _ = X_y
    _, scores = train_kmeans(X, n_clusters=5)
    assert scores.shape == (len(X),)
    assert scores.min() >= 0.0 and scores.max() <= 1.0


# ---
# Autoencoder (skipped when torch not installed)
# ---

@_needs_torch
def test_autoencoder_forward_shape():
    import torch
    model = Autoencoder(in_dim=8, hidden=16, bottleneck=4)
    x = torch.randn(10, 8)
    out = model(x)
    assert out.shape == x.shape, "Autoencoder output should match input shape"


@_needs_torch
def test_autoencoder_bottleneck_reduces_dims():
    import torch
    model = Autoencoder(in_dim=8, hidden=16, bottleneck=4)
    x = torch.randn(10, 8)
    encoded = model.encoder(x)
    assert encoded.shape == (10, 4), "Encoder output shape should be (batch, bottleneck)"


@_needs_torch
def test_train_autoencoder_returns_tuple(X_y):
    X, _ = X_y
    result = train_autoencoder(X, device="cpu", epochs=2)
    assert isinstance(result, tuple) and len(result) == 2


@_needs_torch
def test_train_autoencoder_scores_range(X_y):
    X, _ = X_y
    _, scores = train_autoencoder(X, device="cpu", epochs=2)
    assert scores.shape == (len(X),)
    assert scores.min() >= 0.0 and scores.max() <= 1.0


@_needs_torch
def test_train_autoencoder_fast(X_y):
    X, _ = X_y
    t0 = time.time()
    train_autoencoder(X, device="cpu", epochs=2)
    elapsed = time.time() - t0
    assert elapsed < 10, f"Autoencoder training took too long: {elapsed:.1f}s"


# ---
# Ensemble
# ---

def test_compute_ensemble_range(X_y):
    X, _ = X_y
    s1, s2 = X[:, 0], X[:, 1]
    out = compute_ensemble_scores(s1, s2)
    assert out.shape == (len(X),)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_compute_ensemble_three_arrays(X_y):
    X, _ = X_y
    s1, s2, s3 = X[:, 0], X[:, 1], X[:, 2]
    out = compute_ensemble_scores(s1, s2, s3)
    assert out.shape == (len(X),)


def test_compute_ensemble_single_array(X_y):
    X, _ = X_y
    s = X[:, 0]
    out = compute_ensemble_scores(s)
    # With a single input, normalized result should be all zeros or [0,1]
    assert out.min() >= 0.0 and out.max() <= 1.0
