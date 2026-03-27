# models.py
# training wrappers for all the models we tried in this project

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = None  # type: ignore
    _XGBOOST_AVAILABLE = False


def _normalize(s: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]. Small epsilon avoids div-by-zero on constant arrays."""
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo + 1e-9)


# --- supervised models ---

def train_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> LogisticRegression:
    """Logistic regression with balanced class weights for imbalanced fraud data."""
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Random forest with balanced class weights."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
):
    """XGBoost with scale_pos_weight to handle class imbalance."""
    if not _XGBOOST_AVAILABLE:
        raise RuntimeError(
            "XGBoost is not available. Fix: brew install libomp, then restart the kernel."
        )
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


# --- unsupervised anomaly detection ---

def train_isolation_forest(
    X_train: np.ndarray,
    contamination: float | str = "auto",
    random_state: int = 42,
) -> Tuple[IsolationForest, np.ndarray]:
    """Isolation forest - returns (model, anomaly_scores) where higher = more anomalous."""
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train)
    # decision_function gives higher values for normal points, so flip it
    scores = _normalize(-model.decision_function(X_train))
    return model, scores


def train_lof(
    X_train: np.ndarray,
    n_neighbors: int = 20,
    contamination: float = 0.01,
) -> LocalOutlierFactor:
    """Local Outlier Factor. novelty=True so we can call predict on held-out data."""
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=True,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def train_kmeans(
    X_train: np.ndarray,
    n_clusters: int = 10,
    random_state: int = 42,
) -> Tuple[KMeans, np.ndarray]:
    """K-means anomaly detector - anomaly score = distance to nearest cluster center."""
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    model.fit(X_train)
    dists = np.min(model.transform(X_train), axis=1)
    scores = _normalize(dists)
    return model, scores


# --- autoencoder (only available if torch is installed) ---

if _TORCH_AVAILABLE:
    class Autoencoder(nn.Module):
        """Simple autoencoder for anomaly detection via reconstruction error.

        Architecture: in_dim -> hidden -> bottleneck -> hidden -> in_dim
        """

        def __init__(self, in_dim: int, hidden: int = 32, bottleneck: int = 8):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, bottleneck), nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck, hidden), nn.ReLU(),
                nn.Linear(hidden, in_dim),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    def train_autoencoder(
        X_train: np.ndarray,
        device: str = "cpu",
        epochs: int = 30,
        lr: float = 1e-3,
        hidden: int = 32,
        bottleneck: int = 8,
    ) -> Tuple:
        """Train autoencoder and return (model, anomaly_scores).

        Anomaly score = reconstruction MSE, normalized to [0, 1].
        """
        model = Autoencoder(X_train.shape[1], hidden=hidden, bottleneck=bottleneck).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        X_t = torch.tensor(X_train, dtype=torch.float32).to(device)

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = ((model(X_t) - X_t) ** 2).mean()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            recon = model(X_t).cpu().numpy()

        mse = np.mean((recon - X_train) ** 2, axis=1)
        scores = _normalize(mse)
        return model, scores

else:
    Autoencoder = None  # type: ignore

    def train_autoencoder(*args, **kwargs):  # type: ignore
        raise RuntimeError("PyTorch is required for train_autoencoder. Install pytorch to use this function.")


# --- ensemble ---

def compute_ensemble_scores(*score_arrays: np.ndarray) -> np.ndarray:
    """Average multiple anomaly score arrays into a single ensemble score.

    Each array is re-normalized before averaging so scale differences don't bias the result.
    """
    return np.mean([_normalize(s) for s in score_arrays], axis=0)


# --- GNN scaffold (needs torch_geometric, fully optional) ---

try:
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv

    class GNNScaffold(nn.Module):
        """Two-layer GraphSAGE for node classification on the transaction graph.

        TODO: still need to build the node feature matrix and edge index
        from the transaction data before this can be used end-to-end.
        """

        def __init__(self, in_channels: int, hidden: int = 64, out_channels: int = 2):
            super().__init__()
            self.conv1 = SAGEConv(in_channels, hidden)
            self.conv2 = SAGEConv(hidden, out_channels)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)
            x = self.conv2(x, edge_index)
            return x

except ImportError:
    GNNScaffold = None  # type: ignore
