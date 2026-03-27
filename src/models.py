
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from xgboost import XGBClassifier

def train_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> LogisticRegression:
    """Logistic Regression with balanced class weights."""
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
    """Random Forest with balanced class weights."""
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


# ---------------------------------------------------------------------------
# Unsupervised anomaly detection
# ---------------------------------------------------------------------------

def train_isolation_forest(
    X_train: np.ndarray,
    contamination: float = 0.01,
    random_state: int = 42,
) -> IsolationForest:
    """Isolation Forest anomaly detector.

    Predicts -1 for anomalies, 1 for inliers.
    Use decision_function() for a continuous score.
    """
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def train_lof(
    X_train: np.ndarray,
    n_neighbors: int = 20,
    contamination: float = 0.01,
) -> LocalOutlierFactor:
    """Local Outlier Factor (novelty=True so it supports predict on new data).

    Predicts -1 for anomalies, 1 for inliers.
    Use negative_outlier_factor_ or decision_function() for scores.
    """
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=True,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


# ---------------------------------------------------------------------------
# GNN scaffold (PyTorch Geometric)
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv

    class GNNScaffold(nn.Module):
        """Two-layer GraphSAGE for transaction graph node classification.

        Graph construction (TODO):
          - Nodes  : unique accounts
          - Edges  : transactions (Account → Account.1)
          - Node features : account-level aggregates from preprocessing

        Usage (skeleton):
          model = GNNScaffold(in_channels=..., hidden=64, out_channels=2)
          optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

          for epoch in range(epochs):
              model.train()
              optimizer.zero_grad()
              out = model(data.x, data.edge_index)
              loss = F.cross_entropy(out[train_mask], data.y[train_mask],
                                     weight=class_weights)
              loss.backward()
              optimizer.step()
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
    # torch_geometric not installed — GNNScaffold unavailable
    GNNScaffold = None  # type: ignore
