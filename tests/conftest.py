# conftest.py
# shared fixtures with synthetic data so tests don't need the real datasets

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ibm_df():
    """200-row synthetic transactions that match the IBM dataset schema."""
    n = 200
    rng = np.random.default_rng(42)
    accounts = [f"ACC{i:04d}" for i in range(50)]
    return pd.DataFrame({
        "Timestamp":          pd.date_range("2022-01-01", periods=n, freq="h"),
        "From Bank":          rng.integers(1, 10, n).astype(str),
        "Account":            rng.choice(accounts, n),
        "To Bank":            rng.integers(1, 10, n).astype(str),
        "Account.1":          rng.choice(accounts, n),
        "Amount Paid":        rng.exponential(1000, n),
        "Amount Received":    rng.exponential(1000, n),
        "Payment Currency":   rng.choice(["US Dollar", "Euro", "Yuan"], n),
        "Receiving Currency": rng.choice(["US Dollar", "Euro", "Yuan"], n),
        "Payment Format":     rng.choice(["Cheque", "Wire", "Credit Card", "Reinvestment"], n),
        "Is Laundering":      rng.choice([0, 1], n, p=[0.92, 0.08]),
    })


@pytest.fixture
def czech_tables():
    """Small Czech trans + loan tables with the same columns as the real data."""
    n = 100
    rng = np.random.default_rng(0)
    trans = pd.DataFrame({
        "account_id": np.arange(n),
        "date":       rng.integers(930101, 981231, n),   # YYMMDD int format
        "amount":     rng.exponential(5000, n),
        "type":       rng.choice(["PRIJEM", "VYDAJ"], n),
        "k_symbol":   [None if i % 5 == 0 else "SIPO" for i in range(n)],
        "partner":    [f"PARTNER{i % 20}" for i in range(n)],
    })
    loan = pd.DataFrame({
        "account_id": np.arange(20),
        "status":     rng.choice(["A", "B", "C", "D"], 20),
        "amount":     rng.exponential(10_000, 20),
    })
    return {"trans": trans, "loan": loan}


@pytest.fixture
def X_y():
    """300-sample feature matrix, ~5% fraud rate to match real class imbalance."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((300, 8)).astype(np.float32)
    y = np.zeros(300, dtype=int)
    y[:15] = 1   # 5% fraud rate
    return X, y
