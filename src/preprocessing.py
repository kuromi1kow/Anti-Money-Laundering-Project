"""
preprocessing.py
----------------
Data loading, cleaning, and feature engineering for the AML project.

All functions are pure (no global state / side effects).
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Ordered list of features used by all downstream models
FEATURE_COLS = [
    "amount_paid",
    "amount_received",
    "same_currency",
    "payment_format_enc",
    "pay_currency_enc",
    "recv_currency_enc",
    "hour",
    "day_of_week",
    "month",
    # account-level aggregates
    "acct_tx_count",
    "acct_mean_amount",
    "acct_std_amount",
    "acct_unique_counterparties",
    # pattern feature
    "pattern_involved",
]

TARGET_COL = "Is Laundering"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _ibm_data_dir(data_dir: str | Path | None = None) -> Path:
    """Return the IBM-AML data directory, preferring the symlink in data/raw/."""
    if data_dir is not None:
        return Path(data_dir)
    symlink = _PROJECT_ROOT / "data" / "raw" / "ibm-aml"
    if symlink.exists():
        return symlink
    # Fallback: let the caller supply it via kagglehub
    raise FileNotFoundError(
        "IBM AML data not found at data/raw/ibm-aml. "
        "Run `python data/download.py` first, or pass data_dir explicitly."
    )


def _czech_data_dir(data_dir: str | Path | None = None) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    symlink = _PROJECT_ROOT / "data" / "raw" / "czech_bank"
    if symlink.exists():
        return symlink
    raise FileNotFoundError(
        "Czech bank data not found at data/raw/czech_bank. "
        "Run `python data/download.py` first, or pass data_dir explicitly."
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ibm(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load the IBM AML HI-Small transactions CSV.

    Parameters
    ----------
    data_dir : path to the IBM dataset directory (optional).
               Defaults to data/raw/ibm-aml symlink.

    Returns
    -------
    pd.DataFrame with raw transactions.
    """
    path = _ibm_data_dir(data_dir) / "HI-Small_Trans.csv"
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


def load_czech(data_dir: str | Path | None = None) -> Dict[str, pd.DataFrame]:
    """Load all Czech Financial dataset tables.

    Returns
    -------
    dict mapping table name → DataFrame
    (keys: trans, account, client, disp, district, loan, order, card)
    """
    base = _czech_data_dir(data_dir)
    tables: Dict[str, pd.DataFrame] = {}
    for fname in ["trans.csv", "account.csv", "client.csv", "disp.csv",
                  "district.csv", "loan.csv", "order.csv", "card.csv"]:
        results = glob.glob(str(base / "**" / fname), recursive=True)
        if results:
            key = fname.split(".")[0]
            tables[key] = pd.read_csv(results[0], sep=";", low_memory=False)
    return tables


def parse_ibm_patterns(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Parse HI-Small_Patterns.txt into a tidy DataFrame.

    Returns
    -------
    pd.DataFrame with columns: pattern_type, timestamp, from_bank,
    from_account, to_bank, to_account, amount, currency
    """
    path = _ibm_data_dir(data_dir) / "HI-Small_Patterns.txt"
    records = []
    current_pattern = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("BEGIN"):
                current_pattern = line.replace("BEGIN LAUNDERING ATTEMPT - ", "")
            elif line.startswith("END"):
                current_pattern = None
            elif current_pattern and line:
                parts = line.split(",")
                if len(parts) >= 7:
                    records.append({
                        "pattern_type": current_pattern,
                        "timestamp": parts[0],
                        "from_bank": parts[1],
                        "from_account": parts[2],
                        "to_bank": parts[3],
                        "to_account": parts[4],
                        "amount": float(parts[5]),
                        "currency": parts[6],
                    })
    df = pd.DataFrame(records)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(
    df: pd.DataFrame,
    patterns_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add engineered features to the IBM transactions DataFrame.

    Steps
    -----
    1. Temporal features from Timestamp
    2. Currency match flag
    3. Label-encode Payment Format and currencies
    4. Account-level aggregate features (count, mean, std amount, unique counterparties)
    5. pattern_involved flag (if patterns_df is provided)

    Parameters
    ----------
    df : raw IBM transactions (output of load_ibm)
    patterns_df : output of parse_ibm_patterns (optional)

    Returns
    -------
    pd.DataFrame with FEATURE_COLS + TARGET_COL columns present.
    """
    out = df.copy()

    # 1. Temporal
    out["hour"] = out["Timestamp"].dt.hour
    out["day_of_week"] = out["Timestamp"].dt.dayofweek
    out["month"] = out["Timestamp"].dt.month

    # 2. Currency match
    out["same_currency"] = (
        out["Payment Currency"] == out["Receiving Currency"]
    ).astype(int)

    # 3. Rename amount cols for convenience
    out = out.rename(columns={
        "Amount Paid": "amount_paid",
        "Amount Received": "amount_received",
    })

    # 4. Label-encode categoricals
    for col, new_col in [
        ("Payment Format", "payment_format_enc"),
        ("Payment Currency", "pay_currency_enc"),
        ("Receiving Currency", "recv_currency_enc"),
    ]:
        le = LabelEncoder()
        out[new_col] = le.fit_transform(out[col].astype(str))

    # 5. Account-level aggregates (sender account)
    acct_stats = (
        out.groupby("Account")["amount_paid"]
        .agg(acct_tx_count="count", acct_mean_amount="mean", acct_std_amount="std")
        .reset_index()
    )
    acct_stats["acct_std_amount"] = acct_stats["acct_std_amount"].fillna(0)

    counterparties = (
        out.groupby("Account")["Account.1"]
        .nunique()
        .rename("acct_unique_counterparties")
        .reset_index()
    )

    out = out.merge(acct_stats, on="Account", how="left")
    out = out.merge(counterparties, on="Account", how="left")

    # 6. Pattern-involved flag
    if patterns_df is not None and not patterns_df.empty:
        laundering_accounts = (
            set(patterns_df["from_account"]) | set(patterns_df["to_account"])
        )
        out["pattern_involved"] = (
            out["Account"].isin(laundering_accounts)
            | out["Account.1"].isin(laundering_accounts)
        ).astype(int)
    else:
        out["pattern_involved"] = 0

    return out


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def clean_ibm(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw IBM AML transactions.

    - Coerces amount columns to numeric
    - Creates sender_id / receiver_id composite keys
    - Drops exact duplicates
    """
    out = df.copy()
    out["Amount Paid"] = pd.to_numeric(out["Amount Paid"], errors="coerce")
    out["Amount Received"] = pd.to_numeric(out["Amount Received"], errors="coerce")
    out["sender_id"] = out["From Bank"].astype(str) + "_" + out["Account"].astype(str)
    out["receiver_id"] = out["To Bank"].astype(str) + "_" + out["Account.1"].astype(str)
    out = out.drop_duplicates()
    return out


def clean_czech(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Clean Czech Financial dataset tables.

    - trans: parses YYMMDD date int → datetime, coerces amount, fills k_symbol nulls
    - loan: adds binary risk column (B/D → 1, A/C → 0)
    - Drops duplicates on all tables
    """
    out = {k: v.copy() for k, v in tables.items()}

    if "trans" in out:
        t = out["trans"]
        def _parse_czech_date(d):
            try:
                s = str(int(d)).zfill(6)
                return pd.Timestamp(f"19{s[:2]}-{s[2:4]}-{s[4:6]}")
            except Exception:
                return pd.NaT
        t["date"] = t["date"].apply(_parse_czech_date)
        t["amount"] = pd.to_numeric(t["amount"], errors="coerce")
        if "k_symbol" in t.columns:
            t["k_symbol"] = t["k_symbol"].fillna("unknown").replace("", "unknown")
        out["trans"] = t.drop_duplicates()

    if "loan" in out:
        risk_map = {"A": 0, "B": 1, "C": 0, "D": 1}
        out["loan"]["risk"] = out["loan"]["status"].map(risk_map)
        out["loan"] = out["loan"].drop_duplicates()

    for key in [k for k in out if k not in ("trans", "loan")]:
        out[key] = out[key].drop_duplicates()

    return out


def harmonize(
    ibm_df: pd.DataFrame,
    czech_tables: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Merge IBM and Czech datasets into a unified transaction DataFrame.

    Columns in output
    -----------------
    amount, amount_received, timestamp, sender_id, receiver_id,
    _label, payment_type, currency_send, currency_recv, source

    IBM _label  : Is Laundering (0/1)
    Czech _label: loan risk (0/1 from loan table); -1 if no loan info
    """
    # ── IBM slice ────────────────────────────────────────────────────────────
    ibm = pd.DataFrame({
        "amount":          pd.to_numeric(ibm_df.get("Amount Paid",  ibm_df.get("amount_paid")),  errors="coerce"),
        "amount_received": pd.to_numeric(ibm_df.get("Amount Received", ibm_df.get("amount_received")), errors="coerce"),
        "timestamp":       pd.to_datetime(ibm_df["Timestamp"], errors="coerce") if "Timestamp" in ibm_df.columns else ibm_df.get("timestamp"),
        "sender_id":       ibm_df.get("sender_id", ibm_df.get("Account", "").astype(str)),
        "receiver_id":     ibm_df.get("receiver_id", ibm_df.get("Account.1", "").astype(str)),
        "_label":          ibm_df["Is Laundering"].astype(int) if "Is Laundering" in ibm_df.columns else -1,
        "payment_type":    ibm_df.get("Payment Format", "unknown"),
        "currency_send":   ibm_df.get("Payment Currency", "USD"),
        "currency_recv":   ibm_df.get("Receiving Currency", "USD"),
        "source":          "ibm",
    })

    # ── Czech slice ───────────────────────────────────────────────────────────
    czech_rows = []
    if "trans" in czech_tables:
        cz_t = czech_tables["trans"].copy()
        # Attach loan risk via account_id
        if "loan" in czech_tables:
            loan_risk = (
                czech_tables["loan"][["account_id", "risk"]]
                .dropna(subset=["risk"])
                .drop_duplicates("account_id")
            )
            cz_t = cz_t.merge(loan_risk, on="account_id", how="left")
        else:
            cz_t["risk"] = -1
        cz_t["risk"] = cz_t["risk"].fillna(-1).astype(int)

        czech_rows = pd.DataFrame({
            "amount":          pd.to_numeric(cz_t["amount"], errors="coerce").abs(),
            "amount_received": pd.to_numeric(cz_t["amount"], errors="coerce").abs(),
            "timestamp":       cz_t["date"],
            "sender_id":       cz_t["account_id"].astype(str),
            "receiver_id":     cz_t.get("partner", pd.Series(["unknown"] * len(cz_t))).astype(str),
            "_label":          cz_t["risk"],
            "payment_type":    cz_t.get("type", "unknown"),
            "currency_send":   "CZK",
            "currency_recv":   "CZK",
            "source":          "czech",
        })

    frames = [ibm] + ([czech_rows] if len(czech_rows) > 0 else [])
    merged = pd.concat(frames, ignore_index=True)
    return merged


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def get_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/test split with StandardScaler applied to features.

    Parameters
    ----------
    df : output of engineer_features
    test_size : fraction of data for test set
    random_state : random seed

    Returns
    -------
    X_train, X_test, y_train, y_test (numpy arrays)
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
