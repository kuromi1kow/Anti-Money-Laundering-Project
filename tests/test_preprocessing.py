# test_preprocessing.py
# tests for src/preprocessing.py - all synthetic data, no real datasets needed

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    FEATURE_COLS,
    TARGET_COL,
    build_feature_matrix,
    build_transaction_graph,
    clean_czech,
    clean_ibm,
    engineer_features,
    harmonize,
)


# ---
# engineer_features
# ---

def test_engineer_features_cols(ibm_df):
    out = engineer_features(ibm_df)
    for col in FEATURE_COLS:
        assert col in out.columns, f"Missing feature column: {col}"


def test_engineer_features_no_nulls(ibm_df):
    out = engineer_features(ibm_df)
    available = [c for c in FEATURE_COLS if c in out.columns]
    assert out[available].isnull().sum().sum() == 0, "NaNs found in feature columns"


def test_engineer_temporal_ranges(ibm_df):
    out = engineer_features(ibm_df)
    assert out["hour"].between(0, 23).all(), "hour out of range [0, 23]"
    assert out["day_of_week"].between(0, 6).all(), "day_of_week out of range [0, 6]"
    assert out["month"].between(1, 12).all(), "month out of range [1, 12]"


def test_engineer_same_currency_flag(ibm_df):
    out = engineer_features(ibm_df)
    assert set(out["same_currency"].unique()).issubset({0, 1}), "same_currency must be 0 or 1"


def test_engineer_pattern_flag_with_patterns(ibm_df):
    # Build a minimal patterns_df whose accounts overlap with ibm_df
    account = ibm_df["Account"].iloc[0]
    patterns_df = pd.DataFrame({
        "from_account": [account],
        "to_account":   ["UNKNOWN_ACC"],
    })
    out = engineer_features(ibm_df, patterns_df)
    # Rows where sender/receiver matches should have pattern_involved=1
    flagged = out[out["Account"] == account]["pattern_involved"]
    assert (flagged == 1).all(), "pattern_involved should be 1 for matched accounts"


def test_engineer_pattern_flag_without_patterns(ibm_df):
    out = engineer_features(ibm_df, patterns_df=None)
    assert (out["pattern_involved"] == 0).all(), "pattern_involved should be 0 when no patterns given"


# ---
# clean_ibm
# ---

def test_clean_ibm_creates_ids(ibm_df):
    out = clean_ibm(ibm_df)
    assert "sender_id" in out.columns
    assert "receiver_id" in out.columns


def test_clean_ibm_id_format(ibm_df):
    out = clean_ibm(ibm_df)
    # sender_id = From Bank + "_" + Account
    expected = ibm_df["From Bank"].astype(str) + "_" + ibm_df["Account"].astype(str)
    assert (out["sender_id"].values == expected.values).all()


def test_clean_ibm_dedup(ibm_df):
    # Insert a duplicate row
    dup = pd.concat([ibm_df, ibm_df.iloc[:5]], ignore_index=True)
    out = clean_ibm(dup)
    assert len(out) <= len(dup), "clean_ibm should drop duplicates"
    assert len(out) == len(out.drop_duplicates()), "Output still has duplicates"


def test_clean_ibm_numeric_amounts(ibm_df):
    # Inject a non-numeric value
    dirty = ibm_df.copy()
    dirty.loc[0, "Amount Paid"] = "bad_value"
    out = clean_ibm(dirty)
    assert pd.api.types.is_float_dtype(out["Amount Paid"]), "Amount Paid should be float after cleaning"


# ---
# clean_czech
# ---

def test_clean_czech_date_parsed(czech_tables):
    out = clean_czech(czech_tables)
    assert pd.api.types.is_datetime64_any_dtype(out["trans"]["date"]), \
        "trans.date should be datetime after cleaning"


def test_clean_czech_risk_column(czech_tables):
    out = clean_czech(czech_tables)
    valid_risk = {0, 1}
    actual = set(out["loan"]["risk"].dropna().unique())
    assert actual.issubset(valid_risk), f"loan.risk must be in {{0, 1}}, got {actual}"


def test_clean_czech_k_symbol_no_nulls(czech_tables):
    out = clean_czech(czech_tables)
    assert out["trans"]["k_symbol"].isnull().sum() == 0, "k_symbol should have no nulls after cleaning"


# ---
# harmonize
# ---

REQUIRED_COLS = {"amount", "amount_received", "timestamp", "sender_id",
                 "receiver_id", "_label", "payment_type",
                 "currency_send", "currency_recv", "source"}


def test_harmonize_schema(ibm_df, czech_tables):
    ibm_c = clean_ibm(ibm_df)
    cz_c  = clean_czech(czech_tables)
    merged = harmonize(ibm_c, cz_c)
    assert REQUIRED_COLS.issubset(merged.columns), \
        f"Missing columns: {REQUIRED_COLS - set(merged.columns)}"


def test_harmonize_source_values(ibm_df, czech_tables):
    ibm_c = clean_ibm(ibm_df)
    cz_c  = clean_czech(czech_tables)
    merged = harmonize(ibm_c, cz_c)
    assert set(merged["source"].unique()).issubset({"ibm", "czech"})


def test_harmonize_label_range(ibm_df, czech_tables):
    ibm_c = clean_ibm(ibm_df)
    cz_c  = clean_czech(czech_tables)
    merged = harmonize(ibm_c, cz_c)
    assert set(merged["_label"].unique()).issubset({-1, 0, 1}), \
        f"_label values out of range: {merged['_label'].unique()}"


def test_harmonize_ibm_only(ibm_df):
    """harmonize should work when czech_tables is empty."""
    ibm_c = clean_ibm(ibm_df)
    merged = harmonize(ibm_c, {})
    assert (merged["source"] == "ibm").all()
    assert len(merged) == len(ibm_c)


# ---
# build_feature_matrix
# ---

def test_build_feature_matrix_shape(ibm_df):
    X, y, cols = build_feature_matrix(ibm_df)
    assert X.shape[0] == len(ibm_df), "Row count mismatch"
    assert X.shape[1] == len(cols), "Column count mismatch"
    assert len(y) == len(ibm_df), "Label length mismatch"


def test_build_feature_matrix_scaled(ibm_df):
    X, _, _ = build_feature_matrix(ibm_df)
    col_means = np.abs(X.mean(axis=0))
    assert (col_means < 1.0).all(), "Features should be approximately zero-mean after scaling"


# ---
# build_transaction_graph
# ---

def test_build_transaction_graph_type(ibm_df):
    import networkx as nx
    ibm_c = clean_ibm(ibm_df)
    G = build_transaction_graph(ibm_c)
    assert isinstance(G, nx.DiGraph), "Should return a DiGraph"


def test_build_transaction_graph_has_edges(ibm_df):
    ibm_c = clean_ibm(ibm_df)
    G = build_transaction_graph(ibm_c)
    assert G.number_of_edges() > 0, "Graph should have at least one edge"


def test_build_transaction_graph_max_edges(ibm_df):
    ibm_c = clean_ibm(ibm_df)
    G = build_transaction_graph(ibm_c, max_edges=10)
    assert G.number_of_edges() <= 10, "max_edges should be respected"
