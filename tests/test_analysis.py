"""Checks on synthetic data: cleaning, leak-free labels, segment naming, rule metrics, and the app."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data_processor import clean_data, label_churn
from src.models import market_basket, segment_customers


def synthetic(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, inv = [], 0
    start = pd.Timestamp("2011-01-01")
    for c in range(300):
        n_orders = int(rng.integers(1, 12))
        for _ in range(n_orders):
            inv += 1
            day = start + pd.to_timedelta(int(rng.integers(0, 330)), unit="D")
            items = ["PA", "PB"] if rng.random() < 0.3 else []      # PA and PB are bought together on purpose
            items += list(rng.choice(["P" + c for c in "CDEFGHIJ"], size=int(rng.integers(1, 4)), replace=False))
            for code in items:
                rows.append({"InvoiceNo": str(500000 + inv), "StockCode": code, "Description": f"item {code}",
                             "Quantity": int(rng.integers(1, 6)), "InvoiceDate": day, "UnitPrice": 2.5,
                             "CustomerID": str(12000 + c), "Country": "United Kingdom"})
    df = pd.DataFrame(rows)
    extra = df.iloc[:5].copy()
    extra["InvoiceNo"] = "C" + extra["InvoiceNo"]          # cancellations
    extra["Quantity"] = -extra["Quantity"]
    fee = df.iloc[:1].copy()
    fee["StockCode"] = "POST"
    return pd.concat([df, extra, fee, df.iloc[:3]], ignore_index=True)   # + exact duplicates


@pytest.fixture(scope="module")
def cleaned():
    return clean_data(synthetic())


def test_cleaning_removes_cancellations_fees_duplicates(cleaned):
    df, steps = cleaned
    assert not df["InvoiceNo"].str.startswith("C").any()
    assert (df["Quantity"] > 0).all()
    assert "POST" not in set(df["StockCode"])
    assert steps["after_duplicates"] == steps["raw_rows"] - 3


def test_churn_label_uses_only_later_purchases(cleaned):
    df, _ = cleaned
    table = label_churn(df, horizon_days=90)
    cutoff = table.attrs["cutoff"]
    later = set(df.loc[df["InvoiceDate"] >= cutoff, "CustomerID"])
    assert ((table["Churn"] == 0) == table["CustomerID"].isin(later)).all()
    assert (table["Recency"] >= 0).all()          # no feature computed from after the cutoff


def test_segment_names_match_their_numbers(cleaned):
    df, _ = cleaned
    table = label_churn(df)
    _, seg = segment_customers(table, k_range=range(2, 6))
    overall = table["Monetary"].median()
    for s in seg["segments"]:
        status = "active" if s["median_recency"] <= 30 else "cooling" if s["median_recency"] <= 90 else "lapsed"
        assert status in s["name"]
        if s["name"].startswith("High value"):
            assert s["median_monetary"] > 2 * overall
    assert sum(s["customers"] for s in seg["segments"]) == len(table)


def test_rule_metrics_match_a_manual_count(cleaned):
    df, _ = cleaned
    result = market_basket(df, top_n=10, min_support=0.05, min_lift=1.0)
    rule = next(r for r in result["rules"] if r["antecedents"] == ["Item Pa (PA)"] and r["consequents"] == ["Item Pb (PB)"])
    baskets = df.groupby("InvoiceNo")["StockCode"].apply(set)
    baskets = baskets[baskets.apply(len) >= 2]
    n, a, b = len(baskets), baskets.apply(lambda s: "PA" in s), baskets.apply(lambda s: "PB" in s)
    assert rule["support"] == pytest.approx((a & b).sum() / n, abs=1e-3)
    assert rule["confidence"] == pytest.approx((a & b).sum() / a.sum(), abs=1e-3)
    assert rule["lift"] == pytest.approx(((a & b).sum() / n) / ((a.sum() / n) * (b.sum() / n)), abs=0.02)
    keys = [tuple(sorted(p["antecedents"] + p["consequents"])) for p in result["top_pairs"]]
    assert len(keys) == len(set(keys))            # each pair appears once


def test_app_routes():
    import app as dashboard
    client = dashboard.app.test_client()
    for url in ["/", "/segments", "/basket", "/churn", "/recommendations", "/api/basket"]:
        assert client.get(url).status_code == 200, url
    assert client.get("/missing").status_code == 404
    churn = json.loads(client.get("/api/churn").data)
    assert churn["selected"] == max(churn["models"], key=lambda m: churn["models"][m]["cv_roc_auc"])
