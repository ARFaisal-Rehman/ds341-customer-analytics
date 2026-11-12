"""Segmentation, market-basket analysis and a churn baseline for DS-341."""
from __future__ import annotations

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

SEED = 42
SEGMENT_FEATURES = ["Recency", "Frequency", "Monetary", "AvgOrderValue"]


# --------------------------------------------------------------------- segments
def segment_customers(table: pd.DataFrame, k_range=range(2, 9)) -> tuple[pd.DataFrame, dict]:
    """K-means on log-scaled RFM + AOV. k chosen by silhouette; inertia kept for the elbow plot.

    Raw RFM values are heavily skewed, so un-logged k-means isolates a handful of wholesale buyers in
    their own clusters (the first version had clusters of 1 and 6 customers)."""
    X = np.log1p(table[SEGMENT_FEATURES].clip(lower=0))
    Z = StandardScaler().fit_transform(X)
    sample = np.random.RandomState(SEED).choice(len(Z), size=min(3000, len(Z)), replace=False)
    fits, silhouette, inertia = {}, {}, {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Z)
        fits[k] = km
        silhouette[k] = round(float(silhouette_score(Z[sample], km.labels_[sample])), 4)
        inertia[k] = round(float(km.inertia_), 1)
    # k=2 always splits "active" from "lapsed" and wins silhouette trivially; look for k >= 3.
    best_k = max((k for k in silhouette if k >= 3), key=silhouette.get)
    out = table.copy()
    out["Segment"] = fits[best_k].labels_
    names = _name_segments(out)
    out["SegmentName"] = out["Segment"].map(names)

    total_spend = out["Monetary"].sum()
    rows = []
    for name, g in out.groupby("SegmentName"):
        rows.append({
            "name": name, "customers": int(len(g)), "share_customers": round(len(g) / len(out), 4),
            "share_revenue": round(float(g["Monetary"].sum() / total_spend), 4),
            "median_recency": float(g["Recency"].median()), "median_frequency": float(g["Frequency"].median()),
            "median_monetary": round(float(g["Monetary"].median()), 2),
            "median_aov": round(float(g["AvgOrderValue"].median()), 2),
            "churn_rate": round(float(g["Churn"].mean()), 4) if "Churn" in g else None,
        })
    rows.sort(key=lambda r: -r["median_monetary"])
    pts = out.iloc[sample[:1500]]
    scatter = [{"x": float(r.Recency), "y": float(r.Monetary), "s": r.SegmentName} for r in pts.itertuples()]
    return out, {"k": best_k, "silhouette": silhouette, "inertia": inertia, "segments": rows, "scatter": scatter,
                 "features": SEGMENT_FEATURES}


def _name_segments(df: pd.DataFrame) -> dict:
    """Descriptive names from absolute rules, so a name can't contradict its numbers.
    Value: segment median spend vs the median customer (>2x high, <0.5x low).
    Status: median days since last order (<=30 active, <=90 cooling, else lapsed)."""
    overall = df["Monetary"].median()
    prof = df.groupby("Segment").agg(r=("Recency", "median"), m=("Monetary", "median"))
    names, used = {}, {}
    for sid, row in prof.iterrows():
        value = "High value" if row["m"] > 2 * overall else "Low value" if row["m"] < 0.5 * overall else "Mid value"
        status = "active" if row["r"] <= 30 else "cooling" if row["r"] <= 90 else "lapsed"
        base = f"{value}, {status}"
        used[base] = used.get(base, 0) + 1
        names[sid] = base if used[base] == 1 else f"{base} ({used[base]})"
    return names


# ----------------------------------------------------------------- market basket
def market_basket(df: pd.DataFrame, top_n: int = 300, min_support: float = 0.01,
                  min_lift: float = 1.5) -> dict:
    """FP-Growth over invoices restricted to the `top_n` products by number of invoices.

    Returns directional rules plus a de-duplicated list of product pairs (A->B and B->A share one
    itemset, so ranking rules by lift otherwise shows every pair twice)."""
    names = (df.groupby("StockCode")["Description"].agg(lambda s: s.value_counts().index[0]).str.title()
             .str.replace("'S ", "'s ", regex=False))
    popular = df.groupby("StockCode")["InvoiceNo"].nunique().nlargest(top_n).index
    sub = df[df["StockCode"].isin(popular)]
    basket = (sub.assign(v=True).drop_duplicates(["InvoiceNo", "StockCode"])
              .pivot(index="InvoiceNo", columns="StockCode", values="v").fillna(False).astype(bool))
    basket = basket[basket.sum(axis=1) >= 2]        # a one-item basket can't support a rule
    itemsets = fpgrowth(basket, min_support=min_support, use_colnames=True, max_len=3)
    rules = association_rules(itemsets, metric="lift", min_threshold=min_lift)
    n = len(basket)

    def label(items):
        return [f"{names.get(c, c)} ({c})" for c in sorted(items)]

    out_rules = []
    for r in rules.itertuples():
        out_rules.append({
            "antecedents": label(r.antecedents), "consequents": label(r.consequents),
            "support": round(float(r.support), 4), "count": int(round(r.support * n)),
            "confidence": round(float(r.confidence), 4), "lift": round(float(r.lift), 2),
            "conviction": None if not np.isfinite(r.conviction) else round(float(r.conviction), 2),
            "leverage": round(float(r.leverage), 5),
        })
    out_rules.sort(key=lambda r: (-r["lift"], -r["support"]))

    pairs = {}
    for r in out_rules:
        if len(r["antecedents"]) == 1 and len(r["consequents"]) == 1:
            key = tuple(sorted(r["antecedents"] + r["consequents"]))
            best = pairs.get(key)
            if best is None or r["confidence"] > best["confidence"]:
                pairs[key] = r
    top_pairs = sorted(pairs.values(), key=lambda r: (-r["lift"], -r["support"]))[:25]
    return {"invoices": int(n), "products_considered": int(basket.shape[1]), "min_support": min_support,
            "min_lift": min_lift, "n_itemsets": int(len(itemsets)), "n_rules": len(out_rules),
            "n_pairs": len(pairs), "rules": out_rules[:1500], "top_pairs": top_pairs}


# ---------------------------------------------------------------------- churn
def churn_baseline(table: pd.DataFrame) -> dict:
    """Four classifiers on the leak-free churn table. Chosen by 5-fold CV on the training split;
    the 20% test split is scored once. A baseline only: see the Risk Radar project for the full
    treatment (out-of-time validation, calibration)."""
    features = ["Recency", "Frequency", "Monetary", "UniqueProducts", "Tenure", "AvgOrderValue"]
    X = np.log1p(table[features].clip(lower=0))
    y = table["Churn"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=30, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=10, random_state=SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, eval_metric="logloss",
                                 random_state=SEED, n_jobs=-1),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc")
        model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_te)[:, 1]
        pred = (prob >= 0.5).astype(int)
        results[name] = {"cv_roc_auc": round(float(scores.mean()), 4), "cv_std": round(float(scores.std()), 4),
                         "test_roc_auc": round(float(roc_auc_score(y_te, prob)), 4),
                         "precision": round(float(precision_score(y_te, pred)), 4),
                         "recall": round(float(recall_score(y_te, pred)), 4),
                         "f1": round(float(f1_score(y_te, pred)), 4)}
    selected = max(results, key=lambda n: results[n]["cv_roc_auc"])
    return {"features": features, "n": int(len(table)), "churn_rate": round(float(y.mean()), 4),
            "cutoff": str(table.attrs.get("cutoff", ""))[:10], "selected": selected, "models": results}
