"""Run the whole DS-341 analysis and write results/*.json for the dashboard.

    python run_pipeline.py [path/to/data.csv]
"""
import json
import sys
import time
from pathlib import Path

from src.data_processor import clean_data, label_churn, load_data
from src.models import churn_baseline, market_basket, segment_customers

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"


def main(data_path: Path) -> None:
    t0 = time.time()
    raw = load_data(data_path)
    df, cleaning = clean_data(raw)
    table = label_churn(df)
    segmented, segments = segment_customers(table)
    basket = market_basket(df)
    churn = churn_baseline(table)

    monthly = (df.set_index("InvoiceDate").resample("MS")
               .agg({"TotalPrice": "sum", "InvoiceNo": "nunique", "CustomerID": "nunique"}).reset_index())
    overview = {
        "period": [df["InvoiceDate"].min().strftime("%Y-%m-%d"), df["InvoiceDate"].max().strftime("%Y-%m-%d")],
        "cleaning": {k: int(v) for k, v in cleaning.items()},
        "revenue": round(float(df["TotalPrice"].sum()), 2),
        "monthly": [{"month": r.InvoiceDate.strftime("%Y-%m"), "revenue": round(float(r.TotalPrice), 2),
                     "orders": int(r.InvoiceNo), "customers": int(r.CustomerID)} for r in monthly.itertuples()],
        "top_countries": df.groupby("Country")["TotalPrice"].sum().nlargest(6).round(2).to_dict(),
    }

    RESULTS.mkdir(exist_ok=True)
    for name, payload in [("overview", overview), ("segments", segments), ("basket", basket), ("churn", churn)]:
        (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s")
    print("cleaning", cleaning)
    print("segments k", segments["k"], segments["silhouette"])
    for s in segments["segments"]:
        print("  ", s)
    print("basket", {k: basket[k] for k in ("invoices", "products_considered", "n_itemsets", "n_rules", "n_pairs")})
    for p in basket["top_pairs"][:5]:
        print("  ", p["antecedents"], "->", p["consequents"], p["lift"], p["confidence"], p["support"])
    print("churn", churn["selected"], churn["churn_rate"], json.dumps(churn["models"]))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "data.csv")
