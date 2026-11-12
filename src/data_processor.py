"""Load and clean UCI Online Retail, and build the leak-free churn table."""
from __future__ import annotations

import pandas as pd

# Stock codes that are fees, postage or manual adjustments rather than products.
NON_PRODUCT_CODES = {"POST", "D", "M", "DOT", "CRUK", "C2", "BANK CHARGES", "AMAZONFEE", "ADJUST", "B", "S", "PADS"}


def load_data(filepath) -> pd.DataFrame:
    """Read the CSV export of the UCI workbook (Latin-1 encoded)."""
    return pd.read_csv(filepath, encoding="ISO-8859-1", dtype={"CustomerID": str, "InvoiceNo": str, "StockCode": str})


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Purchases only: drop duplicates, rows without a customer, cancellations, fees and non-positive prices."""
    steps = {"raw_rows": len(df)}
    df = df.drop_duplicates()
    steps["after_duplicates"] = len(df)
    df = df.dropna(subset=["CustomerID"]).copy()
    df["CustomerID"] = pd.to_numeric(df["CustomerID"]).astype("int64").astype(str)
    steps["with_customer_id"] = len(df)
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    df["Description"] = df["Description"].fillna("").str.strip()
    is_cancel = df["InvoiceNo"].astype(str).str.startswith("C") | (df["Quantity"] <= 0)
    df = df[~is_cancel & (df["UnitPrice"] > 0) & ~df["StockCode"].isin(NON_PRODUCT_CODES)].copy()
    steps["purchase_rows"] = len(df)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    steps["customers"] = int(df["CustomerID"].nunique())
    steps["invoices"] = int(df["InvoiceNo"].nunique())
    steps["products"] = int(df["StockCode"].nunique())
    return df, steps


def rfm(df: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    """RFM and extra features from transactions strictly before `reference_date`."""
    df = df[df["InvoiceDate"] < reference_date]
    out = df.groupby("CustomerID").agg(
        LastPurchase=("InvoiceDate", "max"),
        FirstPurchase=("InvoiceDate", "min"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
        UniqueProducts=("StockCode", "nunique"),
        Country=("Country", "first"),
    ).reset_index()
    out["Recency"] = (reference_date - out["LastPurchase"]).dt.days
    out["Tenure"] = (reference_date - out["FirstPurchase"]).dt.days
    out["AvgOrderValue"] = out["Monetary"] / out["Frequency"]
    return out.drop(columns=["LastPurchase", "FirstPurchase"])


def label_churn(df: pd.DataFrame, horizon_days: int = 90) -> pd.DataFrame:
    """Leak-free churn table.

    Cutoff = `horizon_days` before the end of the data. Features use only transactions before the cutoff;
    Churn = 1 if the customer makes no purchase in the `horizon_days` after it. (The first version of this
    project labelled churn as Recency > 90 and also used Recency as a feature, so every model scored 1.0.)
    """
    cutoff = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1) - pd.Timedelta(days=horizon_days)
    table = rfm(df, cutoff)
    future = set(df.loc[df["InvoiceDate"] >= cutoff, "CustomerID"])
    table["Churn"] = (~table["CustomerID"].isin(future)).astype(int)
    table.attrs["cutoff"] = cutoff
    return table
