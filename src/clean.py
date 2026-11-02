"""Data cleaning pipeline for the UCI Online Retail II dataset.

Usage:  python src/clean.py
Input:  data/raw/online_retail_II.xlsx  (two sheets: Year 2009-2010, Year 2010-2011)
Output: data/processed/sales.parquet          clean sales lines (incl. guest customers)
        data/processed/cancellations.parquet  cancelled invoice lines
        data/processed/cleaning_log.csv       row counts after each cleaning step
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_XLSX = ROOT / "data" / "raw" / "online_retail_II.xlsx"
RAW_CACHE = ROOT / "data" / "raw" / "online_retail_II_raw.parquet"
OUT = ROOT / "data" / "processed"

# StockCodes that are fees/adjustments rather than physical products
NON_PRODUCT_CODES = {
    "POST", "DOT", "M", "C2", "D", "S", "B", "CRUK", "PADS",
    "BANK CHARGES", "AMAZONFEE", "ADJUST", "ADJUST2", "TEST001", "TEST002",
}


def load_raw() -> pd.DataFrame:
    """Read both Excel sheets once and cache them as parquet (Excel parsing is slow)."""
    if RAW_CACHE.exists():
        return pd.read_parquet(RAW_CACHE)
    sheets = pd.read_excel(RAW_XLSX, sheet_name=None,
                           dtype={"Invoice": str, "StockCode": str, "Description": str})
    df = pd.concat(sheets.values(), ignore_index=True)
    df.to_parquet(RAW_CACHE, index=False)
    return df


def clean(df: pd.DataFrame):
    log = []

    def step(name, frame):
        log.append({"step": name, "rows": len(frame)})
        return frame

    df = step("Raw data (2 sheets merged)", df)

    df = df.drop_duplicates()
    df = step("Exact duplicate rows removed", df)

    df["Invoice"] = df["Invoice"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    df["Description"] = df["Description"].astype("string").str.strip()

    is_cancel = df["Invoice"].str.startswith("C")
    cancellations = df[is_cancel].copy()
    df = step("Cancellation invoices separated", df[~is_cancel])

    df = step("Non-positive quantity removed", df[df["Quantity"] > 0])
    df = step("Non-positive price removed", df[df["Price"] > 0])

    df = step("Non-product stock codes removed", df[~df["StockCode"].isin(NON_PRODUCT_CODES)])

    df = df.copy()
    df["Customer ID"] = df["Customer ID"].astype("Int64")
    df["Revenue"] = df["Quantity"] * df["Price"]
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["Weekday"] = df["InvoiceDate"].dt.day_name()
    df["Hour"] = df["InvoiceDate"].dt.hour
    df["IsGuest"] = df["Customer ID"].isna()

    log.append({"step": "  of which with Customer ID (used for customer analysis)",
                "rows": int((~df["IsGuest"]).sum())})

    cancellations["Customer ID"] = cancellations["Customer ID"].astype("Int64")
    cancellations["Revenue"] = cancellations["Quantity"] * cancellations["Price"]
    return df, cancellations, pd.DataFrame(log)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    sales, cancellations, log = clean(raw)
    sales.to_parquet(OUT / "sales.parquet", index=False)
    cancellations.to_parquet(OUT / "cancellations.parquet", index=False)
    log.to_csv(OUT / "cleaning_log.csv", index=False)
    print(log.to_string(index=False))
    print(f"\nMissing Customer ID in raw data: {raw['Customer ID'].isna().mean():.1%}")
    print(f"Date range: {sales['InvoiceDate'].min()} -> {sales['InvoiceDate'].max()}")
    print(f"Total revenue (clean): £{sales['Revenue'].sum():,.0f}")


if __name__ == "__main__":
    main()
