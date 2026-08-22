"""
Week 1: Data Acquisition, Cleaning and Preprocessing
Dataset: UCI Machine Learning Repository - Online Retail
Input file: Online Retail.xlsx

Place the downloaded UCI file in the same folder as this script.
Run:
    python week1_online_retail_preprocessing.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = Path("Online Retail.xlsx")
OUTPUT_FILE = Path("online_retail_cleaned.csv")

# 1. Acquire / load
df = pd.read_excel(INPUT_FILE)

print("RAW SHAPE:", df.shape)
print("\nRAW DTYPES:\n", df.dtypes)
print("\nRAW MISSING VALUES:\n", df.isna().sum())
print("\nRAW DUPLICATES:", df.duplicated().sum())

# 2. Preserve a copy for auditability
raw = df.copy()

# 3. Standardize text fields
text_cols = ["InvoiceNo", "StockCode", "Description", "Country"]
for col in text_cols:
    df[col] = df[col].astype("string").str.strip()

# 4. Convert data types explicitly
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce").astype("Int64")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

# 5. Remove exact duplicate records
before = len(df)
df = df.drop_duplicates()
print("\nDUPLICATES REMOVED:", before - len(df))

# 6. Missing-value handling
# CustomerID is an identifier, so imputing an unknown customer would create
# a false identity. For customer-level sales analysis, remove these rows.
df = df.dropna(subset=["CustomerID"])

# Product descriptions are useful but not essential for transaction validity.
# Preserve the transaction and make the missingness explicit.
df["Description"] = df["Description"].fillna("Unknown")

# 7. Cancellation / return handling
# In this dataset, InvoiceNo beginning with C denotes a cancellation.
df["IsCancellation"] = df["InvoiceNo"].str.upper().str.startswith("C", na=False)

# Keep a separate return/cancellation flag for auditability.
# For a clean POSITIVE-SALES analysis, exclude cancellations and
# non-positive quantities/prices.
df["IsReturn"] = df["Quantity"] < 0

sales_df = df.loc[
    (~df["IsCancellation"]) &
    (df["Quantity"] > 0) &
    (df["UnitPrice"] > 0) &
    (df["InvoiceDate"].notna())
].copy()

# 8. Outlier detection using IQR
# Do not automatically delete outliers: wholesale transactions can be legitimate.
def iqr_bounds(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

q_low, q_high = iqr_bounds(sales_df["Quantity"])
p_low, p_high = iqr_bounds(sales_df["UnitPrice"])

sales_df["Quantity_Outlier"] = (
    (sales_df["Quantity"] < q_low) | (sales_df["Quantity"] > q_high)
)
sales_df["UnitPrice_Outlier"] = (
    (sales_df["UnitPrice"] < p_low) | (sales_df["UnitPrice"] > p_high)
)

# 9. Feature engineering
sales_df["SalesAmount"] = sales_df["Quantity"] * sales_df["UnitPrice"]
sales_df["Year"] = sales_df["InvoiceDate"].dt.year
sales_df["Month"] = sales_df["InvoiceDate"].dt.month
sales_df["Day"] = sales_df["InvoiceDate"].dt.day
sales_df["Hour"] = sales_df["InvoiceDate"].dt.hour
sales_df["Weekday"] = sales_df["InvoiceDate"].dt.day_name()

# 10. Final validation
assert sales_df["Quantity"].gt(0).all()
assert sales_df["UnitPrice"].gt(0).all()
assert sales_df["InvoiceDate"].notna().all()
assert sales_df["CustomerID"].notna().all()
assert sales_df["SalesAmount"].notna().all()

print("\nCLEAN SHAPE:", sales_df.shape)
print("\nFINAL MISSING VALUES:\n", sales_df.isna().sum())
print("\nFINAL DUPLICATES:", sales_df.duplicated().sum())
print("\nOUTLIER FLAGS:")
print(sales_df[["Quantity_Outlier", "UnitPrice_Outlier"]].sum())

# 11. Save reproducible output
sales_df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved cleaned data to: {OUTPUT_FILE.resolve()}")

# 12. Optional summary for the report
summary = {
    "rows": len(sales_df),
    "columns": sales_df.shape[1],
    "customers": sales_df["CustomerID"].nunique(),
    "invoices": sales_df["InvoiceNo"].nunique(),
    "products": sales_df["StockCode"].nunique(),
    "countries": sales_df["Country"].nunique(),
    "revenue": sales_df["SalesAmount"].sum(),
}
print("\nSUMMARY:")
for k, v in summary.items():
    print(f"{k}: {v}")
