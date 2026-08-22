"""
Week 2: Exploratory Data Analysis and Visualization
Dataset: UCI Machine Learning Repository - Online Retail

Input priority:
1) online_retail_cleaned.csv from Week 1, if available
2) Online Retail.xlsx from UCI

Outputs are saved to: week2_eda_outputs/
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CSV_FILE = Path("online_retail_cleaned.csv")
XLSX_FILE = Path("Online Retail.xlsx")
OUT = Path("week2_eda_outputs")
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
if CSV_FILE.exists():
    df = pd.read_csv(CSV_FILE)
    print("Loaded:", CSV_FILE)
else:
    df = pd.read_excel(XLSX_FILE)
    print("Loaded:", XLSX_FILE)

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# ---------------------------------------------------------
# 2. BASIC TRANSFORMATIONS
# ---------------------------------------------------------
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

# If Week 1 cleaned file is used, positive-sales data are already prepared.
# If raw Excel is used, create a positive-sales analysis table.
if "IsCancellation" not in df.columns:
    df["IsCancellation"] = df["InvoiceNo"].astype("string").str.upper().str.startswith("C", na=False)

eda = df.loc[
    (~df["IsCancellation"]) &
    (df["Quantity"] > 0) &
    (df["UnitPrice"] > 0) &
    (df["InvoiceDate"].notna())
].copy()

eda["SalesAmount"] = eda["Quantity"] * eda["UnitPrice"]
eda["Year"] = eda["InvoiceDate"].dt.year
eda["Month"] = eda["InvoiceDate"].dt.month
eda["MonthName"] = eda["InvoiceDate"].dt.strftime("%b")
eda["Hour"] = eda["InvoiceDate"].dt.hour
eda["Weekday"] = eda["InvoiceDate"].dt.day_name()

print("\nEDA table shape:", eda.shape)
print("\nDescriptive statistics:\n", eda[["Quantity", "UnitPrice", "SalesAmount"]].describe())

# ---------------------------------------------------------
# 3. MONTHLY SALES TREND
# ---------------------------------------------------------
monthly = (
    eda.assign(MonthStart=eda["InvoiceDate"].dt.to_period("M").dt.to_timestamp())
       .groupby("MonthStart", as_index=False)["SalesAmount"].sum()
)

plt.figure(figsize=(10,5))
sns.lineplot(data=monthly, x="MonthStart", y="SalesAmount", marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales Amount (£)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUT / "01_monthly_sales_trend.png", dpi=200)
plt.close()

# ---------------------------------------------------------
# 4. TOP 10 COUNTRIES BY TRANSACTION ROWS
# ---------------------------------------------------------
country = eda["Country"].value_counts().head(10).sort_values()

plt.figure(figsize=(9,5.5))
country.plot(kind="barh")
plt.title("Top 10 Countries by Transaction Rows")
plt.xlabel("Transaction Rows")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig(OUT / "02_top_countries.png", dpi=200)
plt.close()

# ---------------------------------------------------------
# 5. TOP 10 PRODUCTS BY QUANTITY
# ---------------------------------------------------------
top_products = (
    eda.groupby("Description", dropna=False)["Quantity"]
       .sum()
       .sort_values(ascending=False)
       .head(10)
       .sort_values()
)

plt.figure(figsize=(10,6))
top_products.plot(kind="barh")
plt.title("Top 10 Products by Units Sold")
plt.xlabel("Units Sold")
plt.ylabel("Product")
plt.tight_layout()
plt.savefig(OUT / "03_top_products_by_quantity.png", dpi=200)
plt.close()

# ---------------------------------------------------------
# 6. SALES BY HOUR
# ---------------------------------------------------------
hourly = eda.groupby("Hour")["SalesAmount"].sum()

plt.figure(figsize=(9,5))
sns.lineplot(x=hourly.index, y=hourly.values, marker="o")
plt.title("Sales by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Sales Amount (£)")
plt.xticks(range(0,24))
plt.tight_layout()
plt.savefig(OUT / "04_sales_by_hour.png", dpi=200)
plt.close()

# ---------------------------------------------------------
# 7. DISTRIBUTIONS
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11,4.5))

sns.histplot(eda["Quantity"].clip(upper=eda["Quantity"].quantile(.99)),
             bins=50, ax=axes[0])
axes[0].set_title("Quantity Distribution (99th percentile clipped)")
axes[0].set_xlabel("Quantity")

sns.histplot(eda["UnitPrice"].clip(upper=eda["UnitPrice"].quantile(.99)),
             bins=50, ax=axes[1])
axes[1].set_title("Unit Price Distribution (99th percentile clipped)")
axes[1].set_xlabel("Unit Price (£)")

plt.tight_layout()
plt.savefig(OUT / "05_numeric_distributions.png", dpi=200)
plt.close()

# ---------------------------------------------------------
# 8. BOXPLOTS FOR OUTLIERS
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10,4))
sns.boxplot(y=eda["Quantity"], ax=axes[0])
axes[0].set_title("Quantity Boxplot")
sns.boxplot(y=eda["UnitPrice"], ax=axes[1])
axes[1].set_title("Unit Price Boxplot")
plt.tight_layout()
plt.savefig(OUT / "06_boxplots.png", dpi=200)
plt.close()

# ---------------------------------------------------------
# 9. CORRELATION HEATMAP
# ---------------------------------------------------------
numeric = eda[["Quantity", "UnitPrice", "SalesAmount"]].copy()
corr = numeric.corr()

plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", square=True)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(OUT / "07_correlation_heatmap.png", dpi=200)
plt.close()

# ---------------------------------------------------------
# 10. WEEKDAY SALES
# ---------------------------------------------------------
weekday_order = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]
weekday_sales = eda.groupby("Weekday")["SalesAmount"].sum().reindex(weekday_order)

plt.figure(figsize=(9,5))
weekday_sales.plot(kind="bar")
plt.title("Sales by Day of Week")
plt.xlabel("Day")
plt.ylabel("Sales Amount (£)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUT / "08_sales_by_weekday.png", dpi=200)
plt.close()

# ---------------------------------------------------------
# 11. SUMMARY TABLES
# ---------------------------------------------------------
summary = pd.DataFrame({
    "Metric": [
        "EDA rows", "EDA columns",
        "Unique customers", "Unique invoices",
        "Unique products", "Countries",
        "Total sales (£)", "Average line sales (£)"
    ],
    "Value": [
        len(eda),
        eda.shape[1],
        eda["CustomerID"].nunique() if "CustomerID" in eda else np.nan,
        eda["InvoiceNo"].nunique(),
        eda["StockCode"].nunique(),
        eda["Country"].nunique(),
        eda["SalesAmount"].sum(),
        eda["SalesAmount"].mean()
    ]
})
summary.to_csv(OUT / "eda_summary.csv", index=False)

print("\nEDA summary:")
print(summary.to_string(index=False))

print("\nAll EDA outputs saved in:", OUT.resolve())
