"""
eda.py — Exploratory analysis of weekly sales data.
Saves plots to ../images/ for use in the README and write-up.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

df = pd.read_csv(ROOT / "data" / "sales_data.csv, parse_dates=["date"])

# 1. Sales trend over time by store
plt.figure(figsize=(10, 5))
for store, group in df.groupby("store"):
    plt.plot(group["date"], group["units_sold"], label=store)
plt.title("Weekly Units Sold by Store")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()
plt.tight_layout()
plt.savefig(ROOT / "images" / "sales_trend.png", dpi=120)
plt.close()

# 2. Promotion effect
promo_avg = df.groupby("promotion_flag")["units_sold"].mean()
plt.figure(figsize=(6, 4))
promo_avg.plot(kind="bar", color=["#888", "#2a7"])
plt.xticks([0, 1], ["No Promotion", "Promotion"], rotation=0)
plt.title("Average Weekly Units Sold: Promotion vs. No Promotion")
plt.ylabel("Avg Units Sold")
plt.tight_layout()
plt.savefig(ROOT / "images" / "promotion_effect.png", dpi=120)
plt.close()

# 3. Holiday effect
holiday_avg = df.groupby("holiday_flag")["units_sold"].mean()
print("Average units sold, holiday vs non-holiday weeks:")
print(holiday_avg)

print("\nOverall summary stats:")
print(df["units_sold"].describe())

print("\nSaved plots to images/sales_trend.png and images/promotion_effect.png")
