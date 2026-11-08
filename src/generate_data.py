"""
generate_data.py

Generates a synthetic but realistic weekly retail sales dataset for a
demand forecasting portfolio project.

Why synthetic data: it lets the project run end-to-end with no external
downloads, and the generating process is fully documented here so anyone
reviewing the repo can see exactly what patterns are baked in (trend,
yearly seasonality, promotions, holidays, noise) and check that the model
actually recovers them. Swap in a real POS/sales export and the rest of
the pipeline (eda.py, model.py) works unchanged.
"""

import numpy as np
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

np.random.seed(42)

# 3 years of weekly data across 3 stores
start_date = pd.Timestamp("2023-01-01")
n_weeks = 156  # 3 years
stores = ["Store_A", "Store_B", "Store_C"]

rows = []
for store in stores:
    base_level = {"Store_A": 1200, "Store_B": 800, "Store_C": 1500}[store]
    trend_rate = {"Store_A": 1.5, "Store_B": 0.8, "Store_C": -0.5}[store]

    dates = pd.date_range(start_date, periods=n_weeks, freq="W-MON")
    week_num = np.arange(n_weeks)

    trend = base_level + trend_rate * week_num
    yearly_seasonality = 250 * np.sin(2 * np.pi * week_num / 52 - np.pi / 2) + 250
    promotion_flag = np.random.binomial(1, 0.15, n_weeks)
    promotion_effect = promotion_flag * np.random.uniform(200, 450, n_weeks)

    # Holiday spikes: weeks 47-48 (Black Friday/Thanksgiving) and 51-52 (Christmas)
    week_of_year = dates.isocalendar().week.values
    holiday_flag = np.isin(week_of_year, [47, 48, 51, 52]).astype(int)
    holiday_effect = holiday_flag * np.random.uniform(400, 700, n_weeks)

    noise = np.random.normal(0, 90, n_weeks)

    units_sold = trend + yearly_seasonality + promotion_effect + holiday_effect + noise
    units_sold = np.clip(units_sold, 50, None).round().astype(int)

    price = np.random.uniform(15, 25, n_weeks).round(2)
    price[promotion_flag == 1] *= 0.85  # promo weeks are discounted
    price = price.round(2)

    df = pd.DataFrame({
        "date": dates,
        "store": store,
        "units_sold": units_sold,
        "price": price,
        "promotion_flag": promotion_flag,
        "holiday_flag": holiday_flag,
    })
    rows.append(df)

full_df = pd.concat(rows, ignore_index=True).sort_values(["date", "store"])
full_df.to_csv(ROOT / "data" / "sales_data.csv", index=False)
print(f"Generated {len(full_df)} rows across {len(stores)} stores.")
print(full_df.head())
