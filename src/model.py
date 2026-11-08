"""
model.py — Predicts weekly units_sold using a Random Forest Regressor.

Approach:
- Time-based train/test split (never shuffle time series data randomly —
  that leaks the future into training)
- Features engineered from the date itself, plus store, price, promo,
  and holiday flags
- Evaluated with MAE and RMSE (in real units — easy to explain to a
  non-technical stakeholder) and R^2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

df = pd.read_csv(ROOT / "data" / "sales_data.csv", parse_dates=["date"])

# --- Feature engineering ---
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
df["month"] = df["date"].dt.month
df = pd.get_dummies(df, columns=["store"], drop_first=False)

feature_cols = [c for c in df.columns if c.startswith("store_")] + [
    "price", "promotion_flag", "holiday_flag", "week_of_year", "month"
]
X = df[feature_cols]
y = df["units_sold"]

# --- Time-based split: last 20% of dates per store held out as test ---
df_sorted = df.sort_values("date")
split_date = df["date"].quantile(0.8)
train_mask = df["date"] <= split_date
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

# --- Model ---
model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print(f"MAE:  {mae:.1f} units")
print(f"RMSE: {rmse:.1f} units")
print(f"R^2:  {r2:.3f}")

# --- Feature importance plot ---
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
plt.figure(figsize=(7, 5))
importances.plot(kind="barh", color="#2a7")
plt.title("Feature Importance — What Drives Predicted Sales")
plt.tight_layout()
plt.savefig("/home/claude/sales-demand-forecasting/images/feature_importance.png", dpi=120)
plt.close()

# --- Actual vs predicted plot ---
test_dates = df.loc[~train_mask, "date"]
plt.figure(figsize=(10, 5))
plt.plot(test_dates, y_test.values, label="Actual", marker="o", markersize=3)
plt.plot(test_dates, preds, label="Predicted", marker="x", markersize=3)
plt.title(f"Actual vs. Predicted Weekly Sales (Test Set)\nMAE={mae:.0f} units, R^2={r2:.2f}")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()
plt.tight_layout()
plt.savefig("/home/claude/sales-demand-forecasting/images/actual_vs_predicted.png", dpi=120)
plt.close()

print("\nSaved images/feature_importance.png and images/actual_vs_predicted.png")

# Save metrics for README
with open("/home/claude/sales-demand-forecasting/images/metrics.txt", "w") as f:
    f.write(f"MAE: {mae:.1f}\nRMSE: {rmse:.1f}\nR2: {r2:.3f}\n")
