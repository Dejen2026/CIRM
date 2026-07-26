import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

D = "data/donnees_cemracs_2026"

# --- 1. Daily peak demand + daily mean temperature ---
ds = xr.open_dataset(f"{D}/app/eco2mix-demand_nuts1-fr_hour-20132024.nc")
national = ds["demand"].sum(dim="region").to_series()
demand_peak = national.resample("D").max()

dt = xr.open_dataset(f"{D}/app/pecd42-t2m_nuts1-fr_hour-20102020.nc")
temp = dt["surface_temperature"].mean(dim="region").to_series().resample("D").mean()

df = pd.DataFrame({"temp": temp, "demand": demand_peak / 1000}).dropna()

# --- 2. Build lagged temperatures (previous days) ---
for lag in range(1, 5):
    df[f"temp_lag{lag}"] = df["temp"].shift(lag)

df["dayofweek"] = df.index.dayofweek
df = df.dropna()                       # drop rows without full lags
df = df[df["temp"] < 15.0].copy()      # cold regime

def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1 - (resid @ resid) / ((y - y.mean()) ** 2).sum()
    return beta, r2

y = df["demand"].values
dow = np.column_stack([(df["dayofweek"] == d).astype(float) for d in range(1, 7)])

# --- Model B: today's temperature only (+ calendar) ---
XB = np.column_stack([np.ones(len(df)), df["temp"].values, dow])
bB, r2B = ols(XB, y)

# --- Model C: today + 4 previous days (+ calendar) ---
temps = np.column_stack([df["temp"].values] + [df[f"temp_lag{l}"].values for l in range(1, 5)])
XC = np.column_stack([np.ones(len(df)), temps, dow])
bC, r2C = ols(XC, y)

print("=== Model B: today's temperature only (+ calendar) ===")
print(f"  R² = {r2B:.3f}")

print("\n=== Model C: today + 4 previous days (+ calendar) ===")
print(f"  R² = {r2C:.3f}")
print("  Temperature coefficients (MW/°C):")
lag_labels = ["today", "yesterday", "2 days ago", "3 days ago", "4 days ago"]
for lab, coef in zip(lag_labels, bC[1:6]):
    print(f"    {lab:12s}: {coef*1000:+.0f}")
total = bC[1:6].sum() * 1000
print(f"    {'TOTAL':12s}: {total:+.0f} MW/°C (sustained cold)")
print(f"    (today-only was {bB[1]*1000:.0f} MW/°C)")

print(f"\n  R² improved from {r2B:.3f} to {r2C:.3f}")

# --- Illustrate: day 1 vs day 4 of a cold spell ---
print("\n=== Answering the journal's question ===")
print("For the SAME daily-mean temperature, a cold spell's later days have")
print("colder preceding days, so predicted demand is higher.")
print("The lag coefficients above quantify exactly how much.")
