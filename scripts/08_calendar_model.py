import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

D = "data/donnees_cemracs_2026"

# --- 1. Daily demand (peak) and daily mean temperature ---
ds = xr.open_dataset(f"{D}/app/eco2mix-demand_nuts1-fr_hour-20132024.nc")
national = ds["demand"].sum(dim="region").to_series()
demand_peak = national.resample("D").max()

dt = xr.open_dataset(f"{D}/app/pecd42-t2m_nuts1-fr_hour-20102020.nc")
temp = dt["surface_temperature"].mean(dim="region").to_series().resample("D").mean()

df = pd.DataFrame({"temp": temp, "demand": demand_peak / 1000}).dropna()

# --- 2. Restrict to the cold regime (heating side) ---
df = df[df["temp"] < 15.0].copy()

# --- 3. Build calendar features ---
df["dayofweek"] = df.index.dayofweek       # 0=Mon .. 6=Sun
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

# --- 4. Two models, compared ---
# Model A: temperature only (what we had)
# Model B: temperature + weekday dummies

def ols(X, y):
    """Ordinary least squares. X includes a column of ones."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1 - (resid @ resid) / ((y - y.mean()) ** 2).sum()
    return beta, r2, resid

y = df["demand"].values

# Model A
XA = np.column_stack([np.ones(len(df)), df["temp"].values])
bA, r2A, _ = ols(XA, y)

# Model B: add 6 weekday dummies (Monday is the reference)
dow_dummies = np.column_stack([(df["dayofweek"] == d).astype(float) for d in range(1, 7)])
XB = np.column_stack([np.ones(len(df)), df["temp"].values, dow_dummies])
bB, r2B, residB = ols(XB, y)

print("=== Model A: temperature only ===")
print(f"  Slope: {bA[1]*1000:.0f} MW/°C   R² = {r2A:.3f}")

print("\n=== Model B: temperature + day-of-week ===")
print(f"  Slope: {bB[1]*1000:.0f} MW/°C   R² = {r2B:.3f}")
labels = ["Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
print("  Day-of-week effects (GW vs Monday):")
for lab, coef in zip(labels, bB[2:]):
    print(f"    {lab}: {coef:+.2f}")

print(f"\n  R² improved from {r2A:.3f} to {r2B:.3f}")
