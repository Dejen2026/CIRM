import xarray as xr
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

D = "data/donnees_cemracs_2026"

# --- 1. Load hourly demand, aggregate to daily ---
ds = xr.open_dataset(f"{D}/app/eco2mix-demand_nuts1-fr_hour-20132024.nc")
national = ds["demand"].sum(dim="region").to_series()
demand_daily = national.resample("D").mean()          # daily mean demand (MW)
demand_peak  = national.resample("D").max()           # daily peak demand (MW)

# --- 2. Load hourly temperature, aggregate to daily mean ---
dt = xr.open_dataset(f"{D}/app/pecd42-t2m_nuts1-fr_hour-20102020.nc")
temp_national = dt["surface_temperature"].mean(dim="region").to_series()
temp_daily = temp_national.resample("D").mean()       # daily mean temperature (°C)

# --- 3. Join on date ---
df = pd.DataFrame({
    "temp": temp_daily,
    "demand_mean": demand_daily,
    "demand_peak": demand_peak,
}).dropna()
print(f"Days with both demand and temperature: {len(df)}")
print(f"Range: {df.index.min().date()} to {df.index.max().date()}")

# --- 4. Fit threshold-linear model on the COLD side (below 15°C) ---
THRESHOLD = 15.0
cold = df[df["temp"] < THRESHOLD]

slope, intercept, r, p, se = stats.linregress(cold["temp"], cold["demand_peak"] / 1000)
print(f"\n=== Threshold-linear model (peak demand, T < {THRESHOLD}°C) ===")
print(f"  Slope:        {slope*1000:.0f} MW per °C")
print(f"  R²:           {r**2:.3f}")
print(f"  (RTE's published figure is about -2400 MW/°C)")

# --- 5. Plot: data + fitted line ---
plt.figure(figsize=(10, 6))
plt.scatter(df["temp"], df["demand_peak"] / 1000, s=6, alpha=0.3,
            color="steelblue", label="Daily peak demand")
x_line = np.linspace(cold["temp"].min(), THRESHOLD, 50)
plt.plot(x_line, slope * x_line + intercept, "r-", lw=2,
         label=f"Fit: {slope*1000:.0f} MW/°C, R²={r**2:.2f}")
plt.axvline(THRESHOLD, color="gray", ls="--", alpha=0.7,
            label=f"Threshold = {THRESHOLD}°C")
plt.xlabel("Daily mean temperature (°C)")
plt.ylabel("Daily peak demand (GW)")
plt.title("Threshold-linear demand model")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figs/threshold_model.png", dpi=150)
print("\nSaved figs/threshold_model.png")
