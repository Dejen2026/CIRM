import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

D = "data/donnees_cemracs_2026"

# --- 1. Load hourly national demand ---
ds = xr.open_dataset(f"{D}/app/eco2mix-demand_nuts1-fr_hour-20132024.nc")
national = ds["demand"].sum(dim="region").to_series()

df = national.reset_index()
df.columns = ["time", "demand"]
df["month"] = df["time"].dt.month
df["dayofweek"] = df["time"].dt.dayofweek
df["hour"] = df["time"].dt.hour

# --- 2. Build "expected" demand from the cycles ---
# Expected = average demand for each (month, weekday, hour) combination.
# This captures the seasonal + weekly + daily cycles all at once.
df["expected"] = df.groupby(["month", "dayofweek", "hour"])["demand"].transform("mean")

# --- 3. Residual = how far actual demand departs from its cyclic norm ---
df["residual"] = df["demand"] - df["expected"]

# --- 4. The biggest positive deviations (demand far ABOVE normal) ---
top = df.nlargest(15, "residual")[["time", "demand", "expected", "residual"]].copy()
top["demand_GW"] = top["demand"] / 1000
top["residual_GW"] = top["residual"] / 1000

print("=== 15 hours where demand most EXCEEDED its cyclic norm ===")
print(top[["time", "demand_GW", "residual_GW"]].to_string(index=False))

# --- 5. Which dates do these belong to? ---
print("\n=== Dates of the biggest positive deviations ===")
top_dates = df.nlargest(200, "residual")["time"].dt.date.value_counts().head(10)
print(top_dates)

# --- 6. Save residuals for the next step (joining with temperature) ---
df[["time", "demand", "expected", "residual"]].to_csv(
    "figs/demand_residuals.csv", index=False
)
print("\nSaved figs/demand_residuals.csv (for temperature comparison next)")
