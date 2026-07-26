import xarray as xr
import pandas as pd

D = "data/donnees_cemracs_2026"
ds = xr.open_dataset(f"{D}/app/eco2mix-demand_nuts1-fr_hour-20132024.nc")

# National demand = sum of all regions, each hour
national = ds["demand"].sum(dim="region").to_series()

# Assign each hour to a "winter year" (December belongs to the next year's winter)
df = national.reset_index()
df.columns = ["time", "demand"]
df["winter_year"] = df["time"].dt.year + (df["time"].dt.month == 12).astype(int)

# Find the single peak hour of each winter year
peaks = df.loc[df.groupby("winter_year")["demand"].idxmax()]
peaks = peaks[["winter_year", "time", "demand"]].reset_index(drop=True)
peaks["demand_GW"] = peaks["demand"] / 1000
peaks["hour"] = peaks["time"].dt.hour
peaks["weekday"] = peaks["time"].dt.day_name()

print("=== Annual peak demand (national, hourly) ===")
print(peaks.to_string(index=False))

print("\n=== Which HOUR do peaks occur? ===")
print(peaks["hour"].value_counts().sort_index())
