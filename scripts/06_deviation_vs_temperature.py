import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

D = "data/donnees_cemracs_2026"

# --- 1. Load demand residuals (actual minus cyclic-expected) ---
res = pd.read_csv("figs/demand_residuals.csv", parse_dates=["time"])

# --- 2. Load hourly temperature, average across the 12 regions ---
dt = xr.open_dataset(f"{D}/app/pecd42-t2m_nuts1-fr_hour-20102020.nc")
temp_national = dt["surface_temperature"].mean(dim="region").to_series()
temp_df = temp_national.reset_index()
temp_df.columns = ["time", "temperature"]

# --- 3. Join demand residuals with temperature on the shared timestamps ---
merged = pd.merge(res, temp_df, on="time", how="inner")
print(f"Overlapping hours: {len(merged)}")
print(f"Time range: {merged['time'].min()} to {merged['time'].max()}")

# --- 4. The key plot: demand deviation vs temperature ---
plt.figure(figsize=(10, 6))
plt.scatter(merged["temperature"], merged["residual"] / 1000,
            s=2, alpha=0.2, color="steelblue")
plt.axhline(0, color="black", lw=1)
plt.xlabel("Temperature (°C)")
plt.ylabel("Demand deviation from cyclic norm (GW)")
plt.title("Demand deviation vs temperature")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figs/deviation_vs_temperature.png", dpi=150)
print("Saved figs/deviation_vs_temperature.png")

# --- 5. Confirm: are the biggest deviations the coldest hours? ---
print("\n=== Temperature on the 15 biggest positive deviations ===")
top = merged.nlargest(15, "residual")[["time", "residual", "temperature"]].copy()
top["residual_GW"] = top["residual"] / 1000
print(top[["time", "residual_GW", "temperature"]].to_string(index=False))

print(f"\nMean temperature overall: {merged['temperature'].mean():.1f} °C")
print(f"Mean temperature on top-100 deviation hours: "
      f"{merged.nlargest(100, 'residual')['temperature'].mean():.1f} °C")
