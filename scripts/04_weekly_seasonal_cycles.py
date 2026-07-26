import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

D = "data/donnees_cemracs_2026"
ds = xr.open_dataset(f"{D}/app/eco2mix-demand_nuts1-fr_hour-20132024.nc")

national = ds["demand"].sum(dim="region").to_series()
df = national.reset_index()
df.columns = ["time", "demand"]
df["dayofweek"] = df["time"].dt.dayofweek      # 0 = Monday, 6 = Sunday
df["month"] = df["time"].dt.month

# --- Weekly cycle ---
weekly = df.groupby("dayofweek")["demand"].mean() / 1000
day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# --- Seasonal cycle ---
seasonal = df.groupby("month")["demand"].mean() / 1000
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# --- Plot both side by side ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(range(7), weekly.values, color="steelblue")
axes[0].set_xticks(range(7))
axes[0].set_xticklabels(day_labels)
axes[0].set_ylabel("Average national demand (GW)")
axes[0].set_title("Weekly cycle")
axes[0].grid(alpha=0.3, axis="y")

axes[1].plot(range(1, 13), seasonal.values, "o-", color="firebrick")
axes[1].set_xticks(range(1, 13))
axes[1].set_xticklabels(month_labels)
axes[1].set_ylabel("Average national demand (GW)")
axes[1].set_title("Seasonal cycle")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figs/weekly_seasonal_cycles.png", dpi=150)
print("Saved figs/weekly_seasonal_cycles.png")

print("\n=== Weekly averages (GW) ===")
for i, d in enumerate(day_labels):
    print(f"  {d}: {weekly.values[i]:.1f}")

print("\n=== Monthly averages (GW) ===")
for i, m in enumerate(month_labels):
    print(f"  {m}: {seasonal.values[i]:.1f}")
