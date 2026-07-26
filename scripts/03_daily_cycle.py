import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

D = "data/donnees_cemracs_2026"
ds = xr.open_dataset(f"{D}/app/eco2mix-demand_nuts1-fr_hour-20132024.nc")

national = ds["demand"].sum(dim="region").to_series()
df = national.reset_index()
df.columns = ["time", "demand"]
df["hour"] = df["time"].dt.hour
df["month"] = df["time"].dt.month

# Winter = Dec/Jan/Feb, Summer = Jun/Jul/Aug
winter = df[df["month"].isin([12, 1, 2])]
summer = df[df["month"].isin([6, 7, 8])]

winter_cycle = winter.groupby("hour")["demand"].mean() / 1000
summer_cycle = summer.groupby("hour")["demand"].mean() / 1000

plt.figure(figsize=(10, 6))
plt.plot(winter_cycle.index, winter_cycle.values, "o-", color="steelblue", label="Winter (DJF)")
plt.plot(summer_cycle.index, summer_cycle.values, "o-", color="firebrick", label="Summer (JJA)")
plt.xlabel("Hour of day")
plt.ylabel("Average national demand (GW)")
plt.title("Average daily demand cycle: winter vs summer")
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(range(0, 24, 2))
plt.tight_layout()
plt.savefig("figs/daily_cycle_winter_summer.png", dpi=150)
print("Saved figs/daily_cycle_winter_summer.png")
