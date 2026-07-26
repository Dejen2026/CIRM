import xarray as xr, pandas as pd, numpy as np
import matplotlib.pyplot as plt

D = "data/donnees_cemracs_2026"

# --- Load daily peak demand + daily mean temperature ---
ds = xr.open_dataset(f"{D}/app/eco2mix-demand_nuts1-fr_hour-20132024.nc")
demand = ds["demand"].sum(dim="region").to_series().resample("D").max() / 1000
dt = xr.open_dataset(f"{D}/app/pecd42-t2m_nuts1-fr_hour-20102020.nc")
temp = dt["surface_temperature"].mean(dim="region").to_series().resample("D").mean()

df = pd.DataFrame({"temp": temp, "demand": demand}).dropna()
for lag in range(1, 5):
    df[f"lag{lag}"] = df["temp"].shift(lag)
df = df.dropna()
df = df[df["temp"] < 15].copy()
df["dow"] = df.index.dayofweek

def is_holiday(idx):
    fixed = {(1,1),(5,1),(5,8),(7,14),(8,15),(11,1),(11,11),(12,25)}
    return np.array([(m,d) in fixed for m,d in zip(idx.month, idx.day)], dtype=float)
df["holiday"] = is_holiday(df.index)
df["covid"] = (df.index.year == 2020).astype(float)

def ols(X, y):
    b,*_ = np.linalg.lstsq(X, y, rcond=None); r = y - X@b
    return b, 1 - (r@r)/((y-y.mean())**2).sum()

y = df["demand"].values
dow = np.column_stack([(df["dow"]==d).astype(float) for d in range(1,7)])
temps = np.column_stack([df["temp"].values] + [df[f"lag{l}"].values for l in range(1,5)])

# ---- FIGURE 1: model progression ----
r2s = [0.743, 0.846, 0.892, 0.910]
labels = ["Threshold\nlinear", "+ Calendar", "+ Duration", "+ Holidays\n+ COVID"]
fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.bar(labels, r2s, color=["#c0392b", "#e67e22", "#f1c40f", "#27ae60"])
for b, v in zip(bars, r2s):
    ax.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.3f}", ha="center", fontweight="bold")
ax.set_ylabel("R² (variance explained)")
ax.set_ylim(0.6, 1.0)
ax.axhline(0.95, ls="--", color="gray", alpha=0.7, label="~0.95 (brief target)")
ax.set_title("Model progression — each step adds interpretable structure")
ax.legend()
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("figs/model_progression.png", dpi=150)
print("Saved figs/model_progression.png")

# ---- FINAL MODEL D: fit + predict everywhere ----
XD = np.column_stack([np.ones(len(df)), temps, dow, df["holiday"].values, df["covid"].values])
bD, r2D = ols(XD, y)
df["predicted"] = XD @ bD
print(f"Final model R2 = {r2D:.3f}")

# ---- FIGURE 2: SIMULATION on the Feb-2018 cold wave ----
mask = (df.index >= "2018-02-01") & (df.index <= "2018-03-15")
spell = df[mask]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                gridspec_kw={"height_ratios":[2,1]})
ax1.plot(spell.index, spell["demand"], "o-", color="black", label="Actual demand", lw=1.5)
ax1.plot(spell.index, spell["predicted"], "s--", color="crimson", label="Model prediction", lw=1.5)
ax1.set_ylabel("Daily peak demand (GW)")
ax1.set_title("Simulation: model vs reality during the Feb-2018 cold wave")
ax1.legend(); ax1.grid(alpha=0.3)
ax2.plot(spell.index, spell["temp"], "o-", color="steelblue")
ax2.set_ylabel("Temp (°C)"); ax2.set_xlabel("Date")
ax2.axhline(0, color="gray", lw=0.8); ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figs/simulation_feb2018.png", dpi=150)
print("Saved figs/simulation_feb2018.png")

# how good on the spell specifically?
err = (spell["predicted"] - spell["demand"])
print(f"\nFeb-2018 spell: mean abs error = {err.abs().mean():.2f} GW, "
      f"max error = {err.abs().max():.2f} GW")
