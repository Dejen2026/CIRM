import xarray as xr, pandas as pd, numpy as np

D = "data/donnees_cemracs_2026"
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

# French public holidays (fixed-date ones + a few movable; good enough to test)
def is_holiday(idx):
    md = list(zip(idx.month, idx.day))
    fixed = {(1,1),(5,1),(5,8),(7,14),(8,15),(11,1),(11,11),(12,25)}
    return np.array([(m,d) in fixed for m,d in md], dtype=float)

df["holiday"] = is_holiday(df.index)
df["covid"]   = (df.index.year == 2020).astype(float)

def ols(X, y):
    b,*_ = np.linalg.lstsq(X, y, rcond=None); r = y - X@b
    return b, 1 - (r@r)/((y-y.mean())**2).sum()

y = df["demand"].values
dow = np.column_stack([(df["dow"]==d).astype(float) for d in range(1,7)])
temps = np.column_stack([df["temp"].values] + [df[f"lag{l}"].values for l in range(1,5)])

# Model C (previous best): temps + calendar
XC = np.column_stack([np.ones(len(df)), temps, dow])
_, r2C = ols(XC, y)

# Model D: + holidays + covid
XD = np.column_stack([np.ones(len(df)), temps, dow, df["holiday"].values, df["covid"].values])
bD, r2D = ols(XD, y)

print(f"Model C (temps + calendar)          : R2 = {r2C:.3f}")
print(f"Model D (+ holidays + covid)        : R2 = {r2D:.3f}")
print(f"\n  Holiday effect: {bD[-2]:+.2f} GW")
print(f"  COVID 2020 effect: {bD[-1]:+.2f} GW")
print(f"  Cumulative thermosensitivity: {sum(bD[1:6])*1000:.0f} MW/degC")
