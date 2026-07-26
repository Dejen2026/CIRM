import xarray as xr, pandas as pd, numpy as np
import matplotlib.pyplot as plt

D = "data/donnees_cemracs_2026"
ds = xr.open_dataset(f"{D}/app/eco2mix-demand_nuts1-fr_hour-20132024.nc")
national = ds["demand"].sum(dim="region").to_series()
demand = national.resample("D").max() / 1000
dt = xr.open_dataset(f"{D}/app/pecd42-t2m_nuts1-fr_hour-20102020.nc")
temp = dt["surface_temperature"].mean(dim="region").to_series().resample("D").mean()

df = pd.DataFrame({"temp": temp, "demand": demand}).dropna()
df = df[df["temp"] < 15].copy()
df["dow"] = df.index.dayofweek

# refit Model B (temp + calendar), get residuals
dow = np.column_stack([(df["dow"]==d).astype(float) for d in range(1,7)])
X = np.column_stack([np.ones(len(df)), df["temp"].values, dow])
beta,*_ = np.linalg.lstsq(X, df["demand"].values, rcond=None)
df["resid"] = df["demand"].values - X @ beta

# DIAGNOSTIC 1: are residuals curved in temperature? (would justify nonlinear)
df["tbin"] = pd.cut(df["temp"], bins=np.arange(-6, 16, 2))
print("=== Mean residual by temperature bin (GW) ===")
print("If these show a systematic U or arch -> nonlinearity is real.")
print("If they're flat near zero -> linear form is fine.\n")
print(df.groupby("tbin", observed=True)["resid"].mean().round(2))

# DIAGNOSTIC 2: how big are residuals on holidays vs normal days?
holidays_approx = ((df.index.month==12)&(df.index.day.isin([25,26]))) | \
                  ((df.index.month==1)&(df.index.day==1)) | \
                  ((df.index.month==5)&(df.index.day.isin([1,8])))
print("\n=== Residual on approx holidays vs normal ===")
print(f"Normal days : {df.loc[~holidays_approx,'resid'].mean():+.2f} GW")
print(f"Holidays    : {df.loc[holidays_approx,'resid'].mean():+.2f} GW")

# DIAGNOSTIC 3: yearly trend in residuals?
print("\n=== Mean residual by year (GW) ===")
print(df.groupby(df.index.year)["resid"].mean().round(2))
