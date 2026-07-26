import xarray as xr, pandas as pd, numpy as np
import matplotlib.pyplot as plt

D = "data/donnees_cemracs_2026"

ds = xr.open_dataset(f"{D}/app/eco2mix-demand_nuts1-fr_hour-20132024.nc")
demand = ds["demand"].sum(dim="region").to_series().resample("D").max() / 1000
dt = xr.open_dataset(f"{D}/app/pecd42-t2m_nuts1-fr_hour-20102020.nc")
temp = dt["surface_temperature"].mean(dim="region").to_series().resample("D").mean()

df = pd.DataFrame({"temp": temp, "demand": demand}).dropna().sort_index()

def effective_temp(T, a):
    """Exponentially weighted 'thermal memory' temperature.
    theta_t = a*theta_{t-1} + (1-a)*T_t  -- discrete thermal relaxation."""
    theta = np.empty(len(T))
    theta[0] = T[0]
    for i in range(1, len(T)):
        theta[i] = a * theta[i-1] + (1 - a) * T[i]
    return theta

def is_holiday(idx):
    fixed = {(1,1),(5,1),(5,8),(7,14),(8,15),(11,1),(11,11),(12,25)}
    return np.array([(m,d) in fixed for m,d in zip(idx.month, idx.day)], dtype=float)

def ols(X, y):
    b,*_ = np.linalg.lstsq(X, y, rcond=None); r = y - X@b
    return b, 1 - (r@r)/((y-y.mean())**2).sum()

Tv = df["temp"].values

# --- Scan over memory parameter a, find the best thermal time constant ---
print("Scanning thermal memory parameter a:")
print(f"{'a':>6}{'tau (days)':>12}{'R2':>10}{'slope':>10}")
results = []
for a in np.arange(0.0, 0.91, 0.05):
    df["theta"] = effective_temp(Tv, a)
    d = df[df["theta"] < 15].copy()
    d["dow"] = d.index.dayofweek
    d["holiday"] = is_holiday(d.index)
    d["covid"] = (d.index.year == 2020).astype(float)
    dow = np.column_stack([(d["dow"]==j).astype(float) for j in range(1,7)])
    X = np.column_stack([np.ones(len(d)), d["theta"].values, dow,
                         d["holiday"].values, d["covid"].values])
    b, r2 = ols(X, d["demand"].values)
    tau = -1/np.log(a) if a > 0 else 0.0
    results.append((a, tau, r2, b[1]))
    if round(a,2) in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8]:
        print(f"{a:>6.2f}{tau:>12.2f}{r2:>10.4f}{b[1]*1000:>10.0f}")

best = max(results, key=lambda x: x[2])
a_best, tau_best, r2_best, slope_best = best
print(f"\nBEST: a = {a_best:.2f}, thermal time constant tau = {tau_best:.2f} days")
print(f"      R2 = {r2_best:.4f}, thermosensitivity = {slope_best*1000:.0f} MW/degC")
print(f"\nCompare:")
print(f"  Free-lag model (5 params):  R2 = 0.910")
print(f"  Thermal model (1 param a):  R2 = {r2_best:.4f}")

# --- Plot R2 vs tau ---
taus = [r[1] for r in results[1:]]   # skip a=0
r2list = [r[2] for r in results[1:]]
plt.figure(figsize=(9,5.5))
plt.plot(taus, r2list, "o-", color="steelblue")
plt.axvline(tau_best, ls="--", color="crimson", label=f"best τ = {tau_best:.1f} days")
plt.xlabel("Thermal time constant τ (days)")
plt.ylabel("R²")
plt.title("Finding the building thermal memory")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figs/thermal_time_constant.png", dpi=150)
print("\nSaved figs/thermal_time_constant.png")
