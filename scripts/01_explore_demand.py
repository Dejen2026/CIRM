import xarray as xr

D = "data/donnees_cemracs_2026"

# Load the hourly demand data
ds = xr.open_dataset(f"{D}/app/eco2mix-demand_nuts1-fr_hour-20132024.nc")

print("=== Dataset overview ===")
print(ds)

print("\n=== Regions ===")
print(list(ds.region.values))

print("\n=== Time range ===")
print("Start:", ds.time.values[0])
print("End:  ", ds.time.values[-1])
print("Number of hours:", ds.time.size)

# National demand = sum across all regions, each hour
national = ds["demand"].sum(dim="region")
print("\n=== National hourly demand (MW) ===")
print("Max :", float(national.max()))
print("Min :", float(national.min()))
print("Mean:", float(national.mean()))
