#!/juno/opt/anaconda/3-2022.10/bin/python
import os
import glob
import argparse
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, DateFormatter
import numpy as np
import pandas as pd


# ----------------------------
# Configuration (defaults; can be overridden by CLI)
# ----------------------------
IN_DIR = "./data/out"
OUT_DIR = "./data/figures_out"
START_YEAR = 2002
END_YEAR = 2022
PLOT_MODE = "all"
# PLOT_MODE = "single"
SELECTED_REGION = "Global"
# SELECTED_REGION = "South American Tropical"


CSV_V_COL = "NEE_VUT_REF"


# ----------------------------
# Helpers
# ----------------------------
def to_minus180_180(lon):
    return (lon + 180) % 360 - 180

def normalize_longitudes(ds, lon_name="lon"):
    lon = ds[lon_name]
    # If longitudes are [0, 360), map to [-180, 180) and sort
    if (lon.max() > 180).item() or (lon.min() >= 0).item():
        lon_new = to_minus180_180(lon)
        ds = ds.assign_coords({lon_name: lon_new})
        ds = ds.sortby(lon_name)
    return ds

def compute_cell_areas_km2(lat, lon):
    """
    Compute 2D cell area (km^2) for a latitude-longitude grid using spherical geometry.
    lat: 1D [nlat] (degrees)
    lon: 1D [nlon] (degrees)
    Returns: 2D array [nlat, nlon] of cell areas in km^2
    """
    R = 6371.0  # Earth radius in km
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    # Build edges via midpoints; for ends, assume half-spacing
    def edges_from_centers(c):
        dc = np.diff(c)
        dc_first = dc[0] if dc.size > 0 else 1.0
        dc_last = dc[-1] if dc.size > 0 else 1.0
        left = c[0] - 0.5 * dc_first
        right = c[-1] + 0.5 * dc_last
        mid = c[:-1] + 0.5 * dc
        return np.concatenate([[left], mid, [right]])

    lat_e = edges_from_centers(lat)
    lon_e = edges_from_centers(lon)
    # Clip latitude edges to the poles
    lat_e = np.clip(lat_e, -90.0, 90.0)

    # Convert to radians
    lat_e_rad = np.deg2rad(lat_e)
    lon_e_rad = np.deg2rad(lon_e)

    # Cell areas: R^2 * dlon * (sin(lat_north) - sin(lat_south))
    dlon = np.diff(lon_e_rad)[None, :]  # [1, nlon]
    sin_lat = np.sin(lat_e_rad)
    dphi = (sin_lat[1:] - sin_lat[:-1])[:, None]  # [nlat, 1]

    area = (R**2) * (dlon * dphi)  # [nlat, nlon] in km^2
    return area.astype(np.float32)

def safe_sel_box(da, lat_min, lat_max, lon_min, lon_max, lat_name="lat", lon_name="lon"):
    lat = da[lat_name]
    # Handle ascending or descending latitude
    if lat[0] < lat[-1]:
        lat_slice = slice(lat_min, lat_max)
    else:
        lat_slice = slice(lat_max, lat_min)
    return da.sel({lat_name: lat_slice, lon_name: slice(lon_min, lon_max)})


def get_fill_value(da):
    if isinstance(da, xr.DataArray):
        attrs = da.attrs
    else:
        attrs = getattr(da, 'attrs', {})
    if "_FillValue" in attrs:
        return attrs["_FillValue"]
    if "missing_value" in attrs:
        return attrs["missing_value"]
    return 1e36



member_ids = [f"{i:04d}" for i in range(1, 31)]  # 0001..0030
# Merged region metadata: bounds, representative site, and default FluxNET CSV
REGIONS = {
    "Global": {
        "bounds": (-90, 90, -180, 180),
        "site": {"SITE_ID": "GH-Ank", "LOCATION_LAT": 5.2685, "LOCATION_LONG": -2.6942},
        "csv_file": "./data/FluxNET/FLX_GH-Ank_FLUXNET2015_SUBSET_MM_2011-2014_1-4.csv",
    },
    "North American Boreal": {
        "bounds": (50, 70, -170, -50),
        "site": {"SITE_ID": "CA-SF1", "LOCATION_LAT": 54.485, "LOCATION_LONG": -105.8176},
        "csv_file": "./data/FluxNET/FLX_CA-SF1_FLUXNET2015_SUBSET_MM_2003-2006_1-4.csv",
    },
    "North American Temperate": {
        "bounds": (30, 50, -130, -60),
        "site": {"SITE_ID": "US-Ne3", "LOCATION_LAT": 41.1797, "LOCATION_LONG": -96.4397},
        "csv_file": "./data/FluxNET/FLX_US-Ne3_FLUXNET2015_SUBSET_MM_2001-2013_1-4.csv",
    },
    "South American Tropical": {
        "bounds": (-20, 10, -80, -35),
        "site": {"SITE_ID": "BR-Sa3", "LOCATION_LAT": -3.018, "LOCATION_LONG": -54.9714},
        "csv_file": "./data/FluxNET/FLX_BR-Sa3_FLUXNET2015_SUBSET_MM_2000-2004_1-4.csv",
    },
    "South American Temperate": {
        "bounds": (-40, -20, -70, -50),
        "site": {"SITE_ID": "AR-Vir", "LOCATION_LAT": -28.2395, "LOCATION_LONG": -56.1886},
        "csv_file": "./data/FluxNET/FLX_AR-Vir_FLUXNET2015_SUBSET_MM_2009-2012_1-4.csv",
    },
    "Northern Africa": {
        "bounds": (10, 30, -20, 30),
        "site": {"SITE_ID": "SN-Dhr", "LOCATION_LAT": 15.4028, "LOCATION_LONG": -15.4322},
        "csv_file": "./data/FluxNET/FLX_SN-Dhr_FLUXNET2015_SUBSET_MM_2010-2013_1-4.csv",
    },
    "Southern Africa": {
        "bounds": (-35, -15, 10, 40),
        "site": {"SITE_ID": "ZM-Mon", "LOCATION_LAT": -15.4391, "LOCATION_LONG": 23.2525},
        "csv_file": "./data/FluxNET/FLX_ZM-Mon_FLUXNET2015_SUBSET_MM_2000-2009_2-4.csv",
    },
    "Eurasian Boreal": {
        "bounds": (50, 70, 10, 180),
        "site": {"SITE_ID": "RU-Ha1", "LOCATION_LAT": 54.7252, "LOCATION_LONG": 90.0022},
        "csv_file": "./data/FluxNET/FLX_RU-Ha1_FLUXNET2015_SUBSET_MM_2002-2004_1-4.csv",
    },
    "Eurasian Temperate": {
        "bounds": (30, 50, -10, 180),
        "site": {"SITE_ID": "CN-Dan", "LOCATION_LAT": 30.4978, "LOCATION_LONG": 91.0664},
        "csv_file": "./data/FluxNET/FLX_CN-Dan_FLUXNET2015_SUBSET_MM_2004-2005_1-4.csv",
    },
    "Tropical Asia": {
        "bounds": (-10, 20, 60, 120),
        "site": {"SITE_ID": "MY-PSO", "LOCATION_LAT": 2.973, "LOCATION_LONG": 102.3062},
        "csv_file": "./data/FluxNET/FLX_MY-PSO_FLUXNET2015_SUBSET_MM_2003-2009_1-4.csv",
    },
    "Australia": {
        "bounds": (-45, -10, 110, 155),
        "site": {"SITE_ID": "AU-ASM", "LOCATION_LAT": -22.283, "LOCATION_LONG": 133.249},
        "csv_file": "./data/FluxNET/FLX_AU-ASM_FLUXNET2015_SUBSET_MM_2010-2014_2-4.csv",
    },
    "Europe": {
        "bounds": (35, 70, -10, 40),
        "site": {"SITE_ID": "DE-Spw", "LOCATION_LAT": 51.8922, "LOCATION_LONG": 14.0337},
        "csv_file": "./data/FluxNET/FLX_DE-Spw_FLUXNET2015_SUBSET_MM_2010-2014_1-4.csv",
    },
}


# ----------------------------
# CLI arguments
# ----------------------------
parser = argparse.ArgumentParser(
    description=(
        "Plot monthly NEE regional time series: either all 12 regions (subplots) "
        "or a single region occupying the full figure, with optional CSV overlay."
    )
)
parser.add_argument("--in-dir", default=IN_DIR, help="Input directory for NEE_monthmean_*.nc files")
parser.add_argument("--out-dir", default=OUT_DIR, help="Output directory for figures")
parser.add_argument("--start-year", type=int, default=START_YEAR, help="Start year (inclusive)")
parser.add_argument("--end-year", type=int, default=END_YEAR, help="End year (inclusive)")
parser.add_argument(
    "--mode", choices=["all", "single"], default=PLOT_MODE,
    help="Plot mode: 'all' = 12 subplots, 'single' = one region full-figure"
)
parser.add_argument("--region", default=SELECTED_REGION, help="Region name to plot in single mode")
parser.add_argument("--fluxnet-csv", default=None, help="Optional CSV file to overlay in single mode; if not provided, uses the region's default FluxNET CSV")
parser.add_argument("--fluxnet-timestamp-col", default="TIMESTAMP", help="CSV column with YYYYMM timestamps")
parser.add_argument("--fluxnet-value-col", default=CSV_V_COL, help="CSV column with NEE values")
parser.add_argument("--fluxnet-timestamp-fmt", default="%Y%m", help="Timestamp format for CSV (default %%Y%%m)")
args = parser.parse_args()

IN_DIR = args.in_dir
OUT_DIR = args.out_dir
START_YEAR = args.start_year
END_YEAR = args.end_year
PLOT_MODE = args.mode
SELECTED_REGION = args.region
CSV_V_COL = args.fluxnet_value_col

CSV_T_COL = args.fluxnet_timestamp_col
CSV_T_FMT = args.fluxnet_timestamp_fmt

os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# Read MEAN and ensemble MEMBER files per month and compute regional means
# ----------------------------

# Storage
time_list = []
region_mean_series = {name: [] for name in REGIONS}
region_member_series = {name: {mid: [] for mid in member_ids} for name in REGIONS}

area = None  # area weights (km^2)

for year in range(START_YEAR, END_YEAR + 1):
    for month in range(1, 13):
        mean_path = os.path.join(IN_DIR, f"NEE_monthmean_{year}_{month:02d}_mean.nc")

        found_any = False
        ds_ref = None  # for grid/area detection

        ds_mean = None
        if os.path.exists(mean_path):
            print(f"Opening MEAN file: {mean_path}")
            ds_mean = xr.open_dataset(mean_path, decode_times=False)
            print("Opened OK !!!")
            ds_mean = normalize_longitudes(ds_mean, lon_name="lon")
            ds_ref = ds_mean
            found_any = True

        # Ensure we know if at least one member exists and get a ref grid if needed
        has_any_member = False
        for mid in member_ids:
            member_path = os.path.join(IN_DIR, f"NEE_monthmean_{year}_{month:02d}_{mid}.nc")
            if os.path.exists(member_path):
                has_any_member = True
                if ds_ref is None:
                    print(f"Opening MEMBER file for reference grid: {member_path}")
                    ds_tmp = xr.open_dataset(member_path, decode_times=False)
                    print("Opened OK !!!")
                    ds_ref = normalize_longitudes(ds_tmp, lon_name="lon")
                    ds_tmp.close()
        found_any = found_any or has_any_member

        if not found_any:
            continue  # nothing to process for this month

        # Build area weights once
        if area is None and ds_ref is not None:
            lat = ds_ref["lat"].values
            lon = ds_ref["lon"].values
            area2d = compute_cell_areas_km2(lat, lon)
            area = xr.DataArray(area2d, coords={"lat": ds_ref["lat"], "lon": ds_ref["lon"]}, dims=("lat", "lon"))

        # Use calendar month as time index
        tval = pd.to_datetime(f"{year}-{month:02d}-01")
        time_list.append(tval)

        # Compute region means for MEAN file (if available), else fill with NaN for now
        if ds_mean is not None:
            var_mean = ds_mean["NEE"]
            fv_mean = get_fill_value(var_mean)
            for name, info in REGIONS.items():
                lat_min, lat_max, lon_min, lon_max = info["bounds"]
                subset = safe_sel_box(var_mean, lat_min, lat_max, lon_min, lon_max)
                area_subset = safe_sel_box(area, lat_min, lat_max, lon_min, lon_max)
                subset_masked = subset.where(subset != fv_mean)
                weights = area_subset.where(subset_masked.notnull())
                num = (subset_masked * weights).sum(dim=("lat", "lon"))
                den = weights.sum(dim=("lat", "lon"))
                region_mean_series[name].append((num / den).item())
        else:
            for name in REGIONS:
                region_mean_series[name].append(np.nan)

        # Compute region means for each MEMBER (append NaN if missing)
        for mid in member_ids:
            member_path = os.path.join(IN_DIR, f"NEE_monthmean_{year}_{month:02d}_{mid}.nc")
            print(f"Processing MEMBER file: {member_path}")
            if os.path.exists(member_path):
                ds_m = xr.open_dataset(member_path, decode_times=False)
                ds_m = normalize_longitudes(ds_m, lon_name="lon")
                var_m = ds_m["NEE"]
                fv_m = get_fill_value(var_m)
                for name, info in REGIONS.items():
                    lat_min, lat_max, lon_min, lon_max = info["bounds"]
                    subset = safe_sel_box(var_m, lat_min, lat_max, lon_min, lon_max)
                    area_subset = safe_sel_box(area, lat_min, lat_max, lon_min, lon_max)
                    subset_masked = subset.where(subset != fv_m)
                    weights = area_subset.where(subset_masked.notnull())
                    num = (subset_masked * weights).sum(dim=("lat", "lon"))
                    den = weights.sum(dim=("lat", "lon"))
                    region_member_series[name][mid].append((num / den).item())
                ds_m.close()
            else:
                for name in REGIONS:
                    region_member_series[name][mid].append(np.nan)

        # If MEAN missing, backfill with mean across available members for this month
        if ds_mean is None:
            for name in REGIONS:
                vals = np.array([region_member_series[name][mid][-1] for mid in member_ids], dtype=float)
                mean_val = np.nanmean(vals) if np.isfinite(vals).any() else np.nan
                region_mean_series[name][-1] = mean_val

        if ds_mean is not None:
            ds_mean.close()

# ----------------------------
# Plot
# ----------------------------
x_time_plot = pd.to_datetime(time_list)

if PLOT_MODE == "all":
    fig, axes = plt.subplots(3, 4, figsize=(20, 12), constrained_layout=True)
    axes = axes.flatten()

    for i, name in enumerate(REGIONS.keys()):
        ax = axes[i]

        # Prepare data arrays
        mean_series = np.array(region_mean_series[name], dtype=float)
        members_arr = np.stack([region_member_series[name][mid] for mid in member_ids], axis=1)  # [ntime, nmembers]

        # Uncertainty shading: min..max across members
        low = np.nanmin(members_arr, axis=1)
        high = np.nanmax(members_arr, axis=1)
        ax.fill_between(x_time_plot, low, high, color="tab:blue", alpha=0.15, label="Member range")

        # Plot member lines (light, transparent)
        for j in range(members_arr.shape[1]):
            ax.plot(x_time_plot, members_arr[:, j], color="gray", alpha=0.15, linewidth=0.7)

        # Plot MEAN line (main)
        ax.plot(x_time_plot, mean_series, color="tab:blue", linewidth=1.8, label="MEAN")

        # Optional overlay: CSV time series for this region (from REGIONS dict)
        auto_csv = REGIONS[name].get("csv_file")
        if auto_csv is not None and os.path.exists(auto_csv):
            try:
                df_csv = pd.read_csv(auto_csv)
                ts_vals = pd.to_datetime(df_csv[CSV_T_COL].astype(str), format=CSV_T_FMT, errors="coerce")
                y_vals = pd.to_numeric(df_csv[CSV_V_COL], errors="coerce")
                mask = ts_vals.notna() & y_vals.notna()
                ts_vals = ts_vals[mask]
                y_vals = y_vals[mask]
                ax.plot(ts_vals, y_vals, color="tab:red", linewidth=1.6, label="Reference CSV")
            except Exception as e:
                print(f"Warning: failed to overlay CSV '{auto_csv}' for region '{name}': {e}")

        ax.set_title(name)
        ax.set_xlabel("Year")
        ax.set_ylabel("NEE (gC/m²/day)")
        # ax.set_ylim(-2, 2)
        ax.xaxis.set_major_locator(YearLocator(base=5))
        ax.xaxis.set_major_formatter(DateFormatter("%Y"))
        # Minor ticks: every year
        ax.xaxis.set_minor_locator(YearLocator(1))

        # Grid
        ax.grid(True, which="major", axis="both")
        ax.grid(True, which="minor", axis="x", linestyle=":", alpha=0.6)

    # Hide any empty subplot if regions < 12
    for j in range(len(REGIONS), len(axes)):
        axes[j].axis("off")

    # Put a single legend in the first axis to avoid clutter
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(loc="upper right", frameon=False)

    fig.suptitle(f"Monthly NEE (MEAN and Ensemble Uncertainty) {START_YEAR}-{END_YEAR}", fontsize=16)
    out_png = os.path.join(OUT_DIR, f"NEE_monthly_means_{START_YEAR}_{END_YEAR}.png")
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Saved {out_png}")

elif PLOT_MODE == "single":
    region_name = SELECTED_REGION
    if region_name not in REGIONS:
        valid = ", ".join(REGIONS.keys())
        raise SystemExit(f"Region '{region_name}' not found. Valid options: {valid}")

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

    # Prepare data arrays for selected region
    mean_series = np.array(region_mean_series[region_name], dtype=float)
    members_arr = np.stack([region_member_series[region_name][mid] for mid in member_ids], axis=1)  # [ntime, nmembers]

    # Uncertainty shading: min..max across members
    low = np.nanmin(members_arr, axis=1)
    high = np.nanmax(members_arr, axis=1)
    ax.fill_between(x_time_plot, low, high, color="tab:blue", alpha=0.15, label="Member range")

    # Plot member lines (light, transparent)
    for j in range(members_arr.shape[1]):
        ax.plot(x_time_plot, members_arr[:, j], color="gray", alpha=0.15, linewidth=0.7)

    # Plot MEAN line (main)
    ax.plot(x_time_plot, mean_series, color="tab:blue", linewidth=1.8, label="MEAN")

    # Optional overlay: CSV time series (auto-select by region unless overridden)
    auto_csv = REGIONS[region_name].get("csv_file")
    csv_to_use = args.fluxnet_csv if args.fluxnet_csv else auto_csv
    if csv_to_use is not None and os.path.exists(csv_to_use):
        try:
            df_csv = pd.read_csv(csv_to_use)
            ts_vals = pd.to_datetime(df_csv[CSV_T_COL].astype(str), format=CSV_T_FMT, errors="coerce")
            y_vals = pd.to_numeric(df_csv[CSV_V_COL], errors="coerce")
            mask = ts_vals.notna() & y_vals.notna()
            ts_vals = ts_vals[mask]
            y_vals = y_vals[mask]
            ax.plot(ts_vals, y_vals, color="tab:red", linewidth=1.6, label="Reference CSV")
        except Exception as e:
            print(f"Warning: failed to overlay CSV '{csv_to_use}': {e}")

    ax.set_title(f"{region_name}")
    ax.set_xlabel("Year")
    ax.set_ylabel("NEE (gC/m²/day)")
    # ax.set_ylim(-2, 2)
    ax.xaxis.set_major_locator(YearLocator(base=5))
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(YearLocator(1))
    ax.grid(True, which="major", axis="both")
    ax.grid(True, which="minor", axis="x", linestyle=":", alpha=0.6)

    ax.legend(loc="upper right", frameon=False)

    safe_region = region_name.replace(" ", "_").replace("/", "-")
    out_png = os.path.join(OUT_DIR, f"NEE_monthly_{CSV_V_COL}_{safe_region}_{START_YEAR}_{END_YEAR}.png")
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Saved {out_png}")

else:
    raise SystemExit(f"Unknown mode: {PLOT_MODE}")