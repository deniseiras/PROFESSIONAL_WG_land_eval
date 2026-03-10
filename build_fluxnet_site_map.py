#!/usr/bin/env python3
import os
import sys
import json
import argparse
import pandas as pd
import numpy as np

# ----------------------------
# Configuration
# ----------------------------
FLUXNET_DIR = "./data/FluxNET"
DEFAULT_SITES_CSV = os.path.join(FLUXNET_DIR, "FluxNET_sites.csv")

# Define regions: (lat_min, lat_max, lon_min, lon_max)
regions = {
    "Global": (-90, 90, -180, 180),
    "North American Boreal": (50, 70, -170, -50),
    "North American Temperate": (30, 50, -130, -60),
    "South American Tropical": (-20, 10, -80, -35),
    "South American Temperate": (-40, -20, -70, -50),
    "Northern Africa": (10, 30, -20, 30),
    "Southern Africa": (-35, -15, 10, 40),
    "Eurasian Boreal": (50, 70, 10, 180),
    "Eurasian Temperate": (30, 50, -10, 180),
    "Tropical Asia": (-10, 20, 60, 120),
    "Australia": (-45, -10, 110, 155),
    "Europe": (35, 70, -10, 40),
}

REQUIRED_COLS = ["SITE_ID", "LOCATION_LAT", "LOCATION_LONG"]


def _validate_columns(df: pd.DataFrame, csv_path: str):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Missing required columns in {csv_path}: {missing}. "
            f"Expected at least: {REQUIRED_COLS}"
        )


def _clean_lat_lon(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["LOCATION_LAT"] = pd.to_numeric(df["LOCATION_LAT"], errors="coerce")
    df["LOCATION_LONG"] = pd.to_numeric(df["LOCATION_LONG"], errors="coerce")
    return df.dropna(subset=["LOCATION_LAT", "LOCATION_LONG"])  # keep only valid rows


def _haversine_min_index(pool_lats: np.ndarray, pool_lons: np.ndarray, center_lat: float, center_lon: float) -> int:
    """Return index of the minimum great-circle distance to the center."""
    lat1 = np.deg2rad(center_lat)
    lat2 = np.deg2rad(pool_lats)
    dlon = np.deg2rad(pool_lons - center_lon)
    a = np.sin((lat2 - lat1) / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    dist = 2.0 * np.arcsin(np.sqrt(a))
    return int(np.nanargmin(dist))


def select_best_site_for_region(df_sites: pd.DataFrame, region_name: str) -> pd.Series:
    if region_name not in regions:
        valid = ", ".join(regions.keys())
        raise SystemExit(f"Region '{region_name}' not found. Valid options: {valid}")

    lat_min, lat_max, lon_min, lon_max = regions[region_name]
    lat_c = 0.5 * (lat_min + lat_max)
    lon_c = 0.5 * (lon_min + lon_max)

    dfv = _clean_lat_lon(df_sites)

    inside = dfv[(dfv["LOCATION_LAT"].between(lat_min, lat_max)) & (dfv["LOCATION_LONG"].between(lon_min, lon_max))]
    pool = inside if not inside.empty else dfv

    idx = _haversine_min_index(pool["LOCATION_LAT"].values, pool["LOCATION_LONG"].values, lat_c, lon_c)
    return pool.iloc[idx]


def build_siteid_to_filename(site_ids) -> dict:
    """Return mapping SITE_ID -> template filename (placeholders for periods)."""
    mapping = {}
    for sid in site_ids:
        sid_str = str(sid)
        filename = f"FLX_{sid_str}_FLUXNET2015_SUBSET_MM_YYYY_ini-YYYY_end_M-N.csv"
        mapping[sid_str] = os.path.join(FLUXNET_DIR, filename)
    return mapping


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a dictionary mapping FluxNET SITE_ID to a CSV filename template "
            "(YYYY_ini-YYYY_end_M-N placeholders), selecting the best site for a region "
            "or for all regions."
        )
    )
    parser.add_argument("--sites-csv", default=DEFAULT_SITES_CSV, help="Path to FluxNET_sites.csv")
    parser.add_argument("--region", default="South American Tropical", help="Region name to select (ignored if --all-regions)")
    parser.add_argument("--all-regions", action="store_true", help="If set, compute best site for each predefined region")
    parser.add_argument("--out-json", default=None, help="Optional path to write JSON mapping. If omitted, prints to stdout.")
    args = parser.parse_args()

    if not os.path.exists(args.sites_csv):
        raise SystemExit(f"Sites CSV not found: {args.sites_csv}")

    df_sites = pd.read_csv(args.sites_csv)
    _validate_columns(df_sites, args.sites_csv)

    if args.all_regions:
        site_ids = []
        region_to_site = {}
        for rname in regions.keys():
            best = select_best_site_for_region(df_sites, rname)
            sid = str(best["SITE_ID"])
            region_to_site[rname] = {
                "SITE_ID": sid,
                "LOCATION_LAT": float(best["LOCATION_LAT"]),
                "LOCATION_LONG": float(best["LOCATION_LONG"]),
            }
            if sid not in site_ids:
                site_ids.append(sid)
        mapping = build_siteid_to_filename(site_ids)
        payload = {
            "region_to_site": region_to_site,
            "site_id_to_file": mapping,
        }
    else:
        best = select_best_site_for_region(df_sites, args.region)
        sid = str(best["SITE_ID"])
        mapping = build_siteid_to_filename([sid])
        payload = {
            "selected_region": args.region,
            "selected_site": {
                "SITE_ID": sid,
                "LOCATION_LAT": float(best["LOCATION_LAT"]),
                "LOCATION_LONG": float(best["LOCATION_LONG"]),
            },
            "site_id_to_file": mapping,
        }

    if args.out_json:
        out_dir = os.path.dirname(os.path.abspath(args.out_json))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote mapping JSON to {args.out_json}")
    else:
        json.dump(payload, sys.stdout, indent=2)
        print()


if __name__ == "__main__":
    main()
