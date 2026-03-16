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
DEFAULT_SITES_INFO = os.path.join(FLUXNET_DIR, "FluxNET_sites_info.csv")

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
REQUIRED_INFO_COLS = ["SITE_ID", "filename", "filetype"]


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


def select_best_site_for_region(df_sites: pd.DataFrame, region_name: str, allowed_site_ids=None) -> pd.Series:
    if region_name not in regions:
        valid = ", ".join(regions.keys())
        raise SystemExit(f"Region '{region_name}' not found. Valid options: {valid}")

    lat_min, lat_max, lon_min, lon_max = regions[region_name]
    lat_c = 0.5 * (lat_min + lat_max)
    lon_c = 0.5 * (lon_min + lon_max)

    dfv = _clean_lat_lon(df_sites)

    # Restrict to allowed SITE_IDs if provided
    if allowed_site_ids is not None:
        dfv = dfv[dfv["SITE_ID"].astype(str).isin(allowed_site_ids)]
        if dfv.empty:
            raise SystemExit("No candidate sites available after filtering by sites_info and subset/filetype.")

    inside = dfv[(dfv["LOCATION_LAT"].between(lat_min, lat_max)) & (dfv["LOCATION_LONG"].between(lon_min, lon_max))]
    pool = inside if not inside.empty else dfv

    idx = _haversine_min_index(pool["LOCATION_LAT"].values, pool["LOCATION_LONG"].values, lat_c, lon_c)
    return pool.iloc[idx]


def choose_best_info_rows(df_info: pd.DataFrame) -> pd.DataFrame:
    """
    For each SITE_ID in df_info, pick the single best row.
    Preference: longest coverage (end_year - start_year), then most recent timestamp.
    Assumes df_info has columns: SITE_ID, filename, start_year, end_year, timestamp (optional).
    """
    work = df_info.copy()
    # Ensure numeric types where present
    for c in ["start_year", "end_year"]:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")
    # Normalize timestamp column name and content
    cols_norm = {c: c.strip() for c in work.columns}
    work.rename(columns=cols_norm, inplace=True)
    if "timestamp" in work.columns:
        work["timestamp"] = pd.to_numeric(work["timestamp"].astype(str).str.strip(), errors="coerce")
    else:
        work["timestamp"] = np.nan

    # Coverage score
    if {"start_year", "end_year"}.issubset(work.columns):
        coverage = (work["end_year"].fillna(-np.inf) - work["start_year"].fillna(np.inf)).astype(float)
    else:
        coverage = pd.Series(np.zeros(len(work)), index=work.index)
    work["_coverage"] = coverage

    # Rank: higher coverage first, then newer timestamp
    work.sort_values(by=["SITE_ID", "_coverage", "timestamp"], ascending=[True, False, False], inplace=True)

    # Keep first per SITE_ID
    best = work.drop_duplicates(subset=["SITE_ID"], keep="first")
    return best


def build_siteid_to_filename_from_info(df_info_best: pd.DataFrame) -> dict:
    """Return mapping SITE_ID -> full path filename using 'filename' column in df_info_best."""
    mapping = {}
    for _, row in df_info_best.iterrows():
        sid_str = str(row["SITE_ID"]).strip()
        fname = str(row["filename"]).strip()
        mapping[sid_str] = os.path.join(FLUXNET_DIR, fname)
    return mapping


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a dictionary mapping FluxNET SITE_ID to an existing FLUX-MET filename "
            "from FluxNET_sites_info.csv (filtered by subset level), selecting the best site "
            "for a region, for all regions, or returning every matching FluxNET site."
        )
    )
    parser.add_argument("--sites-csv", default=DEFAULT_SITES_CSV, help="Path to FluxNET_sites.csv (with locations)")
    parser.add_argument("--sites-info", default=DEFAULT_SITES_INFO, help="Path to FluxNET_sites_info.csv (with filenames).")
    parser.add_argument("--subset", default="MM", choices=["YY", "MM", "WW", "DD", "HH", "HR"], help="Desired temporal subset to use (default MM)")
    parser.add_argument("--region", default="South American Tropical", help="Region name to select (ignored if --all-regions or --all-sites)")
    parser.add_argument("--all-regions", action="store_true", help="If set, compute best site for each predefined region")
    parser.add_argument("--all-sites", action="store_true", help="If set, include every FluxNET site that matches the filters")
    parser.add_argument("--out-json", default=None, help="Optional path to write JSON mapping. If omitted, prints to stdout.")
    args = parser.parse_args()


   
    sites_info_path = args.sites_info

    if args.all_sites and args.all_regions:
        raise SystemExit("--all-sites cannot be used with --all-regions")

    if not os.path.exists(args.sites_csv):
        raise SystemExit(f"Sites CSV not found: {args.sites_csv}")
    if not os.path.exists(sites_info_path):
        raise SystemExit(f"Sites info CSV not found: {sites_info_path}")

    df_sites = pd.read_csv(args.sites_csv)
    # Load and validate
    _validate_columns(df_sites, args.sites_csv)

    df_info = pd.read_csv(sites_info_path)
    # Normalize column names (strip spaces)
    df_info.columns = [c.strip() for c in df_info.columns]
    missing_info = [c for c in REQUIRED_INFO_COLS if c not in df_info.columns]
    if missing_info:
        raise SystemExit(
            f"Missing required columns in {sites_info_path}: {missing_info}. "
            f"Expected at least: {REQUIRED_INFO_COLS}"
        )

    # Filter info: only FLUX-MET and desired subset level (by filename pattern)
    subset_tag = f"_SUBSET_{args.subset.upper()}_"
    mask_type = (df_info["filetype"].astype(str).str.strip() == "FLUX-MET")
    mask_subset = df_info["filename"].astype(str).str.contains(subset_tag, regex=False)
    df_info_filt = df_info[mask_type & mask_subset].copy()

    if df_info_filt.empty:
        raise SystemExit(
            f"No entries found in sites_info for filetype=FLUX-MET and subset={args.subset}."
        )

    # Allowed SITE_IDs based on info
    allowed_site_ids = set(df_info_filt["SITE_ID"].astype(str).unique())

    # Restrict sites table to allowed SITE_IDs
    df_sites_allowed = df_sites[df_sites["SITE_ID"].astype(str).isin(allowed_site_ids)].copy()
    if df_sites_allowed.empty:
        raise SystemExit("No SITE_IDs in FluxNET_sites.csv match those in FluxNET_sites_info.csv after filtering.")

    if args.all_sites:
        df_locations = _clean_lat_lon(df_sites_allowed)
        df_locations["SITE_ID"] = df_locations["SITE_ID"].astype(str).str.strip()
        df_locations = df_locations.dropna(subset=["SITE_ID"])
        df_locations = df_locations.drop_duplicates(subset=["SITE_ID"]).sort_values(by="SITE_ID")
        region_sites = {}
        for rname, bounds in regions.items():
            if rname == "Global":
                continue
            lat_min, lat_max, lon_min, lon_max = bounds
            inside = df_locations[
                (df_locations["LOCATION_LAT"].between(lat_min, lat_max))
                & (df_locations["LOCATION_LONG"].between(lon_min, lon_max))
            ]
            site_ids = sorted(inside["SITE_ID"].tolist())
            if not site_ids:
                continue
            best = select_best_site_for_region(
                df_locations, rname, allowed_site_ids=set(site_ids)
            )
            region_sites[rname] = {
                "SITE_IDS": site_ids,
                "LOCATION_LAT": float(best["LOCATION_LAT"]),
                "LOCATION_LONG": float(best["LOCATION_LONG"]),
            }
        df_info_best = choose_best_info_rows(df_info_filt)
        df_info_best = df_info_best[df_info_best["SITE_ID"].astype(str).isin(df_locations["SITE_ID"])]
        mapping = build_siteid_to_filename_from_info(df_info_best)
        payload = {
            "regions": region_sites,
            "site_id_to_file": mapping,
        }
    elif args.all_regions:
        chosen_site_ids = []
        region_to_site = {}
        for rname in regions.keys():
            best = select_best_site_for_region(df_sites_allowed, rname, allowed_site_ids=allowed_site_ids)
            sid = str(best["SITE_ID"]).strip()
            region_to_site[rname] = {
                "SITE_ID": sid,
                "LOCATION_LAT": float(best["LOCATION_LAT"]),
                "LOCATION_LONG": float(best["LOCATION_LONG"]),
            }
            if sid not in chosen_site_ids:
                chosen_site_ids.append(sid)
        # Pick best filename rows for the chosen site IDs only
        df_info_best = choose_best_info_rows(df_info_filt[df_info_filt["SITE_ID"].astype(str).isin(chosen_site_ids)])
        mapping = build_siteid_to_filename_from_info(df_info_best)
        payload = {
            "region_to_site": region_to_site,
            "site_id_to_file": mapping,
        }
    else:
        best = select_best_site_for_region(df_sites_allowed, args.region, allowed_site_ids=allowed_site_ids)
        sid = str(best["SITE_ID"]).strip()
        df_info_best = choose_best_info_rows(df_info_filt[df_info_filt["SITE_ID"].astype(str) == sid])
        mapping = build_siteid_to_filename_from_info(df_info_best)
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
