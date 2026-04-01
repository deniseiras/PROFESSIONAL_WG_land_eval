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
END_YEAR = 2003
PLOT_MODE = "all"
# PLOT_MODE = "single"
# SELECTED_REGION = "Global"
INPUT_FNAME_VARS_PREFIX="NEE_TSOI_NEP_TLAI_SNOW_DEPTH_H2OSNO"
# INPUT_FNAME_VARS_PREFIX="NEE"
VAR_NAME = "TSOI"
SELECTED_REGION = "South American Tropical"
SPATIAL_AGG = "boxmean"  # one of: boxmean, nearest_center, nearest_site

CSV_V_COL = "NEE_VUT_REF"

COMPARE_SOURCE = "both"  # one of: none, fluxnet, fluxcom, both
FLUXCOM_DIR = "./data/FluxComm/CarbonFluxes/RS_METEO/ensemble/ERA5/monthly"

# member_ids = [f"{i:04d}" for i in range(1, 31)]  # 0001..0030
member_ids = [f"{i:04d}" for i in range(1, 31)]  # 0001..0030
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


def get_units(da):
    attrs = getattr(da, "attrs", {})
    for k in ("units", "Units", "unit"):
        if k in attrs and str(attrs[k]).strip():
            return str(attrs[k]).strip()
    return None


def get_region_target_latlon(info, mode):
    """
    Determine a target lat/lon for a region based on the requested spatial aggregation mode.
    mode:
      - 'nearest_center': center of the region bounds
      - 'nearest_site': representative site coordinates from REGIONS metadata
    Returns: (lat_t, lon_t) in degrees, with lon normalized to [-180, 180).
    """
    lat_min, lat_max, lon_min, lon_max = info["bounds"]
    if mode == "nearest_center":
        lat_t = 0.5 * (lat_min + lat_max)
        lon_t = 0.5 * (lon_min + lon_max)
    elif mode == "nearest_site":
        site = info.get("site", {})
        lat_t = float(site.get("LOCATION_LAT", np.nan))
        lon_t = float(site.get("LOCATION_LONG", np.nan))
    else:
        raise ValueError(f"Unsupported mode for single-point selection: {mode}")
    lon_t = to_minus180_180(lon_t)
    return float(lat_t), float(lon_t)



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


# Additional FluxNET site metadata to overlay multiple site CSVs per region
REGION_SITES = {
"regions": {
"North American Boreal": {
"SITE_IDS": [
"CA-Man",
"CA-NS1",
"CA-NS2",
"CA-NS3",
"CA-NS4",
"CA-NS5",
"CA-NS6",
"CA-NS7",
"CA-Oas",
"CA-Obs",
"CA-SF1",
"CA-SF2",
"CA-SF3",
"GL-NuF",
"US-Ivo",
"US-Prr"
],
"LOCATION_LAT": 54.485,
"LOCATION_LONG": -105.8176
},
"North American Temperate": {
"SITE_IDS": [
"CA-Gro",
"CA-Qfo",
"CA-TP1",
"CA-TP2",
"CA-TP3",
"CA-TP4",
"CA-TPD",
"US-AR1",
"US-AR2",
"US-ARM",
"US-ARb",
"US-ARc",
"US-Blo",
"US-CRT",
"US-Cop",
"US-GBT",
"US-GLE",
"US-Goo",
"US-Ha1",
"US-IB2",
"US-LWW",
"US-Lin",
"US-Los",
"US-MMS",
"US-Me1",
"US-Me2",
"US-Me3",
"US-Me4",
"US-Me5",
"US-Me6",
"US-Myb",
"US-NR1",
"US-Ne1",
"US-Ne2",
"US-Ne3",
"US-ORv",
"US-Oho",
"US-PFa",
"US-SRC",
"US-SRG",
"US-SRM",
"US-Sta",
"US-Syv",
"US-Ton",
"US-Tw1",
"US-Tw2",
"US-Tw3",
"US-Tw4",
"US-Twt",
"US-UMB",
"US-UMd",
"US-Var",
"US-WCr",
"US-WPT",
"US-Whs",
"US-Wi0",
"US-Wi1",
"US-Wi2",
"US-Wi3",
"US-Wi4",
"US-Wi5",
"US-Wi6",
"US-Wi7",
"US-Wi8",
"US-Wi9",
"US-Wkg"
],
"LOCATION_LAT": 41.1797,
"LOCATION_LONG": -96.4397
},
"South American Tropical": {
"SITE_IDS": [
"BR-Sa1",
"BR-Sa3",
"GF-Guy",
"PA-SPn",
"PA-SPs"
],
"LOCATION_LAT": -3.018,
"LOCATION_LONG": -54.9714
},
"South American Temperate": {
"SITE_IDS": [
"AR-SLu",
"AR-Vir"
],
"LOCATION_LAT": -28.2395,
"LOCATION_LONG": -56.1886
},
"Northern Africa": {
"SITE_IDS": [
"SN-Dhr"
],
"LOCATION_LAT": 15.4028,
"LOCATION_LONG": -15.4322
},
"Southern Africa": {
"SITE_IDS": [
"ZM-Mon"
],
"LOCATION_LAT": -15.4391,
"LOCATION_LONG": 23.2525
},
"Eurasian Boreal": {
"SITE_IDS": [
"DE-Akm",
"DE-Geb",
"DE-Gri",
"DE-Hai",
"DE-Kli",
"DE-Lnf",
"DE-Obe",
"DE-Spw",
"DE-Tha",
"DE-Zrk",
"DK-Eng",
"DK-Sor",
"FI-Hyy",
"FI-Jok",
"FI-Let",
"FI-Lom",
"FI-Sod",
"RU-Che",
"RU-Fyo",
"RU-Ha1"
],
"LOCATION_LAT": 54.7252,
"LOCATION_LONG": 90.0022
},
"Eurasian Temperate": {
"SITE_IDS": [
"AT-Neu",
"CH-Cha",
"CH-Dav",
"CH-Fru",
"CH-Lae",
"CH-Oe1",
"CH-Oe2",
"CN-Cha",
"CN-Cng",
"CN-Dan",
"CN-Du2",
"CN-Du3",
"CN-Ha2",
"CN-HaM",
"CN-Sw2",
"CZ-BK1",
"CZ-BK2",
"CZ-wet",
"DE-Lkb",
"DE-SfN",
"ES-Amo",
"ES-LJu",
"ES-LgS",
"ES-Ln2",
"FR-Fon",
"FR-Gri",
"FR-LBr",
"FR-Pue",
"IT-BCi",
"IT-CA1",
"IT-CA2",
"IT-CA3",
"IT-Col",
"IT-Cp2",
"IT-Cpz",
"IT-Isp",
"IT-La2",
"IT-Lav",
"IT-MBo",
"IT-Noe",
"IT-PT1",
"IT-Ren",
"IT-Ro1",
"IT-Ro2",
"IT-SR2",
"IT-SRo",
"IT-Tor",
"JP-MBF",
"JP-SMF"
],
"LOCATION_LAT": 30.4978,
"LOCATION_LONG": 91.0664
},
"Tropical Asia": {
"SITE_IDS": [
"MY-PSO"
],
"LOCATION_LAT": 2.973,
"LOCATION_LONG": 102.3062
},
"Australia": {
"SITE_IDS": [
"AU-ASM",
"AU-Ade",
"AU-Cpr",
"AU-Cum",
"AU-DaP",
"AU-DaS",
"AU-Dry",
"AU-Emr",
"AU-Fog",
"AU-GWW",
"AU-Gin",
"AU-How",
"AU-Lox",
"AU-RDF",
"AU-Rig",
"AU-Rob",
"AU-Stp",
"AU-TTE",
"AU-Tum",
"AU-Wac",
"AU-Whr",
"AU-Wom",
"AU-Ync"
],
"LOCATION_LAT": -22.283,
"LOCATION_LONG": 133.249
},
"Europe": {
"SITE_IDS": [
"AT-Neu",
"BE-Bra",
"BE-Lon",
"BE-Vie",
"CH-Cha",
"CH-Dav",
"CH-Fru",
"CH-Lae",
"CH-Oe1",
"CH-Oe2",
"CZ-BK1",
"CZ-BK2",
"CZ-wet",
"DE-Akm",
"DE-Geb",
"DE-Gri",
"DE-Hai",
"DE-Kli",
"DE-Lkb",
"DE-Lnf",
"DE-Obe",
"DE-RuR",
"DE-RuS",
"DE-Seh",
"DE-SfN",
"DE-Spw",
"DE-Tha",
"DE-Zrk",
"DK-Eng",
"DK-Fou",
"DK-Sor",
"ES-Amo",
"ES-LJu",
"ES-LgS",
"ES-Ln2",
"FI-Hyy",
"FI-Jok",
"FI-Let",
"FI-Lom",
"FI-Sod",
"FR-Fon",
"FR-Gri",
"FR-LBr",
"FR-Pue",
"IT-BCi",
"IT-CA1",
"IT-CA2",
"IT-CA3",
"IT-Col",
"IT-Cp2",
"IT-Cpz",
"IT-Isp",
"IT-La2",
"IT-Lav",
"IT-MBo",
"IT-Noe",
"IT-PT1",
"IT-Ren",
"IT-Ro1",
"IT-Ro2",
"IT-SR2",
"IT-SRo",
"IT-Tor",
"NL-Hor",
"NL-Loo",
"RU-Fyo"
],
"LOCATION_LAT": 51.8922,
"LOCATION_LONG": 14.0337
}
},
"site_id_to_file": {
"AR-SLu": "./data/FluxNET/FLX_AR-SLu_FLUXNET2015_SUBSET_MM_2009-2011_1-4.csv",
"AR-Vir": "./data/FluxNET/FLX_AR-Vir_FLUXNET2015_SUBSET_MM_2009-2012_1-4.csv",
"AT-Neu": "./data/FluxNET/FLX_AT-Neu_FLUXNET2015_SUBSET_MM_2002-2012_1-4.csv",
"AU-ASM": "./data/FluxNET/FLX_AU-ASM_FLUXNET2015_SUBSET_MM_2010-2014_2-4.csv",
"AU-Ade": "./data/FluxNET/FLX_AU-Ade_FLUXNET2015_SUBSET_MM_2007-2009_1-4.csv",
"AU-Cpr": "./data/FluxNET/FLX_AU-Cpr_FLUXNET2015_SUBSET_MM_2010-2014_2-4.csv",
"AU-Cum": "./data/FluxNET/FLX_AU-Cum_FLUXNET2015_SUBSET_MM_2012-2014_2-4.csv",
"AU-DaP": "./data/FluxNET/FLX_AU-DaP_FLUXNET2015_SUBSET_MM_2007-2013_2-4.csv",
"AU-DaS": "./data/FluxNET/FLX_AU-DaS_FLUXNET2015_SUBSET_MM_2008-2014_2-4.csv",
"AU-Dry": "./data/FluxNET/FLX_AU-Dry_FLUXNET2015_SUBSET_MM_2008-2014_2-4.csv",
"AU-Emr": "./data/FluxNET/FLX_AU-Emr_FLUXNET2015_SUBSET_MM_2011-2013_1-4.csv",
"AU-Fog": "./data/FluxNET/FLX_AU-Fog_FLUXNET2015_SUBSET_MM_2006-2008_1-4.csv",
"AU-GWW": "./data/FluxNET/FLX_AU-GWW_FLUXNET2015_SUBSET_MM_2013-2014_1-4.csv",
"AU-Gin": "./data/FluxNET/FLX_AU-Gin_FLUXNET2015_SUBSET_MM_2011-2014_1-4.csv",
"AU-How": "./data/FluxNET/FLX_AU-How_FLUXNET2015_SUBSET_MM_2001-2014_1-4.csv",
"AU-Lox": "./data/FluxNET/FLX_AU-Lox_FLUXNET2015_SUBSET_MM_2008-2009_1-4.csv",
"AU-RDF": "./data/FluxNET/FLX_AU-RDF_FLUXNET2015_SUBSET_MM_2011-2013_1-4.csv",
"AU-Rig": "./data/FluxNET/FLX_AU-Rig_FLUXNET2015_SUBSET_MM_2011-2014_2-4.csv",
"AU-Rob": "./data/FluxNET/FLX_AU-Rob_FLUXNET2015_SUBSET_MM_2014-2014_1-4.csv",
"AU-Stp": "./data/FluxNET/FLX_AU-Stp_FLUXNET2015_SUBSET_MM_2008-2014_1-4.csv",
"AU-TTE": "./data/FluxNET/FLX_AU-TTE_FLUXNET2015_SUBSET_MM_2012-2014_2-4.csv",
"AU-Tum": "./data/FluxNET/FLX_AU-Tum_FLUXNET2015_SUBSET_MM_2001-2014_2-4.csv",
"AU-Wac": "./data/FluxNET/FLX_AU-Wac_FLUXNET2015_SUBSET_MM_2005-2008_1-4.csv",
"AU-Whr": "./data/FluxNET/FLX_AU-Whr_FLUXNET2015_SUBSET_MM_2011-2014_2-4.csv",
"AU-Wom": "./data/FluxNET/FLX_AU-Wom_FLUXNET2015_SUBSET_MM_2010-2014_1-4.csv",
"AU-Ync": "./data/FluxNET/FLX_AU-Ync_FLUXNET2015_SUBSET_MM_2012-2014_1-4.csv",
"BE-Bra": "./data/FluxNET/FLX_BE-Bra_FLUXNET2015_SUBSET_MM_1996-2014_2-4.csv",
"BE-Lon": "./data/FluxNET/FLX_BE-Lon_FLUXNET2015_SUBSET_MM_2004-2014_1-4.csv",
"BE-Vie": "./data/FluxNET/FLX_BE-Vie_FLUXNET2015_SUBSET_MM_1996-2014_1-4.csv",
"BR-Sa1": "./data/FluxNET/FLX_BR-Sa1_FLUXNET2015_SUBSET_MM_2002-2011_1-4.csv",
"BR-Sa3": "./data/FluxNET/FLX_BR-Sa3_FLUXNET2015_SUBSET_MM_2000-2004_1-4.csv",
"CA-Gro": "./data/FluxNET/FLX_CA-Gro_FLUXNET2015_SUBSET_MM_2003-2014_1-4.csv",
"CA-Man": "./data/FluxNET/FLX_CA-Man_FLUXNET2015_SUBSET_MM_1994-2008_1-4.csv",
"CA-NS1": "./data/FluxNET/FLX_CA-NS1_FLUXNET2015_SUBSET_MM_2001-2005_2-4.csv",
"CA-NS2": "./data/FluxNET/FLX_CA-NS2_FLUXNET2015_SUBSET_MM_2001-2005_1-4.csv",
"CA-NS3": "./data/FluxNET/FLX_CA-NS3_FLUXNET2015_SUBSET_MM_2001-2005_1-4.csv",
"CA-NS4": "./data/FluxNET/FLX_CA-NS4_FLUXNET2015_SUBSET_MM_2002-2005_1-4.csv",
"CA-NS5": "./data/FluxNET/FLX_CA-NS5_FLUXNET2015_SUBSET_MM_2001-2005_1-4.csv",
"CA-NS6": "./data/FluxNET/FLX_CA-NS6_FLUXNET2015_SUBSET_MM_2001-2005_1-4.csv",
"CA-NS7": "./data/FluxNET/FLX_CA-NS7_FLUXNET2015_SUBSET_MM_2002-2005_1-4.csv",
"CA-Oas": "./data/FluxNET/FLX_CA-Oas_FLUXNET2015_SUBSET_MM_1996-2010_1-4.csv",
"CA-Obs": "./data/FluxNET/FLX_CA-Obs_FLUXNET2015_SUBSET_MM_1997-2010_1-4.csv",
"CA-Qfo": "./data/FluxNET/FLX_CA-Qfo_FLUXNET2015_SUBSET_MM_2003-2010_1-4.csv",
"CA-SF1": "./data/FluxNET/FLX_CA-SF1_FLUXNET2015_SUBSET_MM_2003-2006_1-4.csv",
"CA-SF2": "./data/FluxNET/FLX_CA-SF2_FLUXNET2015_SUBSET_MM_2001-2005_1-4.csv",
"CA-SF3": "./data/FluxNET/FLX_CA-SF3_FLUXNET2015_SUBSET_MM_2001-2006_1-4.csv",
"CA-TP1": "./data/FluxNET/FLX_CA-TP1_FLUXNET2015_SUBSET_MM_2002-2014_2-4.csv",
"CA-TP2": "./data/FluxNET/FLX_CA-TP2_FLUXNET2015_SUBSET_MM_2002-2007_1-4.csv",
"CA-TP3": "./data/FluxNET/FLX_CA-TP3_FLUXNET2015_SUBSET_MM_2002-2014_1-4.csv",
"CA-TP4": "./data/FluxNET/FLX_CA-TP4_FLUXNET2015_SUBSET_MM_2002-2014_1-4.csv",
"CA-TPD": "./data/FluxNET/FLX_CA-TPD_FLUXNET2015_SUBSET_MM_2012-2014_1-4.csv",
"CG-Tch": "./data/FluxNET/FLX_CG-Tch_FLUXNET2015_SUBSET_MM_2006-2009_1-4.csv",
"CH-Cha": "./data/FluxNET/FLX_CH-Cha_FLUXNET2015_SUBSET_MM_2005-2014_2-4.csv",
"CH-Dav": "./data/FluxNET/FLX_CH-Dav_FLUXNET2015_SUBSET_MM_1997-2014_1-4.csv",
"CH-Fru": "./data/FluxNET/FLX_CH-Fru_FLUXNET2015_SUBSET_MM_2005-2014_2-4.csv",
"CH-Lae": "./data/FluxNET/FLX_CH-Lae_FLUXNET2015_SUBSET_MM_2004-2014_1-4.csv",
"CH-Oe1": "./data/FluxNET/FLX_CH-Oe1_FLUXNET2015_SUBSET_MM_2002-2008_2-4.csv",
"CH-Oe2": "./data/FluxNET/FLX_CH-Oe2_FLUXNET2015_SUBSET_MM_2004-2014_1-4.csv",
"CN-Cha": "./data/FluxNET/FLX_CN-Cha_FLUXNET2015_SUBSET_MM_2003-2005_1-4.csv",
"CN-Cng": "./data/FluxNET/FLX_CN-Cng_FLUXNET2015_SUBSET_MM_2007-2010_1-4.csv",
"CN-Dan": "./data/FluxNET/FLX_CN-Dan_FLUXNET2015_SUBSET_MM_2004-2005_1-4.csv",
"CN-Din": "./data/FluxNET/FLX_CN-Din_FLUXNET2015_SUBSET_MM_2003-2005_1-4.csv",
"CN-Du2": "./data/FluxNET/FLX_CN-Du2_FLUXNET2015_SUBSET_MM_2006-2008_1-4.csv",
"CN-Du3": "./data/FluxNET/FLX_CN-Du3_FLUXNET2015_SUBSET_MM_2009-2010_1-4.csv",
"CN-Ha2": "./data/FluxNET/FLX_CN-Ha2_FLUXNET2015_SUBSET_MM_2003-2005_1-4.csv",
"CN-HaM": "./data/FluxNET/FLX_CN-HaM_FLUXNET2015_SUBSET_MM_2002-2004_1-4.csv",
"CN-Qia": "./data/FluxNET/FLX_CN-Qia_FLUXNET2015_SUBSET_MM_2003-2005_1-4.csv",
"CN-Sw2": "./data/FluxNET/FLX_CN-Sw2_FLUXNET2015_SUBSET_MM_2010-2012_1-4.csv",
"CZ-BK1": "./data/FluxNET/FLX_CZ-BK1_FLUXNET2015_SUBSET_MM_2004-2014_2-4.csv",
"CZ-BK2": "./data/FluxNET/FLX_CZ-BK2_FLUXNET2015_SUBSET_MM_2004-2012_2-4.csv",
"CZ-wet": "./data/FluxNET/FLX_CZ-wet_FLUXNET2015_SUBSET_MM_2006-2014_1-4.csv",
"DE-Akm": "./data/FluxNET/FLX_DE-Akm_FLUXNET2015_SUBSET_MM_2009-2014_1-4.csv",
"DE-Geb": "./data/FluxNET/FLX_DE-Geb_FLUXNET2015_SUBSET_MM_2001-2014_1-4.csv",
"DE-Gri": "./data/FluxNET/FLX_DE-Gri_FLUXNET2015_SUBSET_MM_2004-2014_1-4.csv",
"DE-Hai": "./data/FluxNET/FLX_DE-Hai_FLUXNET2015_SUBSET_MM_2000-2012_1-4.csv",
"DE-Kli": "./data/FluxNET/FLX_DE-Kli_FLUXNET2015_SUBSET_MM_2004-2014_1-4.csv",
"DE-Lkb": "./data/FluxNET/FLX_DE-Lkb_FLUXNET2015_SUBSET_MM_2009-2013_1-4.csv",
"DE-Lnf": "./data/FluxNET/FLX_DE-Lnf_FLUXNET2015_SUBSET_MM_2002-2012_1-4.csv",
"DE-Obe": "./data/FluxNET/FLX_DE-Obe_FLUXNET2015_SUBSET_MM_2008-2014_1-4.csv",
"DE-RuR": "./data/FluxNET/FLX_DE-RuR_FLUXNET2015_SUBSET_MM_2011-2014_1-4.csv",
"DE-RuS": "./data/FluxNET/FLX_DE-RuS_FLUXNET2015_SUBSET_MM_2011-2014_1-4.csv",
"DE-Seh": "./data/FluxNET/FLX_DE-Seh_FLUXNET2015_SUBSET_MM_2007-2010_1-4.csv",
"DE-SfN": "./data/FluxNET/FLX_DE-SfN_FLUXNET2015_SUBSET_MM_2012-2014_1-4.csv",
"DE-Spw": "./data/FluxNET/FLX_DE-Spw_FLUXNET2015_SUBSET_MM_2010-2014_1-4.csv",
"DE-Tha": "./data/FluxNET/FLX_DE-Tha_FLUXNET2015_SUBSET_MM_1996-2014_1-4.csv",
"DE-Zrk": "./data/FluxNET/FLX_DE-Zrk_FLUXNET2015_SUBSET_MM_2013-2014_2-4.csv",
"DK-Eng": "./data/FluxNET/FLX_DK-Eng_FLUXNET2015_SUBSET_MM_2005-2008_1-4.csv",
"DK-Fou": "./data/FluxNET/FLX_DK-Fou_FLUXNET2015_SUBSET_MM_2005-2005_1-4.csv",
"DK-Sor": "./data/FluxNET/FLX_DK-Sor_FLUXNET2015_SUBSET_MM_1996-2014_2-4.csv",
"ES-Amo": "./data/FluxNET/FLX_ES-Amo_FLUXNET2015_SUBSET_MM_2007-2012_1-4.csv",
"ES-LJu": "./data/FluxNET/FLX_ES-LJu_FLUXNET2015_SUBSET_MM_2004-2013_1-4.csv",
"ES-LgS": "./data/FluxNET/FLX_ES-LgS_FLUXNET2015_SUBSET_MM_2007-2009_1-4.csv",
"ES-Ln2": "./data/FluxNET/FLX_ES-Ln2_FLUXNET2015_SUBSET_MM_2009-2009_1-4.csv",
"FI-Hyy": "./data/FluxNET/FLX_FI-Hyy_FLUXNET2015_SUBSET_MM_1996-2014_1-4.csv",
"FI-Jok": "./data/FluxNET/FLX_FI-Jok_FLUXNET2015_SUBSET_MM_2000-2003_1-4.csv",
"FI-Let": "./data/FluxNET/FLX_FI-Let_FLUXNET2015_SUBSET_MM_2009-2012_1-4.csv",
"FI-Lom": "./data/FluxNET/FLX_FI-Lom_FLUXNET2015_SUBSET_MM_2007-2009_1-4.csv",
"FI-Sod": "./data/FluxNET/FLX_FI-Sod_FLUXNET2015_SUBSET_MM_2001-2014_1-4.csv",
"FR-Fon": "./data/FluxNET/FLX_FR-Fon_FLUXNET2015_SUBSET_MM_2005-2014_1-4.csv",
"FR-Gri": "./data/FluxNET/FLX_FR-Gri_FLUXNET2015_SUBSET_MM_2004-2014_1-4.csv",
"FR-LBr": "./data/FluxNET/FLX_FR-LBr_FLUXNET2015_SUBSET_MM_1996-2008_1-4.csv",
"FR-Pue": "./data/FluxNET/FLX_FR-Pue_FLUXNET2015_SUBSET_MM_2000-2014_2-4.csv",
"GF-Guy": "./data/FluxNET/FLX_GF-Guy_FLUXNET2015_SUBSET_MM_2004-2014_2-4.csv",
"GH-Ank": "./data/FluxNET/FLX_GH-Ank_FLUXNET2015_SUBSET_MM_2011-2014_1-4.csv",
"GL-NuF": "./data/FluxNET/FLX_GL-NuF_FLUXNET2015_SUBSET_MM_2008-2014_1-4.csv",
"GL-ZaF": "./data/FluxNET/FLX_GL-ZaF_FLUXNET2015_SUBSET_MM_2008-2011_2-4.csv",
"GL-ZaH": "./data/FluxNET/FLX_GL-ZaH_FLUXNET2015_SUBSET_MM_2000-2014_2-4.csv",
"IT-BCi": "./data/FluxNET/FLX_IT-BCi_FLUXNET2015_SUBSET_MM_2004-2014_2-4.csv",
"IT-CA1": "./data/FluxNET/FLX_IT-CA1_FLUXNET2015_SUBSET_MM_2011-2014_2-4.csv",
"IT-CA2": "./data/FluxNET/FLX_IT-CA2_FLUXNET2015_SUBSET_MM_2011-2014_2-4.csv",
"IT-CA3": "./data/FluxNET/FLX_IT-CA3_FLUXNET2015_SUBSET_MM_2011-2014_2-4.csv",
"IT-Col": "./data/FluxNET/FLX_IT-Col_FLUXNET2015_SUBSET_MM_1996-2014_1-4.csv",
"IT-Cp2": "./data/FluxNET/FLX_IT-Cp2_FLUXNET2015_SUBSET_MM_2012-2014_2-4.csv",
"IT-Cpz": "./data/FluxNET/FLX_IT-Cpz_FLUXNET2015_SUBSET_MM_1997-2009_1-4.csv",
"IT-Isp": "./data/FluxNET/FLX_IT-Isp_FLUXNET2015_SUBSET_MM_2013-2014_1-4.csv",
"IT-La2": "./data/FluxNET/FLX_IT-La2_FLUXNET2015_SUBSET_MM_2000-2002_1-4.csv",
"IT-Lav": "./data/FluxNET/FLX_IT-Lav_FLUXNET2015_SUBSET_MM_2003-2014_2-4.csv",
"IT-MBo": "./data/FluxNET/FLX_IT-MBo_FLUXNET2015_SUBSET_MM_2003-2013_1-4.csv",
"IT-Noe": "./data/FluxNET/FLX_IT-Noe_FLUXNET2015_SUBSET_MM_2004-2014_2-4.csv",
"IT-PT1": "./data/FluxNET/FLX_IT-PT1_FLUXNET2015_SUBSET_MM_2002-2004_1-4.csv",
"IT-Ren": "./data/FluxNET/FLX_IT-Ren_FLUXNET2015_SUBSET_MM_1998-2013_1-4.csv",
"IT-Ro1": "./data/FluxNET/FLX_IT-Ro1_FLUXNET2015_SUBSET_MM_2000-2008_1-4.csv",
"IT-Ro2": "./data/FluxNET/FLX_IT-Ro2_FLUXNET2015_SUBSET_MM_2002-2012_1-4.csv",
"IT-SR2": "./data/FluxNET/FLX_IT-SR2_FLUXNET2015_SUBSET_MM_2013-2014_1-4.csv",
"IT-SRo": "./data/FluxNET/FLX_IT-SRo_FLUXNET2015_SUBSET_MM_1999-2012_1-4.csv",
"IT-Tor": "./data/FluxNET/FLX_IT-Tor_FLUXNET2015_SUBSET_MM_2008-2014_2-4.csv",
"JP-MBF": "./data/FluxNET/FLX_JP-MBF_FLUXNET2015_SUBSET_MM_2003-2005_1-4.csv",
"JP-SMF": "./data/FluxNET/FLX_JP-SMF_FLUXNET2015_SUBSET_MM_2002-2006_1-4.csv",
"MY-PSO": "./data/FluxNET/FLX_MY-PSO_FLUXNET2015_SUBSET_MM_2003-2009_1-4.csv",
"NL-Hor": "./data/FluxNET/FLX_NL-Hor_FLUXNET2015_SUBSET_MM_2004-2011_1-4.csv",
"NL-Loo": "./data/FluxNET/FLX_NL-Loo_FLUXNET2015_SUBSET_MM_1996-2014_1-4.csv",
"PA-SPn": "./data/FluxNET/FLX_PA-SPn_FLUXNET2015_SUBSET_MM_2007-2009_1-4.csv",
"PA-SPs": "./data/FluxNET/FLX_PA-SPs_FLUXNET2015_SUBSET_MM_2007-2009_1-4.csv",
"RU-Che": "./data/FluxNET/FLX_RU-Che_FLUXNET2015_SUBSET_MM_2002-2005_1-4.csv",
"RU-Cok": "./data/FluxNET/FLX_RU-Cok_FLUXNET2015_SUBSET_MM_2003-2014_2-4.csv",
"RU-Fyo": "./data/FluxNET/FLX_RU-Fyo_FLUXNET2015_SUBSET_MM_1998-2014_2-4.csv",
"RU-Ha1": "./data/FluxNET/FLX_RU-Ha1_FLUXNET2015_SUBSET_MM_2002-2004_1-4.csv",
"SD-Dem": "./data/FluxNET/FLX_SD-Dem_FLUXNET2015_SUBSET_MM_2005-2009_2-4.csv",
"SJ-Adv": "./data/FluxNET/FLX_SJ-Adv_FLUXNET2015_SUBSET_MM_2011-2014_1-4.csv",
"SJ-Blv": "./data/FluxNET/FLX_SJ-Blv_FLUXNET2015_SUBSET_MM_2008-2009_1-4.csv",
"SN-Dhr": "./data/FluxNET/FLX_SN-Dhr_FLUXNET2015_SUBSET_MM_2010-2013_1-4.csv",
"US-AR1": "./data/FluxNET/FLX_US-AR1_FLUXNET2015_SUBSET_MM_2009-2012_1-4.csv",
"US-AR2": "./data/FluxNET/FLX_US-AR2_FLUXNET2015_SUBSET_MM_2009-2012_1-4.csv",
"US-ARM": "./data/FluxNET/FLX_US-ARM_FLUXNET2015_SUBSET_MM_2003-2012_1-4.csv",
"US-ARb": "./data/FluxNET/FLX_US-ARb_FLUXNET2015_SUBSET_MM_2005-2006_1-4.csv",
"US-ARc": "./data/FluxNET/FLX_US-ARc_FLUXNET2015_SUBSET_MM_2005-2006_1-4.csv",
"US-Atq": "./data/FluxNET/FLX_US-Atq_FLUXNET2015_SUBSET_MM_2003-2008_1-4.csv",
"US-Blo": "./data/FluxNET/FLX_US-Blo_FLUXNET2015_SUBSET_MM_1997-2007_1-4.csv",
"US-CRT": "./data/FluxNET/FLX_US-CRT_FLUXNET2015_SUBSET_MM_2011-2013_1-4.csv",
"US-Cop": "./data/FluxNET/FLX_US-Cop_FLUXNET2015_SUBSET_MM_2001-2007_1-4.csv",
"US-GBT": "./data/FluxNET/FLX_US-GBT_FLUXNET2015_SUBSET_MM_1999-2006_1-4.csv",
"US-GLE": "./data/FluxNET/FLX_US-GLE_FLUXNET2015_SUBSET_MM_2004-2014_1-4.csv",
"US-Goo": "./data/FluxNET/FLX_US-Goo_FLUXNET2015_SUBSET_MM_2002-2006_1-4.csv",
"US-Ha1": "./data/FluxNET/FLX_US-Ha1_FLUXNET2015_SUBSET_MM_1991-2012_1-4.csv",
"US-IB2": "./data/FluxNET/FLX_US-IB2_FLUXNET2015_SUBSET_MM_2004-2011_1-4.csv",
"US-Ivo": "./data/FluxNET/FLX_US-Ivo_FLUXNET2015_SUBSET_MM_2004-2007_1-4.csv",
"US-KS1": "./data/FluxNET/FLX_US-KS1_FLUXNET2015_SUBSET_MM_2002-2002_1-4.csv",
"US-KS2": "./data/FluxNET/FLX_US-KS2_FLUXNET2015_SUBSET_MM_2003-2006_1-4.csv",
"US-LWW": "./data/FluxNET/FLX_US-LWW_FLUXNET2015_SUBSET_MM_1997-1998_1-4.csv",
"US-Lin": "./data/FluxNET/FLX_US-Lin_FLUXNET2015_SUBSET_MM_2009-2010_1-4.csv",
"US-Los": "./data/FluxNET/FLX_US-Los_FLUXNET2015_SUBSET_MM_2000-2014_2-4.csv",
"US-MMS": "./data/FluxNET/FLX_US-MMS_FLUXNET2015_SUBSET_MM_1999-2014_1-4.csv",
"US-Me1": "./data/FluxNET/FLX_US-Me1_FLUXNET2015_SUBSET_MM_2004-2005_1-4.csv",
"US-Me2": "./data/FluxNET/FLX_US-Me2_FLUXNET2015_SUBSET_MM_2002-2014_1-4.csv",
"US-Me3": "./data/FluxNET/FLX_US-Me3_FLUXNET2015_SUBSET_MM_2004-2009_1-4.csv",
"US-Me4": "./data/FluxNET/FLX_US-Me4_FLUXNET2015_SUBSET_MM_1996-2000_1-4.csv",
"US-Me5": "./data/FluxNET/FLX_US-Me5_FLUXNET2015_SUBSET_MM_2000-2002_1-4.csv",
"US-Me6": "./data/FluxNET/FLX_US-Me6_FLUXNET2015_SUBSET_MM_2010-2014_2-4.csv",
"US-Myb": "./data/FluxNET/FLX_US-Myb_FLUXNET2015_SUBSET_MM_2010-2014_2-4.csv",
"US-NR1": "./data/FluxNET/FLX_US-NR1_FLUXNET2015_SUBSET_MM_1998-2014_1-4.csv",
"US-Ne1": "./data/FluxNET/FLX_US-Ne1_FLUXNET2015_SUBSET_MM_2001-2013_1-4.csv",
"US-Ne2": "./data/FluxNET/FLX_US-Ne2_FLUXNET2015_SUBSET_MM_2001-2013_1-4.csv",
"US-Ne3": "./data/FluxNET/FLX_US-Ne3_FLUXNET2015_SUBSET_MM_2001-2013_1-4.csv",
"US-ORv": "./data/FluxNET/FLX_US-ORv_FLUXNET2015_SUBSET_MM_2011-2011_1-4.csv",
"US-Oho": "./data/FluxNET/FLX_US-Oho_FLUXNET2015_SUBSET_MM_2004-2013_1-4.csv",
"US-PFa": "./data/FluxNET/FLX_US-PFa_FLUXNET2015_SUBSET_MM_1995-2014_1-4.csv",
"US-Prr": "./data/FluxNET/FLX_US-Prr_FLUXNET2015_SUBSET_MM_2010-2014_1-4.csv",
"US-SRC": "./data/FluxNET/FLX_US-SRC_FLUXNET2015_SUBSET_MM_2008-2014_1-4.csv",
"US-SRG": "./data/FluxNET/FLX_US-SRG_FLUXNET2015_SUBSET_MM_2008-2014_1-4.csv",
"US-SRM": "./data/FluxNET/FLX_US-SRM_FLUXNET2015_SUBSET_MM_2004-2014_1-4.csv",
"US-Sta": "./data/FluxNET/FLX_US-Sta_FLUXNET2015_SUBSET_MM_2005-2009_1-4.csv",
"US-Syv": "./data/FluxNET/FLX_US-Syv_FLUXNET2015_SUBSET_MM_2001-2014_1-4.csv",
"US-Ton": "./data/FluxNET/FLX_US-Ton_FLUXNET2015_SUBSET_MM_2001-2014_1-4.csv",
"US-Tw1": "./data/FluxNET/FLX_US-Tw1_FLUXNET2015_SUBSET_MM_2012-2014_1-4.csv",
"US-Tw2": "./data/FluxNET/FLX_US-Tw2_FLUXNET2015_SUBSET_MM_2012-2013_1-4.csv",
"US-Tw3": "./data/FluxNET/FLX_US-Tw3_FLUXNET2015_SUBSET_MM_2013-2014_2-4.csv",
"US-Tw4": "./data/FluxNET/FLX_US-Tw4_FLUXNET2015_SUBSET_MM_2013-2014_1-4.csv",
"US-Twt": "./data/FluxNET/FLX_US-Twt_FLUXNET2015_SUBSET_MM_2009-2014_1-4.csv",
"US-UMB": "./data/FluxNET/FLX_US-UMB_FLUXNET2015_SUBSET_MM_2000-2014_1-4.csv",
"US-UMd": "./data/FluxNET/FLX_US-UMd_FLUXNET2015_SUBSET_MM_2007-2014_1-4.csv",
"US-Var": "./data/FluxNET/FLX_US-Var_FLUXNET2015_SUBSET_MM_2000-2014_1-4.csv",
"US-WCr": "./data/FluxNET/FLX_US-WCr_FLUXNET2015_SUBSET_MM_1999-2014_1-4.csv",
"US-WPT": "./data/FluxNET/FLX_US-WPT_FLUXNET2015_SUBSET_MM_2011-2013_1-4.csv",
"US-Whs": "./data/FluxNET/FLX_US-Whs_FLUXNET2015_SUBSET_MM_2007-2014_1-4.csv",
"US-Wi0": "./data/FluxNET/FLX_US-Wi0_FLUXNET2015_SUBSET_MM_2002-2002_1-4.csv",
"US-Wi1": "./data/FluxNET/FLX_US-Wi1_FLUXNET2015_SUBSET_MM_2003-2003_1-4.csv",
"US-Wi2": "./data/FluxNET/FLX_US-Wi2_FLUXNET2015_SUBSET_MM_2003-2003_1-4.csv",
"US-Wi3": "./data/FluxNET/FLX_US-Wi3_FLUXNET2015_SUBSET_MM_2002-2004_1-4.csv",
"US-Wi4": "./data/FluxNET/FLX_US-Wi4_FLUXNET2015_SUBSET_MM_2002-2005_1-4.csv",
"US-Wi5": "./data/FluxNET/FLX_US-Wi5_FLUXNET2015_SUBSET_MM_2004-2004_1-4.csv",
"US-Wi6": "./data/FluxNET/FLX_US-Wi6_FLUXNET2015_SUBSET_MM_2002-2003_1-4.csv",
"US-Wi7": "./data/FluxNET/FLX_US-Wi7_FLUXNET2015_SUBSET_MM_2005-2005_1-4.csv",
"US-Wi8": "./data/FluxNET/FLX_US-Wi8_FLUXNET2015_SUBSET_MM_2002-2002_1-4.csv",
"US-Wi9": "./data/FluxNET/FLX_US-Wi9_FLUXNET2015_SUBSET_MM_2004-2005_1-4.csv",
"US-Wkg": "./data/FluxNET/FLX_US-Wkg_FLUXNET2015_SUBSET_MM_2004-2014_1-4.csv",
"ZM-Mon": "./data/FluxNET/FLX_ZM-Mon_FLUXNET2015_SUBSET_MM_2000-2009_2-4.csv"
}
}

# ----------------------------
# CLI arguments
# ----------------------------
parser = argparse.ArgumentParser(
    description=(
        "Plot monthly NEE regional time series: either all 12 regions (subplots) "
        "or a single region occupying the full figure, with optional CSV overlay. "
        "Spatial aggregation can be area-weighted over the region bounds or a single nearest grid cell."
    )
)
parser.add_argument("--in-dir", default=IN_DIR, help="Input directory for [VARIABLES]_monthmean_*.nc files")
parser.add_argument("--out-dir", default=OUT_DIR, help="Output directory for figures")
parser.add_argument("--start-year", type=int, default=START_YEAR, help="Start year (inclusive)")
parser.add_argument("--end-year", type=int, default=END_YEAR, help="End year (inclusive)")
parser.add_argument("--var", choices=["NEE","TSOI","NEP","TLAI","SNOW_DEPTH","H2OSNO"], default=VAR_NAME, help="Variable to plot from the NetCDF files")
parser.add_argument(
    "--mode", choices=["all", "single"], default=PLOT_MODE,
    help="Plot mode: 'all' = 12 subplots, 'single' = one region full-figure"
)
parser.add_argument("--region", default=SELECTED_REGION, help="Region name to plot in single mode")
parser.add_argument("--fluxnet-csv", default=None, help="Optional CSV file to overlay in single mode; if not provided, uses the region's default FluxNET CSV")
parser.add_argument("--fluxnet-timestamp-col", default="TIMESTAMP", help="CSV column with YYYYMM timestamps")
parser.add_argument("--fluxnet-value-col", default=CSV_V_COL, help="CSV column with NEE values")
parser.add_argument("--fluxnet-timestamp-fmt", default="%Y%m", help="Timestamp format for CSV (default %%Y%%m)")
parser.add_argument("--fluxnet-qc-col", default="NEE_VUT_REF_QC", help="CSV column with QC fraction for NEE (0-1)")
parser.add_argument("--fluxnet-qc-min", type=float, default=0.5, help="Minimum QC fraction to accept (strictly > threshold)")
parser.add_argument(
    "--spatial-agg",
    choices=["boxmean", "nearest_center", "nearest_site"],
    default=SPATIAL_AGG,
    help=(
        "Spatial aggregation within each region: "
        "boxmean = area-weighted mean over bounds (default); "
        "nearest_center = single grid cell nearest to region box center; "
        "nearest_site = single grid cell nearest to representative site"
    ),
)
parser.add_argument("--compare", choices=["none", "fluxnet", "fluxcom", "both"], default=COMPARE_SOURCE, help="Comparison overlay: 'none' (no external), 'fluxnet' (FluxNET sites), 'fluxcom' (FLUXCOM gridded), or 'both'")
parser.add_argument("--fluxcom-dir", default=FLUXCOM_DIR, help="Directory with FLUXCOM monthly files (one per year)")
args = parser.parse_args()

IN_DIR = args.in_dir
OUT_DIR = args.out_dir
START_YEAR = args.start_year
END_YEAR = args.end_year
PLOT_MODE = args.mode
SELECTED_REGION = args.region
SPATIAL_AGG = args.spatial_agg
CSV_V_COL = args.fluxnet_value_col
COMPARE_SOURCE = args.compare
FLUXCOM_DIR = args.fluxcom_dir
VAR_NAME = args.var

CSV_T_COL = args.fluxnet_timestamp_col
CSV_T_FMT = args.fluxnet_timestamp_fmt
CSV_QC_COL = args.fluxnet_qc_col
QC_MIN = args.fluxnet_qc_min

# Time window bounds for plotting/overlays
START_DT = pd.to_datetime(f"{START_YEAR}-01-01")
END_DT = pd.to_datetime(f"{END_YEAR}-12-31")

os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# Read MEAN and ensemble MEMBER files per month and compute regional means
# ----------------------------

# Storage
time_list = []
region_mean_series = {name: [] for name in REGIONS}
region_member_series = {name: {mid: [] for mid in member_ids} for name in REGIONS}
region_fluxcom_series = {name: [] for name in REGIONS}

area = None  # area weights (km^2) for primary dataset
area_fluxcom = None  # area weights (km^2) for FLUXCOM grid
y_units_str = None  # units string for ylabel
any_member_found = False  # track if any ensemble member file exists

for year in range(START_YEAR, END_YEAR + 1):
    for month in range(1, 13):
        mean_path = os.path.join(IN_DIR, f"{INPUT_FNAME_VARS_PREFIX}_monthmean_{year}_{month:02d}_mean.nc")

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
            member_path = os.path.join(IN_DIR, f"{INPUT_FNAME_VARS_PREFIX}_monthmean_{year}_{month:02d}_{mid}.nc")
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

        # Build area weights once (only if needed)
        if SPATIAL_AGG == "boxmean" and area is None and ds_ref is not None:
            lat = ds_ref["lat"].values
            lon = ds_ref["lon"].values
            area2d = compute_cell_areas_km2(lat, lon)
            area = xr.DataArray(area2d, coords={"lat": ds_ref["lat"], "lon": ds_ref["lon"]}, dims=("lat", "lon"))

        # Use calendar month as time index
        tval = pd.to_datetime(f"{year}-{month:02d}-01")
        time_list.append(tval)

        # Compute region means for MEAN file (if available), else fill with NaN for now
        if ds_mean is not None:
            var_mean = ds_mean[VAR_NAME]
            if y_units_str is None:
                y_units_str = get_units(var_mean)
            fv_mean = get_fill_value(var_mean)
            for name, info in REGIONS.items():
                lat_min, lat_max, lon_min, lon_max = info["bounds"]
                if SPATIAL_AGG == "boxmean":
                    subset = safe_sel_box(var_mean, lat_min, lat_max, lon_min, lon_max)
                    area_subset = safe_sel_box(area, lat_min, lat_max, lon_min, lon_max)
                    subset_masked = subset.where(subset != fv_mean)
                    weights = area_subset.where(subset_masked.notnull())
                    num = (subset_masked * weights).sum(dim=("lat", "lon"))
                    den = weights.sum(dim=("lat", "lon"))
                    region_mean_series[name].append((num / den).item())
                else:
                    lat_t, lon_t = get_region_target_latlon(info, SPATIAL_AGG)
                    val = var_mean.sel(lat=lat_t, lon=lon_t, method="nearest")
                    val_masked = val.where(val != fv_mean)
                    region_mean_series[name].append(val_masked.values.item() if np.isfinite(val_masked.values) else np.nan)
        else:
            for name in REGIONS:
                region_mean_series[name].append(np.nan)

        # Compute region means for each MEMBER (append NaN if missing)
        for mid in member_ids:
            member_path = os.path.join(IN_DIR, f"{INPUT_FNAME_VARS_PREFIX}_monthmean_{year}_{month:02d}_{mid}.nc")
            print(f"Processing MEMBER file: {member_path}")
            if os.path.exists(member_path):
                any_member_found = True
                ds_m = xr.open_dataset(member_path, decode_times=False)
                ds_m = normalize_longitudes(ds_m, lon_name="lon")
                var_m = ds_m[VAR_NAME]
                if y_units_str is None:
                    y_units_str = get_units(var_m)
                fv_m = get_fill_value(var_m)
                for name, info in REGIONS.items():
                    lat_min, lat_max, lon_min, lon_max = info["bounds"]
                    if SPATIAL_AGG == "boxmean":
                        subset = safe_sel_box(var_m, lat_min, lat_max, lon_min, lon_max)
                        area_subset = safe_sel_box(area, lat_min, lat_max, lon_min, lon_max)
                        subset_masked = subset.where(subset != fv_m)
                        weights = area_subset.where(subset_masked.notnull())
                        num = (subset_masked * weights).sum(dim=("lat", "lon"))
                        den = weights.sum(dim=("lat", "lon"))
                        region_member_series[name][mid].append((num / den).item())
                    else:
                        lat_t, lon_t = get_region_target_latlon(info, SPATIAL_AGG)
                        val = var_m.sel(lat=lat_t, lon=lon_t, method="nearest")
                        val_masked = val.where(val != fv_m)
                        region_member_series[name][mid].append(val_masked.values.item() if np.isfinite(val_masked.values) else np.nan)
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

        # Compute FLUXCOM regional means for this month (optional comparison)
        if COMPARE_SOURCE in ("fluxcom", "both") and VAR_NAME == "NEE":
            flux_path = os.path.join(FLUXCOM_DIR, f"NEE.RS_METEO.FP-NONE.MLM-ALL.METEO-ERA5.720_360.monthly.{year}.nc")
            if os.path.exists(flux_path):
                try:
                    ds_fc = xr.open_dataset(flux_path, decode_times=False)
                    ds_fc = normalize_longitudes(ds_fc, lon_name="lon")

                    # Build FLUXCOM area weights once (if needed)
                    if SPATIAL_AGG == "boxmean" and area_fluxcom is None:
                        lat_fc = ds_fc["lat"].values
                        lon_fc = ds_fc["lon"].values
                        area2d_fc = compute_cell_areas_km2(lat_fc, lon_fc)
                        area_fluxcom = xr.DataArray(area2d_fc, coords={"lat": ds_fc["lat"], "lon": ds_fc["lon"]}, dims=("lat", "lon"))

                    # Select this calendar month (index 0..11)
                    var_fc = ds_fc["NEE"].isel(time=month-1)
                    fv_fc = get_fill_value(var_fc)

                    for name, info in REGIONS.items():
                        lat_min, lat_max, lon_min, lon_max = info["bounds"]
                        if SPATIAL_AGG == "boxmean":
                            subset = safe_sel_box(var_fc, lat_min, lat_max, lon_min, lon_max)
                            area_subset = safe_sel_box(area_fluxcom, lat_min, lat_max, lon_min, lon_max)
                            subset_masked = subset.where(subset != fv_fc)
                            weights = area_subset.where(subset_masked.notnull())
                            num = (subset_masked * weights).sum(dim=("lat", "lon"))
                            den = weights.sum(dim=("lat", "lon"))
                            region_fluxcom_series[name].append((num / den).item())
                        else:
                            lat_t, lon_t = get_region_target_latlon(info, SPATIAL_AGG)
                            val = var_fc.sel(lat=lat_t, lon=lon_t, method="nearest")
                            val_masked = val.where(val != fv_fc)
                            region_fluxcom_series[name].append(val_masked.values.item() if np.isfinite(val_masked.values) else np.nan)
                    ds_fc.close()
                except Exception as e:
                    print(f"Warning: failed to process FLUXCOM file '{flux_path}': {e}")
                    for name in REGIONS:
                        region_fluxcom_series[name].append(np.nan)
            else:
                for name in REGIONS:
                    region_fluxcom_series[name].append(np.nan)

# ----------------------------
# Plot
# ----------------------------
x_time_plot = pd.to_datetime(time_list)

if PLOT_MODE == "all":
    fig, axes = plt.subplots(3, 4, figsize=(20, 12), constrained_layout=True)
    axes = axes.flatten()

    fluxnet_any = False  # track if any FluxNET overlay exists in any panel
    fluxcom_any = False  # track if any FLUXCOM overlay exists in any panel

    for i, name in enumerate(REGIONS.keys()):
        ax = axes[i]

        # Prepare data arrays
        mean_series = np.array(region_mean_series[name], dtype=float)

        # Uncertainty shading and member lines only if any member file was found
        if any_member_found:
            members_arr = np.stack([region_member_series[name][mid] for mid in member_ids], axis=1)  # [ntime, nmembers]
            if np.isfinite(members_arr).any():
                low = np.nanmin(members_arr, axis=1)
                high = np.nanmax(members_arr, axis=1)
                ax.fill_between(x_time_plot, low, high, color="tab:blue", alpha=0.15, label="Member range")

                # Plot member lines (light, transparent)
                for j in range(members_arr.shape[1]):
                    ax.plot(x_time_plot, members_arr[:, j], color="gray", alpha=0.15, linewidth=0.7)

        # Plot MEAN line (main)
        ax.plot(x_time_plot, mean_series, color="tab:blue", linewidth=1.8, label="MEAN")

        # Optional overlay: FLUXCOM regional mean (if computed)
        if COMPARE_SOURCE in ("fluxcom", "both") and VAR_NAME == "NEE":
            try:
                fc_series = np.array(region_fluxcom_series[name], dtype=float)
                ax.plot(x_time_plot, fc_series, color="tab:green", linewidth=1.5, label="FLUXCOM mean")
                fluxcom_any = True
            except Exception as e:
                print(f"Warning: failed to overlay FLUXCOM for region '{name}': {e}")

        # Optional overlay: all FluxNET site time series and regional mean (if available)
        if COMPARE_SOURCE in ("fluxnet", "both") and VAR_NAME == "NEE":
            try:
                site_info = REGION_SITES.get("regions", {}).get(name, {})
                # Include Global: use all known site ids if region is Global
                if name == "Global":
                    site_ids = list(REGION_SITES.get("site_id_to_file", {}).keys())
                else:
                    site_ids = site_info.get("SITE_IDS", [])

                plot_individual = (name != "Global")
                series_list = []
                for sid in site_ids:
                    fpath = REGION_SITES.get("site_id_to_file", {}).get(sid)
                    if fpath is None or not os.path.exists(fpath):
                        continue
                    df_csv = pd.read_csv(fpath)
                    ts_vals = pd.to_datetime(df_csv[CSV_T_COL].astype(str), format=CSV_T_FMT, errors="coerce")
                    y_vals = pd.to_numeric(df_csv[CSV_V_COL], errors="coerce")
                    qc_vals = pd.to_numeric(df_csv[CSV_QC_COL], errors="coerce") if CSV_QC_COL in df_csv.columns else None
                    # Base mask for valid timestamps/values and sentinel filtering, then apply QC if available
                    base_mask = ts_vals.notna() & y_vals.notna() & (y_vals != -9999)
                    if qc_vals is not None:
                        good_mask = base_mask & (qc_vals > QC_MIN)
                    else:
                        print(f"Warning: QC column '{CSV_QC_COL}' not found in {fpath}; using only non-missing values")
                        good_mask = base_mask
                    if not good_mask.any():
                        continue
                    # Replace low-quality or invalid values with NaN so lines break across gaps
                    y_plot = y_vals.astype(float).copy()
                    y_plot[~good_mask] = np.nan
                    # Restrict to requested time window
                    window_mask = (ts_vals >= START_DT) & (ts_vals <= END_DT)
                    if not window_mask.any():
                        continue
                    ts_vals_w = ts_vals[window_mask]
                    y_plot_w = y_plot[window_mask]
                    # Plot faint individual site lines (skip for Global)
                    if plot_individual:
                        ax.plot(ts_vals_w, y_plot_w, color="tab:red", alpha=0.15, linewidth=0.8, label=None)
                    # Collect for mean
                    s = pd.Series(y_plot_w.values, index=pd.DatetimeIndex(ts_vals_w.values))
                    series_list.append(s)

                if len(series_list) > 0:
                    df_reg = pd.concat(series_list, axis=1)
                    reg_mean = df_reg.mean(axis=1, skipna=True)
                    ax.plot(reg_mean.index, reg_mean.values, color="tab:red", alpha=0.6, linewidth=1.5, label="FluxNET sites")
                    fluxnet_any = True
            except Exception as e:
                print(f"Warning: failed to overlay FluxNET sites for region '{name}': {e}")

        ax.set_title(name)
        ax.set_xlabel("Year")
        ax.set_ylabel(f"{VAR_NAME} ({y_units_str})" if y_units_str else f"{VAR_NAME}")
        # ax.set_ylim(-2, 2)
        ax.set_xlim(START_DT, END_DT)
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
    # Ensure 'FluxNET sites' appears even if not in the first panel but present elsewhere
    if fluxnet_any:
        existing_labels = [lbl for _, lbl in zip(*axes[0].get_legend_handles_labels())]
        if "FluxNET sites" not in existing_labels:
            axes[0].plot([], [], color="tab:red", alpha=0.6, linewidth=1.5, label="FluxNET sites")
    if fluxcom_any:
        existing_labels = [lbl for _, lbl in zip(*axes[0].get_legend_handles_labels())]
        if "FLUXCOM mean" not in existing_labels:
            axes[0].plot([], [], color="tab:green", linewidth=1.5, label="FLUXCOM mean")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(loc="upper right", frameon=False)

    fig.suptitle(f"Monthly {VAR_NAME} (MEAN and Ensemble Uncertainty) {START_YEAR}-{END_YEAR} [{SPATIAL_AGG}]", fontsize=16)
    out_png = os.path.join(OUT_DIR, f"{VAR_NAME}_monthly_means_{START_YEAR}_{END_YEAR}.png")
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

    # Uncertainty shading and member lines only if any member file was found
    if any_member_found:
        members_arr = np.stack([region_member_series[region_name][mid] for mid in member_ids], axis=1)  # [ntime, nmembers]
        if np.isfinite(members_arr).any():
            low = np.nanmin(members_arr, axis=1)
            high = np.nanmax(members_arr, axis=1)
            ax.fill_between(x_time_plot, low, high, color="tab:blue", alpha=0.15, label="Member range")

            # Plot member lines (light, transparent)
            for j in range(members_arr.shape[1]):
                ax.plot(x_time_plot, members_arr[:, j], color="gray", alpha=0.15, linewidth=0.7)

    # Plot MEAN line (main)
    ax.plot(x_time_plot, mean_series, color="tab:blue", linewidth=1.8, label="MEAN")

    # Optional overlay: CSV time series and/or region FluxNET sites mean
    # If a single CSV is provided via CLI, plot that too (in orange), but also overlay all region sites and their mean if available
    if VAR_NAME == "NEE" and args.fluxnet_csv is not None and os.path.exists(args.fluxnet_csv):
        try:
            df_csv = pd.read_csv(args.fluxnet_csv)
            ts_vals = pd.to_datetime(df_csv[CSV_T_COL].astype(str), format=CSV_T_FMT, errors="coerce")
            y_vals = pd.to_numeric(df_csv[CSV_V_COL], errors="coerce")
            qc_vals = pd.to_numeric(df_csv[CSV_QC_COL], errors="coerce") if CSV_QC_COL in df_csv.columns else None
            base_mask = ts_vals.notna() & y_vals.notna() & (y_vals != -9999)
            if qc_vals is not None:
                good_mask = base_mask & (qc_vals > QC_MIN)
            else:
                print(f"Warning: QC column '{CSV_QC_COL}' not found in {args.fluxnet_csv}; using only non-missing values")
                good_mask = base_mask
            y_plot = y_vals.astype(float).copy()
            y_plot[~good_mask] = np.nan
            # Restrict to requested time window
            window_mask = (ts_vals >= START_DT) & (ts_vals <= END_DT)
            if window_mask.any():
                ts_vals_w = ts_vals[window_mask]
                y_plot_w = y_plot[window_mask]
                ax.plot(ts_vals_w, y_plot_w, color="tab:orange", linewidth=1.6, label="FluxNET CSV")
        except Exception as e:
            print(f"Warning: failed to overlay CSV '{args.fluxnet_csv}': {e}")

    # Optional overlay: FLUXCOM regional mean (if computed)
    if COMPARE_SOURCE in ("fluxcom", "both") and VAR_NAME == "NEE":
        try:
            fc_series = np.array(region_fluxcom_series[region_name], dtype=float)
            ax.plot(x_time_plot, fc_series, color="tab:green", linewidth=1.6, label="FLUXCOM mean")
        except Exception as e:
            print(f"Warning: failed to overlay FLUXCOM for region '{region_name}': {e}")

    # Overlay all region sites (including Global = all known sites) and plot their mean
    if COMPARE_SOURCE in ("fluxnet", "both") and VAR_NAME == "NEE":
        try:
            site_info = REGION_SITES.get("regions", {}).get(region_name, {})
            if region_name == "Global":
                site_ids = list(REGION_SITES.get("site_id_to_file", {}).keys())
            else:
                site_ids = site_info.get("SITE_IDS", [])

            plot_individual = (region_name != "Global")
            series_list = []
            for sid in site_ids:
                fpath = REGION_SITES.get("site_id_to_file", {}).get(sid)
                if fpath is None or not os.path.exists(fpath):
                    continue
                df_csv = pd.read_csv(fpath)
                ts_vals = pd.to_datetime(df_csv[CSV_T_COL].astype(str), format=CSV_T_FMT, errors="coerce")
                y_vals = pd.to_numeric(df_csv[CSV_V_COL], errors="coerce")
                qc_vals = pd.to_numeric(df_csv[CSV_QC_COL], errors="coerce") if CSV_QC_COL in df_csv.columns else None
                base_mask = ts_vals.notna() & y_vals.notna() & (y_vals != -9999)
                if qc_vals is not None:
                    good_mask = base_mask & (qc_vals > QC_MIN)
                else:
                    print(f"Warning: QC column '{CSV_QC_COL}' not found in {fpath}; using only non-missing values")
                    good_mask = base_mask
                if not good_mask.any():
                    continue
                # Replace low-quality or invalid values with NaN so lines break across gaps
                y_plot = y_vals.astype(float).copy()
                y_plot[~good_mask] = np.nan
                # Restrict to requested time window
                window_mask = (ts_vals >= START_DT) & (ts_vals <= END_DT)
                if not window_mask.any():
                    continue
                ts_vals_w = ts_vals[window_mask]
                y_plot_w = y_plot[window_mask]
                # Plot faint individual site lines (skip for Global)
                if plot_individual:
                    ax.plot(ts_vals_w, y_plot_w, color="tab:red", alpha=0.15, linewidth=0.8, label=None)
                # Collect for mean
                s = pd.Series(y_plot_w.values, index=pd.DatetimeIndex(ts_vals_w.values))
                series_list.append(s)


            if len(series_list) > 0:
                df_reg = pd.concat(series_list, axis=1)
                reg_mean = df_reg.mean(axis=1, skipna=True)
                ax.plot(reg_mean.index, reg_mean.values, color="tab:red", alpha=0.6, linewidth=1.5, label="FluxNET sites")
        except Exception as e:
            print(f"Warning: failed to overlay FluxNET sites for region '{region_name}': {e}")

    ax.set_title(f"{region_name}")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{VAR_NAME} ({y_units_str})" if y_units_str else f"{VAR_NAME}")
    # ax.set_ylim(-2, 2)
    ax.set_xlim(START_DT, END_DT)
    ax.xaxis.set_major_locator(YearLocator(base=5))
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(YearLocator(1))
    ax.grid(True, which="major", axis="both")
    ax.grid(True, which="minor", axis="x", linestyle=":", alpha=0.6)

    ax.legend(loc="upper right", frameon=False)

    safe_region = region_name.replace(" ", "_").replace("/", "-")
    out_png = os.path.join(OUT_DIR, f"{VAR_NAME}_monthly_{safe_region}_{START_YEAR}_{END_YEAR}.png")
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Saved {out_png}")

else:
    raise SystemExit(f"Unknown mode: {PLOT_MODE}")