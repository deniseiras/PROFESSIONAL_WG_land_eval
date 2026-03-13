#!/usr/bin/env python3
"""
Compute ensemble-average NEE for a given year-month from per-member monthly files.

Input files (produced by create_NEE_month_mean_in_days.bash):
  ./data/out/NEE_monthsum_YYYY_MM_MEMBER.nc

Output file:
  ./data/out/NEE_monthsum_YYYY_MM_avg.nc

Notes:
- Only files whose MEMBER matches exactly four digits (e.g., 0001) are included in the
  ensemble average. Any files like ..._mean.nc or ..._avg.nc are ignored.
- The variable "NEE" is averaged across members; spatial/temporal coordinates are
  taken from the first valid file.
- Missing or invalid files are skipped with a warning. If no valid members are found,
  the script exits with an error.

Example:
  ./create_NEE_member_monthly_avg.py --year 2003 --month 06
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from datetime import datetime
from typing import List, Tuple

import numpy as np
import xarray as xr


def find_member_files(in_dir: str, prefix: str, year: int, month: int) -> List[Tuple[str, str]]:
    """Return list of (member, filepath) for four-digit members for given year-month."""
    ym = f"{year:04d}_{month:02d}"
    pattern = os.path.join(in_dir, f"{prefix}_{ym}_*.nc")
    paths = sorted(glob.glob(pattern))

    member_re = re.compile(rf"{re.escape(prefix)}_{ym}_(\d{{4}})\.nc$")
    member_files: List[Tuple[str, str]] = []
    for p in paths:
        m = member_re.search(os.path.basename(p))
        if m:
            member_files.append((m.group(1), p))
    return sorted(member_files, key=lambda t: t[0])


def open_member_dataarray(path: str, var: str) -> xr.DataArray | None:
    """Open a single member file and return the target variable as a DataArray.

    Returns None if the variable is missing or file cannot be read.
    """
    try:
        ds = xr.open_dataset(path, decode_times=True, mask_and_scale=True)
    except Exception as e:
        print(f"WARNING: Failed to open {path}: {e}", file=sys.stderr)
        return None

    if var not in ds.variables:
        print(f"WARNING: Variable '{var}' not found in {path}, skipping.", file=sys.stderr)
        ds.close()
        return None

    da = ds[var]
    # Ensure we actually load values before closing (xarray keeps file open otherwise)
    try:
        da.load()
    except Exception as e:
        print(f"WARNING: Failed to load data from {path}: {e}", file=sys.stderr)
        ds.close()
        return None

    # Detach from file-backed dataset to avoid keeping references
    da = da.copy()
    ds.close()
    return da


def ensemble_mean(da_list: List[Tuple[str, xr.DataArray]], var: str) -> xr.DataArray:
    """Concatenate DataArrays along a new 'member' dimension and compute mean."""
    # Align all arrays to the coordinates of the first array to avoid misalignment issues
    ref = da_list[0][1]
    aligned = []
    member_ids = []
    for member, da in da_list:
        try:
            da_aligned = da.broadcast_like(ref)
        except Exception:
            # Fallback to xarray align for more robust handling
            da_aligned, _ = xr.align(da, ref, join="exact")
        aligned.append(da_aligned.expand_dims({"member": [member]}))
        member_ids.append(member)

    stacked = xr.concat(aligned, dim="member")

    # Preserve attributes sensibly
    attrs = dict(ref.attrs)
    long_name = attrs.get("long_name", var)
    attrs["long_name"] = f"{long_name} (ensemble mean)"
    attrs["cell_methods"] = (
        (attrs.get("cell_methods", "") + " ").strip() + " member: mean"
    ).strip()

    mean_da = stacked.mean(dim="member", skipna=True)
    mean_da.attrs = attrs

    # Add helpful coordinate for member IDs used (as an attribute on the variable)
    mean_da.attrs["ensemble_members"] = ",".join(member_ids)

    return mean_da


def build_output_dataset(mean_da: xr.DataArray, src_example: xr.DataArray, var: str) -> xr.Dataset:
    """Build an output dataset containing only the averaged variable and sensible attrs."""
    ds_out = xr.Dataset({var: mean_da})

    # Try to preserve relevant coordinates present on the example variable
    for coord_name in src_example.coords:
        if coord_name not in ds_out.coords and coord_name in src_example.dims:
            ds_out = ds_out.assign_coords({coord_name: src_example.coords[coord_name]})

    return ds_out


def main():
    p = argparse.ArgumentParser(description="Compute ensemble-mean NEE for a given year-month.")
    p.add_argument("--year", type=int, required=True, help="Year, e.g., 2003")
    p.add_argument("--month", type=str, required=True, help="Month as 2-digit string, e.g., 06")
    p.add_argument("--in-dir", default="./data/out", help="Input directory containing per-member files")
    p.add_argument("--out-dir", default="./data/out", help="Directory to write the averaged file")
    p.add_argument("--prefix", default="NEE_monthsum", help="Filename prefix before YYYY_MM, default: NEE_monthsum")
    p.add_argument("--var", default="NEE", help="Variable name to average, default: NEE")
    p.add_argument("--force", action="store_true", help="Overwrite existing output file")

    args = p.parse_args()

    try:
        month_int = int(args.month)
    except ValueError:
        print("ERROR: --month must be an integer-like string such as 06 or 6", file=sys.stderr)
        sys.exit(2)

    ym = f"{args.year:04d}_{month_int:02d}"

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.prefix}_{ym}_avg.nc")

    if os.path.exists(out_path) and not args.force:
        print(f"Output already exists: {out_path} (use --force to overwrite)")
        return

    member_files = find_member_files(args.in_dir, args.prefix, args.year, month_int)
    if not member_files:
        print(
            f"ERROR: No member files found matching {args.in_dir}/{args.prefix}_{ym}_####.nc",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Found {len(member_files)} member files for {ym}:")
    for m, pth in member_files:
        print(f"  {m}: {pth}")

    valid: List[Tuple[str, xr.DataArray]] = []
    for member, path in member_files:
        da = open_member_dataarray(path, args.var)
        if da is None:
            continue
        valid.append((member, da))

    if not valid:
        print(f"ERROR: No valid files contained variable '{args.var}'.", file=sys.stderr)
        sys.exit(1)

    mean_da = ensemble_mean(valid, args.var)

    # Build output dataset and attributes
    ds_out = build_output_dataset(mean_da, valid[0][1], args.var)

    # Compose global attributes: start from first file's attrs if available
    global_attrs = {}
    try:
        with xr.open_dataset(member_files[0][1], decode_times=False) as ds0:
            global_attrs.update({k: str(v) for k, v in ds0.attrs.items()})
    except Exception:
        pass

    history_line = (
        f"{datetime.utcnow().isoformat(timespec='seconds')}Z: ensemble mean over members "
        f"[{','.join([m for m, _ in valid])}] -> {os.path.basename(out_path)}"
    )
    if "history" in global_attrs and global_attrs["history"]:
        global_attrs["history"] = f"{global_attrs['history']}\n{history_line}"
    else:
        global_attrs["history"] = history_line

    ds_out = ds_out.assign_attrs(global_attrs)

    # Encoding: compress and use float32 by default
    enc = {}
    # Try to preserve _FillValue from source variable if present
    src_enc_fill = valid[0][1].encoding.get("_FillValue")
    fill_value = src_enc_fill if src_enc_fill is not None else np.float32(np.nan)
    enc[args.var] = {
        "zlib": True,
        "complevel": 4,
        "dtype": "float32",
        "_FillValue": fill_value,
    }

    try:
        ds_out.to_netcdf(out_path, encoding=enc)
    except Exception as e:
        print(f"ERROR: Failed to write {out_path}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote ensemble-mean file: {out_path}")


if __name__ == "__main__":
    main()
