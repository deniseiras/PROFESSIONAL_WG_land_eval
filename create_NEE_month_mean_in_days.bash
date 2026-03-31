#!/bin/bash

set -euo pipefail
shopt -s nullglob

# switch_user spreads-lnd
module load intel-2021.6.0/cdo-threadsafe/2.1.1-lyjsw
# Mitigate HDF5 attribute access issues and avoid multi-thread races
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING:-FALSE}
export HDF5_DISABLE_ERROR_STACK=${HDF5_DISABLE_ERROR_STACK:-1}

# PARAMETERS
MAX_PARALLEL_CDO=4  # adjust to control how many cdo jobs run at once
OUT_DIR="./data/out"
DEFAULT_YEAR_START=2002
DEFAULT_YEAR_END=2022

YEAR_START="${YEAR_START:-$DEFAULT_YEAR_START}"
YEAR_END="${YEAR_END:-$DEFAULT_YEAR_END}"

if [[ $# -ge 1 ]]; then
  YEAR_START="$1"
  if [[ $# -ge 2 ]]; then
    YEAR_END="$2"
  else
    YEAR_END="$YEAR_START"
  fi
fi

echo "Processing years ${YEAR_START}-${YEAR_END}"

if (( YEAR_START > YEAR_END )); then
  echo "YEAR_START (${YEAR_START}) is greater than YEAR_END (${YEAR_END}); nothing to process." >&2
  exit 1
fi

# the files 2006-01-01, 2007-01-01 are bugged files has more variables and also does not have some global atributes, so we exclude them from the mean calculation
# and log them as missing (since they are not usable for our purpose) 
# ERROR: cdo(3) select (Abort): Input streams have different number of variables per timestep! terminate called recursively terminate called without an active exception 
START_DAY=1  # files 2006-01-01, 2007-01-01 bugged
START_MONTH=1
LAST_MONTH=12     # files 2006-01-01, 2007-01-01 bugged

# LAST_MONTH="01". - CDO bug for this month using the for with seq -w, so we use the variable instead !! BIZARR
# Members selection (parametrize here). Choose 'all' (30 members) or '????' (ensemble mean placeholder)
MEMBERS_ALL=("0001" "0002" "0003" "0004" "0005" "0006" "0007" "0008" "0009" "0010" "0011" "0012" "0013" "0014" "0015" "0016" "0017" "0018" "0019" "0020" "0021" "0022" "0023" "0024" "0025" "0026" "0027" "0028" "0029" "0030")
# MEMBERS_ALL=("0001" "0002" "0003")
MEMBERS_MEAN=("????")

# MEMBERS_SELECT can be provided via env var or as 3rd positional arg:
# - 'all' for all members
# - '????' or 'mean' for the ensemble mean placeholder
MEMBERS_SELECT="${MEMBERS_SELECT:-????}"
if [[ $# -ge 3 ]]; then
  case "$3" in
    all|ALL) MEMBERS_SELECT="all" ;;
    "????"|mean|MEAN) MEMBERS_SELECT="????" ;;
    *) echo "Invalid third argument for members selection: '$3'. Use 'all' or '????' (or 'mean')." >&2; exit 2 ;;
  esac
fi

case "$MEMBERS_SELECT" in
  all|ALL) MEMBERS=("${MEMBERS_ALL[@]}") ;;
  "????"|mean|MEAN) MEMBERS=("${MEMBERS_MEAN[@]}") ;;
  *) echo "Invalid MEMBERS_SELECT='$MEMBERS_SELECT'. Use 'all' or '????' (or 'mean')." >&2; exit 2 ;;
esac

# Variables to process. Override via environment variable VARS (space- or comma-separated list), e.g.:
#   VARS="NEE GPP NPP" ./create_NEE_month_mean_in_days.bash
#   VARS="NEE,GPP,NPP" ./create_NEE_month_mean_in_days.bash
VARIABLES=("NEE")
if [[ -n "${VARS:-}" ]]; then
  IFS=', ' read -r -a VARIABLES <<< "${VARS}"
fi

BASE_ROOT="/data/products/CERISE-LND-REANALYSIS/archive/streams/final_archive"
now=$(date +%Y%m%d%H%M%S)
NOT_FOUND_LOG="not_found_files${now}.log"

mkdir -p "$OUT_DIR"
: > "$NOT_FOUND_LOG"

wait_for_cdo_slots() {
  # Wait until the number of running background jobs is below MAX_PARALLEL_CDO
  # Be robust to 'jobs' behavior in non-interactive shells and pipefail
  local njobs
  while :; do
    njobs=$(jobs -pr 2>/dev/null | wc -l || true)
    # strip whitespace
    njobs=${njobs//[[:space:]]/}
    if (( njobs < MAX_PARALLEL_CDO )); then
      break
    fi
    sleep 0.2
  done
}

# Iterate year->month and directly find daily files under BASE_ROOT; log missing days
# then compute the monthly mean with cdo (parallelized up to MAX_PARALLEL_CDO)
days_in_month=(31 28 31 30 31 30 31 31 30 31 30 31)

for member in "${MEMBERS[@]}"; do
  echo "Processing member ${member}..."
  for year in $(seq "$YEAR_START" "$YEAR_END"); do
    for ((month=$START_MONTH; month<=$LAST_MONTH; month++)); do
      printf -v month_str "%02d" "$month"
      idx=$((month - 1))
      days="${days_in_month[$idx]}"
      files=()
      # days=2
      # for day in $(seq -w 1 "$days"); do.  == Causing strage bug for cdo !
      for ((day=START_DAY; day<=days; day++)); do
        printf -v day_str "%02d" "$day"  # zero-pad day without altering arithmetic variable
        day_dir="${BASE_ROOT}/${year}/output_history_${year}-${month_str}-${day_str}"
        # First, determine the stream prefix by probing member 0001 only
        echo "Looking for files in: ${day_dir}"
        # If the day directory is missing, log and continue quickly
        if [[ ! -d "$day_dir" ]]; then
          echo "MISSING ${year}-${month_str}-${day_str}: ${day_dir} (directory missing)" >> "$NOT_FOUND_LOG"
          continue
        fi

        probe_pattern="${day_dir}/*.clm2_0001.h0.${year}-${month_str}-${day_str}-00000*.nc"
        echo "Probing prefix with: ${probe_pattern}"
        probe_found=false
        probe_newest=""
        probe_count=0
        for src in $probe_pattern; do
          if [[ -f "$src" ]]; then
            probe_found=true
            ((++probe_count))
            if [[ -z "$probe_newest" || "$src" -nt "$probe_newest" ]]; then
              probe_newest="$src"
            fi
          fi
        done
        if [[ "$probe_found" = false ]]; then
          echo "MISSING ${year}-${month_str}-${day_str}: ${probe_pattern} (no member 0001 to determine prefix)" >> "$NOT_FOUND_LOG"
          continue
        fi
        if (( probe_count > 1 )); then
          echo "Found ${probe_count} candidate prefixes; selecting newest: ${probe_newest}"
        fi
        # Extract prefix path up to 'clm2_0001.h0'
        prefix_path="${probe_newest%clm2_0001.h0.*}"

        pattern="${prefix_path}clm2_${member}.h0.${year}-${month_str}-${day_str}-00000*.nc"
        echo "Using prefix $prefix_path"
        echo "Looking for: ${pattern}"

        found_any=false
        newest=""
        count=0
        for src in $pattern; do
          if [[ -f "$src" ]]; then
            found_any=true
            ((++count))
            files+=("$src")
            # ncdump -h "$src" | grep "time ="
            # ncdump -h "$src" | grep "NEE"
          fi
        done

        if [[ "$found_any" = true ]]; then
          echo "Found ${count} files with prefix $prefix_path "
        else
          echo "MISSING ${year}-${month_str}-${day_str}: ${pattern}" >> "$NOT_FOUND_LOG"
        fi
      done

      if [[ ${#files[@]} -eq 0 ]]; then
        echo "No files found for ${year}-${month_str}, skipping." >&2
        continue
      fi

      if [[ "$member" == "????" ]]; then
        member_string="mean"
      else
        member_string="$member"
      fi

      # Process all requested variables together
      # Build CSV and underscore-joined variable lists
      var_csv=$(IFS=,; echo "${VARIABLES[*]}")
      var_us=$(IFS=_; echo "${VARIABLES[*]}")

      # Keep only files that contain all variables and have a time axis
      valid_files=()
      for src in "${files[@]}"; do
        # Check time axis
        if [[ -z "$(cdo -s showtimestamp "$src" 2>/dev/null)" ]]; then
          echo "SKIPPING (no time axis) ${src}" >> "$NOT_FOUND_LOG"
          continue
        fi
        # Check that file contains all requested variables
        names="$(cdo -s showname "$src" 2>/dev/null || true)"
        missing_vars=()
        for v in "${VARIABLES[@]}"; do
          if ! grep -qw "$v" <<< "$names"; then
            missing_vars+=("$v")
          fi
        done
        if [[ ${#missing_vars[@]} -eq 0 ]]; then
          valid_files+=("$src")
        else
          echo "SKIPPING (missing vars: ${missing_vars[*]}) ${src}" >> "$NOT_FOUND_LOG"
        fi
      done

      if [[ ${#valid_files[@]} -eq 0 ]]; then
        echo "No files containing all requested variables (${var_csv}) for ${year}-${month_str}, skipping." >&2
        continue
      fi

      out="${OUT_DIR}/${var_us}_monthmean_${year}_${month_str}_${member_string}.nc"
      # if file $out exists, skip processing (to allow re-running without overwriting existing results)
      if [[ -f "$out" ]]; then
        echo "Output file $out already exists, skipping ${year}-${month_str} member ${member_string} vars ${var_csv}."
        continue
      fi

      echo "Calculating monthly sum for ${year}-${month_str}, member ${member_string}, vars ${var_csv} from ${#valid_files[@]} daily files..."
      wait_for_cdo_slots
      # Build setattribute string for each variable
      attr_parts=()
      for v in "${VARIABLES[@]}"; do
        attr_parts+=("${v}@units=\"gC m-2 day-1\"")
        attr_parts+=("${v}@long_name=\"${v} (monthly sum gC m-2 day-1)\"")
      done
      attr_arg=$(IFS=,; echo "${attr_parts[*]}")

      cdo_inputs=()
      for src in "${valid_files[@]}"; do
        cdo_inputs+=(-selname,"$var_csv" "$src")
      done
      cdo -P 1 -L -s -O -setattribute,"$attr_arg" \
        -mulc,86400 -monmean -mergetime "${cdo_inputs[@]}" "$out" \
        &
    done
  done
done

wait

: # no temp files to remove
echo "All done. Check $NOT_FOUND_LOG for any missing files."
