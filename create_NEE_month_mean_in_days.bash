#!/bin/bash

set -euo pipefail
shopt -s nullglob

# switch_user spreads-lnd
module load intel-2021.6.0/cdo-threadsafe/2.1.1-lyjsw
# Mitigate HDF5 attribute access issues and avoid multi-thread races
# export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
# export HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING:-FALSE}
export HDF5_DISABLE_ERROR_STACK=${HDF5_DISABLE_ERROR_STACK:-1}

# Year range (parametrize here)
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
MEMBERS_MEAN=("????")

# MEMBERS_SELECT can be provided via env var or as 3rd positional arg:
# - 'all' for all members
# - '????' or 'mean' for the ensemble mean placeholder
MEMBERS_SELECT="${MEMBERS_SELECT:-all}"
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

MAX_PARALLEL_CDO=36  # adjust to control how many cdo jobs run at once


BASE_ROOT="/data/products/CERISE-LND-REANALYSIS/archive/streams/final_archive"
now=$(date +%Y%m%d%H%M%S)
NOT_FOUND_LOG="not_found_files${now}.log"

mkdir -p "$OUT_DIR"
: > "$NOT_FOUND_LOG"

wait_for_cdo_slots() {
  # Wait until the number of running background jobs is below MAX_PARALLEL_CDO
  while (( $(jobs -pr | wc -l) >= MAX_PARALLEL_CDO )); do
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
      # Leap year adjustment for February
      if (( month == 2 )); then
        if (((10#$year % 400 == 0) || (10#$year % 4 == 0 && 10#$year % 100 != 0))); then
          days=29
        fi
      fi

      files=()
      # days=2
      # for day in $(seq -w 1 "$days"); do.  == Causing strage bug for cdo !
      for ((day=START_DAY; day<=days; day++)); do
        printf -v day_str "%02d" "$day"  # zero-pad day without altering arithmetic variable
        day_dir="${BASE_ROOT}/${year}/output_history_${year}-${month_str}-${day_str}"
        pattern="${day_dir}/*.clm2_${member}.h0.${year}-${month_str}-${day_str}-00000*.nc"
        echo "Looking for files matching: ${pattern}"
        found_any=false
        for src in $pattern; do
          if [[ -f "$src" ]]; then
            found_any=true
            files+=("$src")
            # ncdump -h "$src" | grep "time ="
            # ncdump -h "$src" | grep "NEE"
          fi
        done
        if [[ "$found_any" = false ]]; then
          echo "MISSING ${year}-${month_str}-${day_str}: ${pattern}" >> "$NOT_FOUND_LOG"
        fi
      done

      if [[ ${#files[@]} -eq 0 ]]; then
        echo "No files found for ${year}-${month_str}, skipping." >&2
        continue
      fi

      # Keep only files that contain NEE and have a time axis
      valid_files=()
      for src in "${files[@]}"; do
        if cdo -s showname "$src" 2>/dev/null | grep -qw "NEE"; then
          if [[ -n "$(cdo -s showtimestamp "$src" 2>/dev/null)" ]]; then
            valid_files+=("$src")
          else
            echo "SKIPPING (no time axis) ${src}" >> "$NOT_FOUND_LOG"
          fi
        else
          echo "SKIPPING (missing NEE) ${src}" >> "$NOT_FOUND_LOG"
        fi
      done

      if [[ ${#valid_files[@]} -eq 0 ]]; then
        echo "No files containing NEE for ${year}-${month_str}, skipping." >&2
        continue
      fi

      if [[ "$member" == "????" ]]; then
        member_string="mean"
      else
        member_string="$member"
      fi
      # out="${OUT_DIR}/NEE_monthmean_${year}_${month_str}_${member_string}.nc"
      out="${OUT_DIR}/NEE_monthsum_${year}_${month_str}_${member_string}.nc"
      # echo "Calculating monthly mean for ${year}-${month_str}, member ${member_string} from ${#valid_files[@]} daily files..."
      echo "Calculating monthly sum for ${year}-${month_str}, member ${member_string} from ${#valid_files[@]} daily files..."
      wait_for_cdo_slots
      cdo_inputs=()
      for src in "${valid_files[@]}"; do
        cdo_inputs+=(-selname,NEE "$src")
      done
      # cdo -P 4 -L -s -O -setattribute,NEE@units="gC m-2 day-1",NEE@long_name="net ecosystem exchange of carbon (monthly mean gC m-2 day-1)" \
      #   -mulc,86400 -monmean -mergetime "${cdo_inputs[@]}" "$out" \
      #   &
      cdo -P 4 -L -s -O -setattribute,NEE@units="gC m-2 day-1",NEE@long_name="net ecosystem exchange of carbon (monthly sum gC m-2 day-1)" \
        -mulc,86400 -monsum -mergetime "${cdo_inputs[@]}" "$out" \
        &

    done
  done
done

wait

: # no temp files to remove
echo "All done. Check $NOT_FOUND_LOG for any missing files."
