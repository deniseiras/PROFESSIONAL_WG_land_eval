#!/bin/bash

set -euo pipefail

# Parameters
YEAR_START=${YEAR_START:-2002}
YEAR_END=${YEAR_END:-2022}
MEMBERS_SELECT=${MEMBERS_SELECT:-all}
# MEMBERS_SELECT=${MEMBERS_SELECT:-????}  # default to ensemble mean placeholder

HERE=$PWD
cd "$HERE"

SCRIPT_NAME="./create_NEE_month_mean_in_days.bash"
SCRIPT_BASENAME=$(basename "$SCRIPT_NAME")

TOTALPES=1
TASKS_PER_NODE=1
export OMP_NUM_THREADS=4
PROJECT=0710

if (( YEAR_START > YEAR_END )); then
  echo "YEAR_START (${YEAR_START}) is greater than YEAR_END (${YEAR_END}); nothing to submit." >&2
  exit 1
fi

for year in $(seq "$YEAR_START" "$YEAR_END"); do
  TIMESTAMP=$(date +%Y%m%d%H%M%S)
  LOG_FILE="${HERE}/${SCRIPT_BASENAME}_${year}_${TIMESTAMP}.log"
  JOB_NAME="${SCRIPT_BASENAME}_${year}"
  echo "Submitting ${JOB_NAME} for year ${year} (log: ${LOG_FILE})"
  YEAR_START="$year" YEAR_END="$year" \
  bsub -K \
       -P "${PROJECT}" \
       -J "${JOB_NAME}" \
       -W 02:00 \
       -o "${LOG_FILE}" \
       -e "${LOG_FILE}" \
       -n "${TOTALPES}" \
       -q p_short \
       -R "rusage[mem=10G]" \
       < "${SCRIPT_NAME}" &
       # -x \
       # -R "span[ptile=${TASKS_PER_NODE}]" \
done