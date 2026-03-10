#!/bin/bash

HERE=$PWD
cd $HERE

# SCRIPT_NAME="./create_NEE_month_mean_in_days.bash"
SCRIPT_NAME="plot_NEE_regions.py"

LOG_FILE="${HERE}/${SCRIPT_NAME}_$(date +%Y%m%d%H%M%S).log"
TOTALPES=1
TASKS_PER_NODE=1
PROJECT=0710
bsub -K \
     -P ${PROJECT} \
     -J "${SCRIPT_NAME}" \
     -W 01:00 \
     -o "${LOG_FILE}" \
     -e "${LOG_FILE}" \
     -n "${TOTALPES}" \
     -q s_short \
     -R "rusage[mem=10G]" \
     < ${SCRIPT_NAME}
     # -x \
     # -R "span[ptile=${TASKS_PER_NODE}]" \