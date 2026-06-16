#!/bin/bash
# Daily pipeline wrapper - runs each step as separate subprocess to avoid memory pressure
#
# Each step runs in its own Python process, so memory is fully freed between steps.
# This prevents macOS from killing the process due to accumulated memory usage.

set -e  # Exit on first error

LOG_FILE="/Users/matthewgillies/PitcherGamePreds/logs/daily_pipeline_$(date +%Y%m%d).log"
mkdir -p /Users/matthewgillies/PitcherGamePreds/logs

PYTHON="/Applications/miniconda3/envs/mlbenv/bin/python"
SCRIPT="/Users/matthewgillies/PitcherGamePreds/scripts/daily_pipeline.py"

cd /Users/matthewgillies/PitcherGamePreds

echo "============================================================" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting daily pipeline (subprocess mode)" >> "$LOG_FILE"
echo "============================================================" >> "$LOG_FILE"

# Step 1: Update Statcast profiles
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 1/5: Updating Statcast profiles..." >> "$LOG_FILE"
$PYTHON "$SCRIPT" --step profiles >> "$LOG_FILE" 2>&1

# Step 2: Update rolling stats
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 2/5: Updating rolling stats..." >> "$LOG_FILE"
$PYTHON "$SCRIPT" --step rolling >> "$LOG_FILE" 2>&1

# Step 3: Retrain models
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 3/5: Retraining models..." >> "$LOG_FILE"
$PYTHON "$SCRIPT" --step retrain >> "$LOG_FILE" 2>&1

# Step 4: Settle auto bets
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 4/5: Settling auto bets..." >> "$LOG_FILE"
$PYTHON "$SCRIPT" --step settle >> "$LOG_FILE" 2>&1

# Step 5: Restart Streamlit app
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 5/5: Restarting Streamlit app..." >> "$LOG_FILE"
$PYTHON "$SCRIPT" --step restart >> "$LOG_FILE" 2>&1

# Write success marker
echo "$(date -Iseconds)" > /Users/matthewgillies/PitcherGamePreds/logs/pipeline_last_success.txt

# Clear failure marker if exists
rm -f /Users/matthewgillies/PitcherGamePreds/logs/pipeline_failure.txt

echo "============================================================" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline complete!" >> "$LOG_FILE"
echo "============================================================" >> "$LOG_FILE"
