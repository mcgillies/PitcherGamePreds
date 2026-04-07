#!/bin/bash
# Daily pipeline wrapper - activates conda and runs pipeline

LOG_FILE="/Users/matthewgillies/PitcherGamePreds/logs/daily_pipeline_$(date +%Y%m%d).log"
mkdir -p /Users/matthewgillies/PitcherGamePreds/logs

# Activate conda
source /Applications/miniconda3/etc/profile.d/conda.sh
conda activate mlbenv

# Run pipeline
cd /Users/matthewgillies/PitcherGamePreds
/Applications/miniconda3/envs/mlbenv/bin/python scripts/daily_pipeline.py --retrain >> "$LOG_FILE" 2>&1

echo "Pipeline finished at $(date)" >> "$LOG_FILE"
