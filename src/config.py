"""
Central configuration for PitcherGamePreds project.

All date ranges, seasons, and model parameters defined here.
"""

import datetime

# Seasons to collect data for
SEASONS = [2021, 2022, 2023, 2024]

# Full data range for Statcast pitch data
DATA_START = "2021-04-01"

# Set DATA_END to today:

DATA_END = datetime.datetime.today().strftime("%Y-%m-%d")

# Temporal split dates (for train/val/test)
VAL_DATE = "2025-06-01"
TEST_DATE = "2025-07-15"

# Rolling window sizes
PITCHER_ROLLING_WINDOWS = [5, 10, 20]    # By starts
BATTER_ROLLING_WINDOWS = [25, 50, 100]   # By games

# Output directories
DATA_DIR = "data"
MODELS_DIR = "models"

# Outcome classes for multi-class prediction
OUTCOME_CLASSES = ['K', 'BB', '1B', '2B', '3B', 'HR', 'OUT']
