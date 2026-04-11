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

# Exponential decay half-lives for profile features (in days)
# Shorter half-life = more recent data weighted more heavily
PITCHER_ARSENAL_HALF_LIFE = 365    # Velo/spin/movement (stable)
PITCHER_USAGE_HALF_LIFE = 180      # Pitch usage percentages
PITCHER_PERF_HALF_LIFE = 90        # Whiff%, zone%, etc.
BATTER_PERF_HALF_LIFE = 90         # Whiff%, chase%, etc.
BATTER_BATTED_BALL_HALF_LIFE = 120 # Exit velo, xwOBA, etc.

# Output directories
DATA_DIR = "data"
MODELS_DIR = "models"

# Outcome classes for multi-class prediction
OUTCOME_CLASSES = ['K', 'BB', '1B', '2B', '3B', 'HR', 'OUT']
