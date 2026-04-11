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

# Exponential decay half-lives for profile features (in GAMES, not days)
# Games-based decay ignores offseason - only actual games played matter
# Shorter half-life = more recent data weighted more heavily
PITCHER_ARSENAL_HALF_LIFE = 60     # Velo/spin/movement (~2 seasons of starts)
PITCHER_USAGE_HALF_LIFE = 30       # Pitch usage percentages (~1 season)
PITCHER_PERF_HALF_LIFE = 15        # Whiff%, zone%, etc. (~half season)
BATTER_PERF_HALF_LIFE = 75         # Whiff%, chase%, etc. (~half season)
BATTER_BATTED_BALL_HALF_LIFE = 100 # Exit velo, xwOBA, etc. (~2/3 season)

# Output directories
DATA_DIR = "data"
MODELS_DIR = "models"

# Outcome classes for multi-class prediction
OUTCOME_CLASSES = ['K', 'BB', '1B', '2B', '3B', 'HR', 'OUT']
