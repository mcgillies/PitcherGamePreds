"""
Central configuration for PitcherGamePreds project.

All date ranges, seasons, and model parameters defined here.
"""

# Seasons to collect data for
SEASONS = [2021, 2022, 2023, 2024]

# Full data range for Statcast pitch data
DATA_START = "2021-04-01"
DATA_END = "2024-09-30"

# Temporal split dates (for train/val/test)
VAL_DATE = "2024-07-01"
TEST_DATE = "2024-08-15"

# Rolling window sizes
PITCHER_ROLLING_WINDOWS = [5, 10, 20]    # By starts
BATTER_ROLLING_WINDOWS = [25, 50, 100]   # By games

# Output directories
DATA_DIR = "data"
MODELS_DIR = "models"

# Outcome classes for multi-class prediction
OUTCOME_CLASSES = ['K', 'BB', '1B', '2B', '3B', 'HR', 'OUT']
