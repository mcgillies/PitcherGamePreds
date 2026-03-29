"""
Feature engineering for pitcher strikeout prediction.

Computes rolling averages, rest days, and temporal features.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def parse_game_date(date_str: str) -> datetime:
    """Parse date string from pybaseball format."""
    # Handle formats like "Apr 30, 2023" or "2024-04-30"
    for fmt in ["%b %d, %Y", "%Y-%m-%d"]:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Could not parse date: {date_str}")


def compute_rest_days(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute days of rest since last start for each pitcher.

    Args:
        games_df: DataFrame with 'Name' and 'game_date' columns

    Returns:
        DataFrame with 'rest_days' column added
    """
    df = games_df.copy()

    # Parse dates
    df['date_parsed'] = df['game_date'].apply(parse_game_date)
    df = df.sort_values(['Name', 'date_parsed'])

    # Compute days since last appearance for each pitcher
    df['prev_game_date'] = df.groupby('Name')['date_parsed'].shift(1)
    df['rest_days'] = (df['date_parsed'] - df['prev_game_date']).dt.days

    # Fill NaN (first appearance) with a default value
    df['rest_days'] = df['rest_days'].fillna(7)  # Assume normal rest for first game

    # Cap extreme values
    df['rest_days'] = df['rest_days'].clip(upper=30)

    df = df.drop(columns=['date_parsed', 'prev_game_date'])

    return df


def compute_rolling_stats(
    games_df: pd.DataFrame,
    stat_columns: list[str],
    windows: list[int] = [3, 5, 10],
    min_periods: int = 1,
) -> pd.DataFrame:
    """
    Compute rolling averages for pitcher stats over recent games.

    Uses shift(1) to avoid data leakage - only uses games BEFORE the current one.

    Args:
        games_df: DataFrame with pitcher game logs
        stat_columns: Columns to compute rolling stats for
        windows: Window sizes for rolling averages (e.g., last 3, 5, 10 games)
        min_periods: Minimum games required for a valid average

    Returns:
        DataFrame with rolling stat columns added
    """
    df = games_df.copy()

    # Parse and sort by date
    df['date_parsed'] = df['game_date'].apply(parse_game_date)
    df = df.sort_values(['Name', 'date_parsed'])

    # Filter to only columns that exist
    available_stats = [c for c in stat_columns if c in df.columns]

    for stat in available_stats:
        for window in windows:
            col_name = f"{stat}_roll{window}"
            # Shift by 1 to exclude current game (avoid leakage)
            df[col_name] = (
                df.groupby('Name')[stat]
                .shift(1)
                .rolling(window=window, min_periods=min_periods)
                .mean()
                .reset_index(level=0, drop=True)
            )

    df = df.drop(columns=['date_parsed'])

    return df


def compute_season_to_date_stats(
    games_df: pd.DataFrame,
    stat_columns: list[str],
) -> pd.DataFrame:
    """
    Compute season-to-date cumulative stats at time of each game.

    Uses expanding window with shift to avoid leakage.

    Args:
        games_df: DataFrame with pitcher game logs
        stat_columns: Columns to compute cumulative stats for

    Returns:
        DataFrame with STD (season-to-date) columns added
    """
    df = games_df.copy()

    # Parse and sort by date
    df['date_parsed'] = df['game_date'].apply(parse_game_date)
    df = df.sort_values(['Name', 'date_parsed'])

    available_stats = [c for c in stat_columns if c in df.columns]

    for stat in available_stats:
        col_name = f"{stat}_std"  # season-to-date
        # Expanding mean up to (but not including) current game
        df[col_name] = (
            df.groupby('Name')[stat]
            .shift(1)
            .expanding(min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    df = df.drop(columns=['date_parsed'])

    return df


def compute_games_started_count(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute number of games started in current season at time of game.

    Args:
        games_df: DataFrame with pitcher game logs

    Returns:
        DataFrame with 'games_started_season' column added
    """
    df = games_df.copy()

    df['date_parsed'] = df['game_date'].apply(parse_game_date)
    df = df.sort_values(['Name', 'date_parsed'])

    # Count games up to (not including) current
    df['games_started_season'] = df.groupby('Name').cumcount()

    df = df.drop(columns=['date_parsed'])

    return df


def add_home_away(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add home/away indicator based on the '@' symbol in game data.

    Args:
        games_df: DataFrame with game logs

    Returns:
        DataFrame with 'is_home' column added
    """
    df = games_df.copy()

    # pybaseball uses '@' column or similar to indicate away games
    if '@' in df.columns:
        df['is_home'] = df['@'].isna() | (df['@'] == '')
    elif 'Unnamed: 6' in df.columns:
        df['is_home'] = df['Unnamed: 6'].isna() | (df['Unnamed: 6'] == '')
    else:
        # Default to None if we can't determine
        df['is_home'] = True

    df['is_home'] = df['is_home'].astype(int)

    return df


def filter_starters_only(games_df: pd.DataFrame) -> pd.DataFrame:
    """Filter to only starting pitcher appearances."""
    if 'GS' in games_df.columns:
        return games_df[games_df['GS'] == 1].copy()
    return games_df


def build_features(
    pitcher_games: pd.DataFrame,
    team_batting: pd.DataFrame,
    rolling_windows: list[int] = [3, 5],
) -> pd.DataFrame:
    """
    Build full feature set for model training.

    Args:
        pitcher_games: Daily pitcher game logs
        team_batting: Team batting statistics
        rolling_windows: Windows for rolling averages

    Returns:
        DataFrame with all features and target (K)
    """
    # Start with starters only
    df = filter_starters_only(pitcher_games)

    # Stats to compute rolling averages for
    rolling_stats = ['SO', 'IP', 'H', 'BB', 'ER', 'K/9', 'WHIP', 'SwStr%']

    # Compute temporal features
    print("Computing rest days...")
    df = compute_rest_days(df)

    print("Computing rolling averages...")
    df = compute_rolling_stats(df, rolling_stats, windows=rolling_windows)

    print("Computing season-to-date stats...")
    df = compute_season_to_date_stats(df, rolling_stats)

    print("Computing games started count...")
    df = compute_games_started_count(df)

    print("Adding home/away indicator...")
    df = add_home_away(df)

    # Merge with opponent batting stats
    print("Merging with opponent batting stats...")
    df = df.merge(
        team_batting,
        left_on='Opp',
        right_on='Team',
        how='left',
        suffixes=('', '_opp'),
    )

    # Target variable
    df = df.rename(columns={'SO': 'K_actual'})

    # Drop rows with missing target
    df = df.dropna(subset=['K_actual'])

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Get list of feature columns for model training.

    Excludes identifiers, dates, and target variable.
    """
    exclude_cols = [
        'Name', 'Opp', 'Tm', 'Team', 'Date', 'game_date', 'date_parsed',
        'K_actual', 'mlbID', 'IDfg', 'playerid',
        '@', 'Unnamed: 6', '#days', 'Lev',
    ]

    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols
        and not c.startswith('Unnamed')
        and df[c].dtype in ['int64', 'float64', 'int32', 'float32']
    ]

    return feature_cols


if __name__ == "__main__":
    # Example usage
    from collect import collect_all_data

    data = collect_all_data(season=2024, end_date="2024-05-01")

    features_df = build_features(
        pitcher_games=data['pitcher_games'],
        team_batting=data['team_batting'],
    )

    print(f"Features shape: {features_df.shape}")
    print(f"Feature columns: {len(get_feature_columns(features_df))}")
