"""
Data collection for pitcher strikeout prediction.

Uses mlb_data package for consistent data collection with accurate team identification.
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Import from mlb_data package
from mlb_data import (
    get_pitcher_game_logs,
    get_pitcher_season_stats,
    get_team_batting,
    ALL_TEAMS,
    TEAM_INFO,
    get_team_abbrev,
    get_team_name,
    list_teams,
)
from mlb_data.utils import compute_rolling, compute_rest_days


def collect_pitcher_game_logs(
    start_date: str,
    end_date: str | None = None,
    starters_only: bool = True,
) -> pd.DataFrame:
    """
    Fetch pitcher game-by-game stats using Statcast.

    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        starters_only: If True, filter to starting pitchers only

    Returns:
        DataFrame with daily pitcher stats and accurate team abbreviations
    """
    return get_pitcher_game_logs(
        start_date=start_date,
        end_date=end_date,
        starters_only=starters_only,
    )


def collect_team_batting(season: int) -> pd.DataFrame:
    """
    Fetch team batting statistics for a given season.

    Args:
        season: The MLB season year

    Returns:
        DataFrame with team batting stats, using team abbreviations
    """
    return get_team_batting(season)


def collect_pitcher_season_stats(season: int, qual: int = 0) -> pd.DataFrame:
    """
    Fetch pitcher season statistics.

    Args:
        season: The MLB season year
        qual: Minimum IP qualifier (0 = all pitchers)

    Returns:
        DataFrame with season pitching stats
    """
    return get_pitcher_season_stats(season, qual=qual)


def collect_all_data(
    season: int,
    start_date: str | None = None,
    end_date: str | None = None,
    output_dir: str = "data/raw",
) -> dict[str, pd.DataFrame]:
    """
    Collect all data needed for training.

    Args:
        season: MLB season year
        start_date: Start date for game logs (defaults to season start)
        end_date: End date for game logs (defaults to yesterday)
        output_dir: Directory to save raw data

    Returns:
        Dictionary with 'team_batting', 'pitcher_games', 'pitcher_season' DataFrames
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Default dates
    if start_date is None:
        start_date = f"{season}-03-28"
    if end_date is None:
        end_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')

    # Collect data using mlb_data package
    team_batting = collect_team_batting(season)
    pitcher_season = collect_pitcher_season_stats(season)
    pitcher_games = collect_pitcher_game_logs(start_date, end_date)

    # Save to CSV
    team_batting.to_csv(output_path / "team_batting.csv", index=False)
    pitcher_season.to_csv(output_path / "pitcher_season.csv", index=False)
    pitcher_games.to_csv(output_path / "pitcher_games.csv", index=False)

    print(f"Saved raw data to {output_path}")

    return {
        'team_batting': team_batting,
        'pitcher_season': pitcher_season,
        'pitcher_games': pitcher_games,
    }


if __name__ == "__main__":
    # Example usage
    data = collect_all_data(season=2024)
    print(f"Team batting: {data['team_batting'].shape}")
    print(f"Pitcher season: {data['pitcher_season'].shape}")
    print(f"Pitcher games: {data['pitcher_games'].shape}")
