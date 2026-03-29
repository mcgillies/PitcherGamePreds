"""
Data collection from pybaseball.

Fetches pitcher stats (season and daily), team batting stats.
"""

import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pybaseball as pb
from pybaseball import cache

# Enable pybaseball caching to avoid repeated API calls
cache.enable()

# Team abbreviation to city name mapping
TEAM_ABBREV_TO_CITY = {
    'ATL': 'Atlanta',
    'ARI': 'Arizona',
    'BAL': 'Baltimore',
    'BOS': 'Boston',
    'CHC': 'Chicago',
    'CHW': 'Chicago',
    'CIN': 'Cincinnati',
    'CLE': 'Cleveland',
    'COL': 'Colorado',
    'DET': 'Detroit',
    'HOU': 'Houston',
    'KCR': 'Kansas City',
    'LAA': 'Los Angeles',
    'LAD': 'Los Angeles',
    'MIA': 'Miami',
    'MIL': 'Milwaukee',
    'MIN': 'Minnesota',
    'NYM': 'New York',
    'NYY': 'New York',
    'OAK': 'Oakland',
    'PHI': 'Philadelphia',
    'PIT': 'Pittsburgh',
    'SDP': 'San Diego',
    'SEA': 'Seattle',
    'SFG': 'San Francisco',
    'STL': 'St. Louis',
    'TBR': 'Tampa Bay',
    'TEX': 'Texas',
    'TOR': 'Toronto',
    'WSN': 'Washington',
}


def collect_team_batting(season: int) -> pd.DataFrame:
    """
    Fetch team batting statistics for a given season.

    Args:
        season: The MLB season year

    Returns:
        DataFrame with team batting stats, indexed by team city name
    """
    print(f"Fetching team batting stats for {season}...")
    team_data = pb.team_batting(start_season=season, end_season=season)

    # Select relevant columns for strikeout prediction
    batting_cols = [
        'Team', 'Age', 'G', 'AB', 'PA', 'H', '1B', '2B', '3B', 'HR',
        'R', 'RBI', 'BB', 'IBB', 'SO', 'HBP', 'AVG',
        'BB%', 'K%', 'BB/K', 'OBP', 'SLG', 'OPS', 'ISO', 'BABIP',
        'GB%', 'FB%', 'LD%', 'HR/FB',
        'wOBA', 'wRC+',
        'O-Swing%', 'Z-Swing%', 'Swing%',
        'O-Contact%', 'Z-Contact%', 'Contact%',
        'Zone%', 'F-Strike%', 'SwStr%',
        'Soft%', 'Med%', 'Hard%',
        'EV', 'Barrel%', 'HardHit%',
        'CStr%', 'CSW%',
    ]

    # Filter to columns that exist
    available_cols = [c for c in batting_cols if c in team_data.columns]
    team_data = team_data[available_cols].copy()

    # Convert team abbreviations to city names
    team_data['Team'] = team_data['Team'].map(TEAM_ABBREV_TO_CITY)

    return team_data


def collect_pitcher_game_logs(
    start_date: str,
    end_date: str,
    existing_data: pd.DataFrame | None = None,
    delay_seconds: float = 5.0,
) -> pd.DataFrame:
    """
    Fetch pitcher game-by-game stats for a date range.

    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        existing_data: Optional existing data to append to
        delay_seconds: Delay between API calls to avoid rate limiting

    Returns:
        DataFrame with daily pitcher stats
    """
    print(f"Fetching pitcher game logs from {start_date} to {end_date}...")

    dates = pd.date_range(start_date, end_date, freq='d')
    dates = [d.strftime('%Y-%m-%d') for d in dates]

    dfs = []
    for i, date_str in enumerate(dates):
        print(f"  Fetching {date_str} ({i+1}/{len(dates)})")
        try:
            daily_data = pb.pitching_stats_range(date_str)
            if not daily_data.empty:
                daily_data['game_date'] = date_str
                dfs.append(daily_data)
        except Exception as e:
            print(f"    Warning: Failed to fetch {date_str}: {e}")

        if delay_seconds > 0 and i < len(dates) - 1:
            time.sleep(delay_seconds)

    if not dfs:
        return pd.DataFrame()

    new_data = pd.concat(dfs, ignore_index=True)

    if existing_data is not None and not existing_data.empty:
        combined = pd.concat([existing_data, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=['Name', 'game_date'], keep='last')
        return combined

    return new_data


def collect_pitcher_season_stats(season: int, qual: int = 0) -> pd.DataFrame:
    """
    Fetch pitcher season statistics.

    Args:
        season: The MLB season year
        qual: Minimum IP qualifier (0 = all pitchers)

    Returns:
        DataFrame with season pitching stats
    """
    print(f"Fetching pitcher season stats for {season}...")
    pitch_season = pb.pitching_stats(season, qual=qual)

    # Select columns relevant for strikeout prediction
    season_cols = [
        'Name', 'Team', 'W', 'L', 'ERA', 'G', 'GS', 'IP', 'TBF',
        'H', 'R', 'ER', 'HR', 'BB', 'SO',
        'K/9', 'BB/9', 'K/BB', 'H/9', 'HR/9',
        'AVG', 'WHIP', 'BABIP', 'LOB%', 'FIP', 'xFIP', 'SIERA',
        'GB%', 'FB%', 'LD%', 'HR/FB',
        'K%', 'BB%', 'K-BB%',
        'O-Swing%', 'Z-Swing%', 'Swing%',
        'O-Contact%', 'Z-Contact%', 'Contact%',
        'Zone%', 'F-Strike%', 'SwStr%',
        'Soft%', 'Med%', 'Hard%',
        # Pitch mix
        'FB%', 'FBv', 'SL%', 'SLv', 'CB%', 'CBv', 'CH%', 'CHv',
        # Pitch values
        'wFB', 'wSL', 'wCB', 'wCH',
        'wFB/C', 'wSL/C', 'wCB/C', 'wCH/C',
        # Advanced
        'CStr%', 'CSW%', 'xERA',
        'EV', 'Barrel%', 'HardHit%',
    ]

    # Filter to columns that exist (avoid duplicates)
    available_cols = []
    seen = set()
    for c in season_cols:
        if c in pitch_season.columns and c not in seen:
            available_cols.append(c)
            seen.add(c)

    return pitch_season[available_cols].copy()


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
        start_date = f"{season}-03-28"  # Approximate season start
    if end_date is None:
        end_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')

    # Collect data
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
