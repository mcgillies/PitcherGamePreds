"""
Data collection from pybaseball.

Fetches pitcher stats (season and daily), team batting stats.
Uses statcast data for 100% accurate team identification.
"""

import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pybaseball as pb
from pybaseball import cache, statcast, playerid_lookup

# Enable pybaseball caching to avoid repeated API calls
cache.enable()

# City name to team abbreviation mapping
# For cities with one team, maps directly
# For cities with two teams, maps to (AL_team, NL_team)
CITY_TO_TEAM = {
    'Atlanta': 'ATL',
    'Arizona': 'ARI',
    'Baltimore': 'BAL',
    'Boston': 'BOS',
    'Cincinnati': 'CIN',
    'Cleveland': 'CLE',
    'Colorado': 'COL',
    'Detroit': 'DET',
    'Houston': 'HOU',
    'Kansas City': 'KCR',
    'Miami': 'MIA',
    'Milwaukee': 'MIL',
    'Minnesota': 'MIN',
    'Oakland': 'OAK',
    'Philadelphia': 'PHI',
    'Pittsburgh': 'PIT',
    'San Diego': 'SDP',
    'Seattle': 'SEA',
    'San Francisco': 'SFG',
    'St. Louis': 'STL',
    'Tampa Bay': 'TBR',
    'Texas': 'TEX',
    'Toronto': 'TOR',
    'Washington': 'WSN',
}

# Cities with two teams: (AL_team, NL_team)
SHARED_CITIES = {
    'New York': ('NYY', 'NYM'),
    'Chicago': ('CHW', 'CHC'),
    'Los Angeles': ('LAA', 'LAD'),
}

# All team abbreviations with full names for reference
TEAM_INFO = {
    'ARI': ('Arizona', 'Diamondbacks', 'NL'),
    'ATL': ('Atlanta', 'Braves', 'NL'),
    'BAL': ('Baltimore', 'Orioles', 'AL'),
    'BOS': ('Boston', 'Red Sox', 'AL'),
    'CHC': ('Chicago', 'Cubs', 'NL'),
    'CHW': ('Chicago', 'White Sox', 'AL'),
    'CIN': ('Cincinnati', 'Reds', 'NL'),
    'CLE': ('Cleveland', 'Guardians', 'AL'),
    'COL': ('Colorado', 'Rockies', 'NL'),
    'DET': ('Detroit', 'Tigers', 'AL'),
    'HOU': ('Houston', 'Astros', 'AL'),
    'KCR': ('Kansas City', 'Royals', 'AL'),
    'LAA': ('Los Angeles', 'Angels', 'AL'),
    'LAD': ('Los Angeles', 'Dodgers', 'NL'),
    'MIA': ('Miami', 'Marlins', 'NL'),
    'MIL': ('Milwaukee', 'Brewers', 'NL'),
    'MIN': ('Minnesota', 'Twins', 'AL'),
    'NYM': ('New York', 'Mets', 'NL'),
    'NYY': ('New York', 'Yankees', 'AL'),
    'OAK': ('Oakland', 'Athletics', 'AL'),
    'PHI': ('Philadelphia', 'Phillies', 'NL'),
    'PIT': ('Pittsburgh', 'Pirates', 'NL'),
    'SDP': ('San Diego', 'Padres', 'NL'),
    'SEA': ('Seattle', 'Mariners', 'AL'),
    'SFG': ('San Francisco', 'Giants', 'NL'),
    'STL': ('St. Louis', 'Cardinals', 'NL'),
    'TBR': ('Tampa Bay', 'Rays', 'AL'),
    'TEX': ('Texas', 'Rangers', 'AL'),
    'TOR': ('Toronto', 'Blue Jays', 'AL'),
    'WSN': ('Washington', 'Nationals', 'NL'),
}

ALL_TEAMS = list(TEAM_INFO.keys())


def get_team_name(abbrev: str) -> str:
    """Get full team name from abbreviation."""
    if abbrev in TEAM_INFO:
        city, name, _ = TEAM_INFO[abbrev]
        return f"{city} {name}"
    return abbrev


def list_teams() -> None:
    """Print all team abbreviations and names."""
    print("\nTeam Abbreviations:")
    print("-" * 40)
    for abbrev in sorted(ALL_TEAMS):
        city, name, league = TEAM_INFO[abbrev]
        print(f"  {abbrev}: {city} {name} ({league})")
    print()


def city_to_team_abbrev(city: str, league: str | None = None) -> str:
    """
    Convert city name to team abbreviation.

    Args:
        city: City name (e.g., "New York", "Boston")
        league: League indicator ("AL", "NL", or from Lev column like "Maj-AL")

    Returns:
        Team abbreviation (e.g., "NYY", "BOS")
        For shared cities without league info, returns "CITY_AMBIGUOUS"
    """
    # Handle already-abbreviated teams
    if city in ALL_TEAMS:
        return city

    # Direct mapping for single-team cities
    if city in CITY_TO_TEAM:
        return CITY_TO_TEAM[city]

    # Shared cities need league info
    if city in SHARED_CITIES:
        al_team, nl_team = SHARED_CITIES[city]

        if league is None:
            # Return a marker for ambiguous cases
            return f"{city}_AMBIGUOUS"

        # Parse league from Lev column format (e.g., "Maj-AL")
        league_upper = league.upper()
        if 'AL' in league_upper:
            return al_team
        elif 'NL' in league_upper:
            return nl_team
        else:
            return f"{city}_AMBIGUOUS"

    # Unknown city
    return f"UNKNOWN_{city}"


def resolve_opponent_team(
    opp_city: str,
    pitcher_league: str | None = None,
) -> tuple[str, bool]:
    """
    Resolve opponent city to team abbreviation.

    For shared cities, uses heuristic that same-league matchups are more common.
    Returns the team abbreviation and a flag indicating if it might be ambiguous.

    Args:
        opp_city: Opponent city name
        pitcher_league: Pitcher's league (from Lev column)

    Returns:
        Tuple of (team_abbrev, is_ambiguous)
    """
    # Direct mapping for single-team cities
    if opp_city in CITY_TO_TEAM:
        return CITY_TO_TEAM[opp_city], False

    # Already an abbreviation
    if opp_city in ALL_TEAMS:
        return opp_city, False

    # Shared cities - use pitcher's league as heuristic
    if opp_city in SHARED_CITIES:
        al_team, nl_team = SHARED_CITIES[opp_city]

        if pitcher_league is None:
            # No league info - default to AL team but mark ambiguous
            return al_team, True

        # Assume same league (works for most regular season games)
        league_upper = pitcher_league.upper()
        if 'AL' in league_upper:
            return al_team, True  # Still mark as potentially ambiguous (interleague)
        elif 'NL' in league_upper:
            return nl_team, True
        else:
            return al_team, True

    return f"UNKNOWN_{opp_city}", True


def collect_team_batting(season: int) -> pd.DataFrame:
    """
    Fetch team batting statistics for a given season.

    Args:
        season: The MLB season year

    Returns:
        DataFrame with team batting stats, using team abbreviations
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

    # Team column already contains abbreviations from pybaseball - keep as is
    # Validate that all teams are recognized
    unknown_teams = set(team_data['Team']) - set(ALL_TEAMS)
    if unknown_teams:
        print(f"  Warning: Unknown team abbreviations: {unknown_teams}")

    return team_data


def collect_pitcher_game_logs(
    start_date: str,
    end_date: str,
    existing_data: pd.DataFrame | None = None,
    starters_only: bool = True,
) -> pd.DataFrame:
    """
    Fetch pitcher game-by-game stats using Statcast data.

    Uses Baseball Savant statcast data which provides accurate team abbreviations
    (home_team, away_team) instead of ambiguous city names.

    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        existing_data: Optional existing data to append to
        starters_only: If True, filter to starting pitchers only

    Returns:
        DataFrame with daily pitcher stats and accurate team abbreviations
    """
    print(f"Fetching pitcher game logs via Statcast from {start_date} to {end_date}...")
    print("  (This provides 100% accurate team identification)")

    # Fetch statcast data - this has pitch-level data with proper team abbrevs
    try:
        statcast_data = statcast(start_dt=start_date, end_dt=end_date)
    except Exception as e:
        print(f"  Error fetching statcast data: {e}")
        return pd.DataFrame()

    if statcast_data.empty:
        print("  No statcast data found for date range")
        return pd.DataFrame()

    print(f"  Retrieved {len(statcast_data)} pitches, aggregating to game level...")

    # Aggregate pitch-level data to game-level stats per pitcher
    game_logs = _aggregate_statcast_to_game_logs(statcast_data, starters_only)

    print(f"  Aggregated to {len(game_logs)} pitcher game logs")

    if existing_data is not None and not existing_data.empty:
        combined = pd.concat([existing_data, game_logs], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=['pitcher_id', 'game_date'],
            keep='last'
        )
        return combined

    return game_logs


def _aggregate_statcast_to_game_logs(
    statcast_df: pd.DataFrame,
    starters_only: bool = True,
) -> pd.DataFrame:
    """
    Aggregate pitch-level statcast data to game-level pitcher stats.

    Args:
        statcast_df: Raw statcast pitch data
        starters_only: If True, filter to starting pitchers

    Returns:
        DataFrame with one row per pitcher per game
    """
    df = statcast_df.copy()

    # Determine pitcher's team (home or away)
    df['is_home_pitcher'] = df['inning_topbot'] == 'Top'  # Top inning = home team pitching
    df['team_abbrev'] = np.where(df['is_home_pitcher'], df['home_team'], df['away_team'])
    df['opp_abbrev'] = np.where(df['is_home_pitcher'], df['away_team'], df['home_team'])

    # Group by game and pitcher
    group_cols = ['game_date', 'game_pk', 'pitcher', 'player_name', 'team_abbrev', 'opp_abbrev', 'home_team', 'away_team']

    # Aggregate stats
    agg_funcs = {
        # Count pitches
        'release_speed': 'count',  # Total pitches
        # Strikeouts: count pitches where events indicate strikeout
        'events': lambda x: (x.isin(['strikeout', 'strikeout_double_play'])).sum(),
        # Walks
        'balls': lambda x: (statcast_df.loc[x.index, 'events'].isin(['walk'])).sum(),
        # Hits
        'launch_speed': lambda x: (statcast_df.loc[x.index, 'events'].isin([
            'single', 'double', 'triple', 'home_run'
        ])).sum(),
        # Swinging strikes
        'description': lambda x: (x.isin(['swinging_strike', 'swinging_strike_blocked'])).sum(),
    }

    # Filter to valid pitcher appearances
    df = df[df['pitcher'].notna()]

    game_logs = df.groupby(group_cols, as_index=False).agg({
        'release_speed': 'count',  # pitch count
        'events': lambda x: x.isin(['strikeout', 'strikeout_double_play']).sum(),
        'description': lambda x: x.isin(['swinging_strike', 'swinging_strike_blocked']).sum(),
    }).rename(columns={
        'release_speed': 'pitches',
        'events': 'SO',
        'description': 'swinging_strikes',
    })

    # Add additional aggregations separately for clarity
    # Calculate IP from outs recorded
    outs_df = df.groupby(group_cols, as_index=False).apply(
        lambda g: pd.Series({
            'outs_recorded': g['events'].notna().sum() - g['events'].isin(['walk', 'hit_by_pitch', 'catcher_interf']).sum(),
            'batters_faced': g['events'].notna().sum() + g[g['events'].isna()].groupby('at_bat_number').ngroups,
            'H': g['events'].isin(['single', 'double', 'triple', 'home_run']).sum(),
            'BB': g['events'].isin(['walk']).sum(),
            'HBP': g['events'].isin(['hit_by_pitch']).sum(),
            'HR': g['events'].isin(['home_run']).sum(),
        })
    )

    if not outs_df.empty and len(outs_df.columns) > len(group_cols):
        game_logs = game_logs.merge(
            outs_df,
            on=group_cols,
            how='left'
        )

    # Rename columns for consistency
    game_logs = game_logs.rename(columns={
        'pitcher': 'pitcher_id',
        'player_name': 'Name',
    })

    # Filter to starters if requested (pitchers with most pitches in game for their team)
    if starters_only:
        # Get the pitcher with most pitches per team per game (likely starter)
        game_logs['pitch_rank'] = game_logs.groupby(
            ['game_date', 'team_abbrev']
        )['pitches'].rank(ascending=False, method='first')

        game_logs = game_logs[game_logs['pitch_rank'] == 1].drop(columns=['pitch_rank'])

    # Add is_home indicator
    game_logs['is_home'] = (game_logs['team_abbrev'] == game_logs['home_team']).astype(int)

    return game_logs


def collect_pitcher_game_logs_legacy(
    start_date: str,
    end_date: str,
    existing_data: pd.DataFrame | None = None,
    delay_seconds: float = 5.0,
) -> pd.DataFrame:
    """
    LEGACY: Fetch pitcher game logs using pitching_stats_range.

    WARNING: This method has ambiguous team identification for cities with
    two teams (New York, Chicago, Los Angeles). Use collect_pitcher_game_logs()
    instead for accurate team identification via Statcast.

    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        existing_data: Optional existing data to append to
        delay_seconds: Delay between API calls to avoid rate limiting

    Returns:
        DataFrame with daily pitcher stats (may have ambiguous opponent mapping)
    """
    print(f"[LEGACY] Fetching pitcher game logs from {start_date} to {end_date}...")
    print("  WARNING: This method may have ambiguous team mapping for NY/Chicago/LA")

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

    # Convert city names to team abbreviations (may be ambiguous)
    new_data = _convert_teams_to_abbrev_legacy(new_data)

    if existing_data is not None and not existing_data.empty:
        combined = pd.concat([existing_data, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=['Name', 'game_date'], keep='last')
        return combined

    return new_data


def _convert_teams_to_abbrev_legacy(df: pd.DataFrame) -> pd.DataFrame:
    """
    LEGACY: Convert city names to team abbreviations.

    Uses Lev column (league) to disambiguate shared cities.
    Adds 'opp_ambiguous' flag for potentially incorrect opponent mappings.
    """
    df = df.copy()

    # Convert pitcher's team (Tm column)
    if 'Tm' in df.columns:
        df['team_abbrev'] = df.apply(
            lambda row: city_to_team_abbrev(
                row['Tm'],
                row['Lev'] if 'Lev' in df.columns else None
            ),
            axis=1
        )

    # Convert opponent team (Opp column)
    if 'Opp' in df.columns:
        def resolve_opp(row):
            opp_city = row['Opp']
            pitcher_league = row['Lev'] if 'Lev' in df.columns else None
            abbrev, is_ambiguous = resolve_opponent_team(opp_city, pitcher_league)
            return pd.Series([abbrev, is_ambiguous])

        df[['opp_abbrev', 'opp_ambiguous']] = df.apply(resolve_opp, axis=1)

    return df


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
