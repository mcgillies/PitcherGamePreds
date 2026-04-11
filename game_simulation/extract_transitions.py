"""
Extract pitcher transition events from historical pitch data.

This module processes raw pitch-by-pitch data to identify when pitching
changes occur and captures the game state at each transition.

A "transition" is when a different pitcher enters the game. We capture:
- Game state: inning, outs, score differential
- Previous pitcher info: ID, whether they were the starter
- Next pitcher info: ID
- Team context: which team is pitching, home/away
"""

import pandas as pd
import numpy as np
from pathlib import Path


def extract_transitions(pitches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract pitcher transition events from pitch data.

    A transition occurs when the pitcher changes within a game.
    We capture the game state at the moment of transition.

    Args:
        pitches_df: DataFrame with pitch-level data including:
            - game_pk: game identifier
            - pitcher: pitcher ID
            - inning: inning number
            - outs_when_up: outs at start of PA
            - home_team, away_team: team codes
            - home_score, away_score: current score (if available)
            - at_bat_number: PA number in game
            - inning_topbot: 'Top' or 'Bot'

    Returns:
        DataFrame with one row per transition event:
            - game_pk: game identifier
            - game_date: date of game
            - inning: inning when transition occurred
            - outs: outs when transition occurred
            - score_diff: score differential from pitching team's perspective
            - prev_pitcher: pitcher being replaced
            - next_pitcher: pitcher entering
            - pitching_team: team making the change
            - is_home_pitching: whether home team is pitching
            - prev_is_starter: whether prev_pitcher was the starter
            - at_bat_number: PA number when change occurred
    """
    print("Extracting pitcher transitions...")

    df = pitches_df.copy()

    # Ensure required columns exist
    required = ['game_pk', 'pitcher', 'inning', 'at_bat_number']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Sort by game and at-bat order
    df = df.sort_values(['game_pk', 'at_bat_number', 'pitch_number']).reset_index(drop=True)

    # Get first pitch of each plate appearance (to capture game state)
    pa_first_pitch = df.groupby(['game_pk', 'at_bat_number']).first().reset_index()

    # Identify the starting pitcher for each team in each game
    # The starter is the first pitcher to appear for each team
    starters = df.groupby(['game_pk', 'inning_topbot']).agg(
        starter=('pitcher', 'first')
    ).reset_index()

    # Detect pitcher changes: when pitcher differs from previous PA
    pa_first_pitch['prev_pitcher'] = pa_first_pitch.groupby('game_pk')['pitcher'].shift(1)
    pa_first_pitch['pitcher_changed'] = (
        (pa_first_pitch['pitcher'] != pa_first_pitch['prev_pitcher']) &
        pa_first_pitch['prev_pitcher'].notna()
    )

    # Filter to only transition events
    transitions = pa_first_pitch[pa_first_pitch['pitcher_changed']].copy()

    print(f"  Found {len(transitions):,} pitcher transitions")

    if len(transitions) == 0:
        return pd.DataFrame()

    # Determine which team is pitching
    # If inning_topbot == 'Top', away team is batting, home team is pitching
    # If inning_topbot == 'Bot', home team is batting, away team is pitching
    if 'inning_topbot' in transitions.columns:
        transitions['is_home_pitching'] = transitions['inning_topbot'] == 'Top'
        transitions['pitching_team'] = np.where(
            transitions['is_home_pitching'],
            transitions['home_team'],
            transitions['away_team']
        )
    else:
        transitions['is_home_pitching'] = None
        transitions['pitching_team'] = None

    # Calculate score differential from pitching team's perspective
    if 'home_score' in transitions.columns and 'away_score' in transitions.columns:
        transitions['score_diff'] = np.where(
            transitions['is_home_pitching'],
            transitions['home_score'] - transitions['away_score'],  # Home pitching: positive = winning
            transitions['away_score'] - transitions['home_score']   # Away pitching: positive = winning
        )
    else:
        # Try to infer from other columns or set to NaN
        transitions['score_diff'] = np.nan

    # Get outs when transition occurred
    if 'outs_when_up' in transitions.columns:
        transitions['outs'] = transitions['outs_when_up']
    else:
        transitions['outs'] = 0

    # Determine if previous pitcher was the starter
    # Merge with starters to check
    transitions = transitions.merge(
        starters.rename(columns={'starter': 'game_starter'}),
        on=['game_pk', 'inning_topbot'],
        how='left'
    )
    transitions['prev_is_starter'] = transitions['prev_pitcher'] == transitions['game_starter']

    # Select and rename output columns
    output_cols = {
        'game_pk': 'game_pk',
        'game_date': 'game_date',
        'inning': 'inning',
        'outs': 'outs',
        'score_diff': 'score_diff',
        'prev_pitcher': 'prev_pitcher',
        'pitcher': 'next_pitcher',
        'pitching_team': 'pitching_team',
        'is_home_pitching': 'is_home_pitching',
        'prev_is_starter': 'prev_is_starter',
        'at_bat_number': 'at_bat_number',
    }

    # Only include columns that exist
    available_cols = {k: v for k, v in output_cols.items() if k in transitions.columns}
    result = transitions[list(available_cols.keys())].rename(columns=available_cols)

    print(f"  Transitions by inning:")
    print(result.groupby('inning').size().head(10).to_string())

    return result


def extract_pitcher_usage_stats(pitches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract usage statistics for each pitcher to help classify roles.

    Computes per-pitcher aggregates:
    - Games appeared
    - Innings pitched (estimated from outs)
    - Average inning of appearance
    - Appearances as starter vs reliever
    - Saves/Holds (if derivable)

    Args:
        pitches_df: DataFrame with pitch-level data

    Returns:
        DataFrame with one row per pitcher with usage stats
    """
    print("Extracting pitcher usage statistics...")

    df = pitches_df.copy()

    # Get unique pitcher appearances per game
    appearances = df.groupby(['game_pk', 'pitcher', 'game_date']).agg(
        first_inning=('inning', 'min'),
        last_inning=('inning', 'max'),
        total_pitches=('pitch_type', 'count'),
        total_outs=('events', lambda x: x.isin([
            'strikeout', 'field_out', 'grounded_into_double_play',
            'force_out', 'sac_fly', 'sac_bunt', 'double_play',
            'fielders_choice_out', 'strikeout_double_play'
        ]).sum()),
    ).reset_index()

    # Determine if this was a start (appeared in 1st inning as first pitcher)
    first_pitcher_per_game = df.sort_values(['game_pk', 'at_bat_number']).groupby(
        ['game_pk', 'inning_topbot']
    )['pitcher'].first().reset_index()
    first_pitcher_per_game['is_start'] = first_pitcher_per_game['pitcher']

    appearances = appearances.merge(
        first_pitcher_per_game[['game_pk', 'pitcher', 'is_start']].drop_duplicates(),
        on=['game_pk', 'pitcher'],
        how='left'
    )
    appearances['is_start'] = appearances['pitcher'] == appearances['is_start']

    # Aggregate to pitcher level
    pitcher_stats = appearances.groupby('pitcher').agg(
        games=('game_pk', 'nunique'),
        starts=('is_start', 'sum'),
        relief_apps=('is_start', lambda x: (~x).sum()),
        avg_first_inning=('first_inning', 'mean'),
        avg_last_inning=('last_inning', 'mean'),
        avg_pitches=('total_pitches', 'mean'),
        avg_outs=('total_outs', 'mean'),
        total_outs=('total_outs', 'sum'),
    ).reset_index()

    # Estimate innings pitched
    pitcher_stats['est_ip'] = pitcher_stats['total_outs'] / 3
    pitcher_stats['avg_ip_per_app'] = pitcher_stats['avg_outs'] / 3

    # Relief percentage
    pitcher_stats['relief_pct'] = (
        pitcher_stats['relief_apps'] / pitcher_stats['games']
    ).fillna(0)

    print(f"  Computed stats for {len(pitcher_stats):,} pitchers")

    return pitcher_stats


def get_game_final_scores(pitches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract final scores for each game to help identify save situations.

    Args:
        pitches_df: DataFrame with pitch-level data

    Returns:
        DataFrame with game_pk and final scores
    """
    # Get last recorded score in each game
    if 'home_score' not in pitches_df.columns:
        return pd.DataFrame()

    final_scores = pitches_df.sort_values(
        ['game_pk', 'at_bat_number']
    ).groupby('game_pk').agg(
        final_home_score=('home_score', 'last'),
        final_away_score=('away_score', 'last'),
        home_team=('home_team', 'first'),
        away_team=('away_team', 'first'),
    ).reset_index()

    final_scores['home_win'] = final_scores['final_home_score'] > final_scores['final_away_score']
    final_scores['score_diff'] = abs(
        final_scores['final_home_score'] - final_scores['final_away_score']
    )

    return final_scores


if __name__ == "__main__":
    # Test with sample data
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    print("Loading pitch data...")
    pitches = pd.read_parquet("data/raw/pitches.parquet")
    print(f"Loaded {len(pitches):,} pitches")

    # Extract transitions
    transitions = extract_transitions(pitches)
    print(f"\nTransitions shape: {transitions.shape}")
    print(transitions.head())

    # Extract pitcher stats
    pitcher_stats = extract_pitcher_usage_stats(pitches)
    print(f"\nPitcher stats shape: {pitcher_stats.shape}")
    print(pitcher_stats.head())
