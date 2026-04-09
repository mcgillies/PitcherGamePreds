#!/usr/bin/env python3
"""
Daily data pipeline for pitcher predictions.

Updates:
1. Statcast pitch data (incremental)
2. Pitcher/batter profiles
3. Rolling stats
4. Optionally retrains models

Run with: python scripts/daily_pipeline.py [--retrain]

Memory-optimized for 8GB RAM systems.
"""

import argparse
import gc
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from pybaseball import statcast, cache

# Import mlb_data for rich profile generation
from mlb_data import get_pitcher_arsenal, get_batter_pitch_stats

# Import config for rolling windows
from src.config import PITCHER_ROLLING_WINDOWS, BATTER_ROLLING_WINDOWS

# Enable pybaseball cache
cache.enable()


def clear_mem():
    """Run garbage collection and log memory."""
    gc.collect()
    try:
        import psutil
        mem_gb = psutil.Process().memory_info().rss / 1024**3
        log(f"  [Memory: {mem_gb:.2f} GB]")
    except ImportError:
        pass


def log(msg: str):
    """Print timestamped log message."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def update_statcast_profiles(end_date: str = None):
    """
    Update pitcher and batter profiles from Statcast using incremental updates.

    Loads existing historical pitches, fetches only new data, appends it,
    and rebuilds profiles from the complete dataset.
    """
    log("Updating Statcast profiles (incremental)...")

    # Import config for DATA_START
    from src.config import DATA_START

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    raw_dir = PROJECT_ROOT / "data" / "raw"
    output_dir = PROJECT_ROOT / "data" / "profiles"
    output_dir.mkdir(parents=True, exist_ok=True)

    pitches_path = raw_dir / "pitches.parquet"

    # Load existing historical data if available
    if pitches_path.exists():
        log(f"  Loading existing pitches from {pitches_path}...")
        existing_pitches = pd.read_parquet(pitches_path)
        log(f"  Loaded {len(existing_pitches):,} existing pitches")

        # Find the last date in existing data
        existing_pitches['game_date'] = pd.to_datetime(existing_pitches['game_date'])
        last_date = existing_pitches['game_date'].max()
        # Start fetching from the day after the last date
        start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        log(f"  Last date in existing data: {last_date.strftime('%Y-%m-%d')}")
    else:
        log(f"  No existing pitches found, fetching full history from {DATA_START}")
        existing_pitches = None
        start_date = DATA_START

    # Check if we need to fetch new data
    if start_date > end_date:
        log(f"  Data is already up to date (last: {start_date}, today: {end_date})")
        if existing_pitches is not None:
            data = existing_pitches
        else:
            log("  No data available")
            return None
    else:
        # Fetch new data
        log(f"  Fetching new Statcast data from {start_date} to {end_date}...")
        try:
            new_pitches = statcast(start_dt=start_date, end_dt=end_date)
            log(f"  Retrieved {len(new_pitches):,} new pitches")
        except Exception as e:
            log(f"  Error fetching Statcast data: {e}")
            if existing_pitches is not None:
                data = existing_pitches
            else:
                return None
        else:
            # Combine existing and new data
            if existing_pitches is not None and not new_pitches.empty:
                log("  Appending new pitches to existing data...")
                data = pd.concat([existing_pitches, new_pitches], ignore_index=True)
                # Remove any duplicates (by game_pk + at_bat_number + pitch_number)
                if all(col in data.columns for col in ['game_pk', 'at_bat_number', 'pitch_number']):
                    before_dedup = len(data)
                    data = data.drop_duplicates(subset=['game_pk', 'at_bat_number', 'pitch_number'], keep='last')
                    if before_dedup != len(data):
                        log(f"  Removed {before_dedup - len(data)} duplicate pitches")
                log(f"  Combined dataset: {len(data):,} total pitches")
            elif existing_pitches is not None:
                data = existing_pitches
            else:
                data = new_pitches

            # Save updated pitches
            log(f"  Saving updated pitches to {pitches_path}...")
            data.to_parquet(pitches_path, index=False)
            log(f"  Saved {len(data):,} pitches")

    if data.empty:
        log("  No Statcast data available")
        return None

    # Convert game_date to string once (in place, no copy)
    log("  Preparing data for profile generation...")
    data['game_date'] = pd.to_datetime(data['game_date']).dt.strftime('%Y-%m-%d')
    clear_mem()

    # Build pitcher profiles using mlb_data (rich features) from FULL dataset
    log("  Building pitcher profiles (rich features) from full dataset...")
    try:
        pitcher_profiles = get_pitcher_arsenal(
            DATA_START, end_date,
            min_pitches=100,  # Higher threshold for multi-year data
            pitches_df=data
        )
        pitcher_profiles.to_csv(output_dir / "pitcher_arsenal.csv", index=False)
        log(f"  Saved {len(pitcher_profiles)} pitcher profiles with {len(pitcher_profiles.columns)} columns")
        del pitcher_profiles
        clear_mem()
    except Exception as e:
        log(f"  Error building pitcher profiles: {e}")
        import traceback
        traceback.print_exc()

    # Build batter profiles using mlb_data (rich features) from FULL dataset
    log("  Building batter profiles (rich features) from full dataset...")
    try:
        batter_profiles = get_batter_pitch_stats(
            DATA_START, end_date,
            min_pitches=100,  # Higher threshold for multi-year data
            pitches_df=data
        )
        batter_profiles.to_csv(output_dir / "batter_profiles.csv", index=False)
        log(f"  Saved {len(batter_profiles)} batter profiles with {len(batter_profiles.columns)} columns")
        del batter_profiles
        clear_mem()
    except Exception as e:
        log(f"  Error building batter profiles: {e}")
        import traceback
        traceback.print_exc()

    # Save plate appearances for rolling stats (filter without copy)
    log("  Saving plate appearances...")
    pa_mask = data['events'].notna()
    data[pa_mask].to_parquet(output_dir / "plate_appearances.parquet", index=False)
    log(f"  Saved {pa_mask.sum():,} plate appearances")
    del pa_mask
    clear_mem()

    # Don't return full data - let caller reload if needed
    # This allows memory to be freed
    return None


def compute_pitcher_rolling_latest(pitches_df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Compute latest rolling stats for pitchers using proper methodology.

    Returns one row per pitcher with their most recent rolling averages.
    """
    df = pitches_df  # No copy - work with original

    # Get unique pitcher-game combinations (starts)
    starts = df.groupby(['pitcher', 'game_date']).agg(
        pitches=('release_speed', 'count'),
        whiffs=('description', lambda x: x.isin(['swinging_strike', 'swinging_strike_blocked']).sum()),
        swings=('description', lambda x: x.isin([
            'swinging_strike', 'swinging_strike_blocked',
            'foul', 'foul_tip', 'hit_into_play'
        ]).sum()),
        called_strikes=('description', lambda x: (x == 'called_strike').sum()),
        in_zone=('zone', lambda x: x.between(1, 9).sum()),
        out_zone=('zone', lambda x: (~x.between(1, 9)).sum()),
        strikeouts=('events', lambda x: x.isin(['strikeout', 'strikeout_double_play']).sum()),
        walks=('events', lambda x: x.isin(['walk']).sum()),
        pa_count=('events', lambda x: x.notna().sum()),
        avg_velo=('release_speed', 'mean'),
        avg_exit_velo=('launch_speed', 'mean'),
        xwoba=('estimated_woba_using_speedangle', 'mean'),
        batted_balls=('launch_speed', 'count'),
        hard_hit=('launch_speed', lambda x: (x >= 95).sum()),
    ).reset_index()

    # Compute chase swings separately
    chase_data = df[~df['zone'].between(1, 9)].groupby(['pitcher', 'game_date']).agg(
        out_zone_swings=('description', lambda x: x.isin([
            'swinging_strike', 'swinging_strike_blocked',
            'foul', 'foul_tip', 'hit_into_play'
        ]).sum()),
    ).reset_index()

    starts = starts.merge(chase_data, on=['pitcher', 'game_date'], how='left')
    starts['out_zone_swings'] = starts['out_zone_swings'].fillna(0)
    starts = starts.sort_values(['pitcher', 'game_date'])

    # Compute rates
    starts['whiff_rate'] = starts['whiffs'] / starts['swings'].replace(0, np.nan)
    starts['csw_rate'] = (starts['whiffs'] + starts['called_strikes']) / starts['pitches']
    starts['k_rate'] = starts['strikeouts'] / starts['pa_count'].replace(0, np.nan)
    starts['bb_rate'] = starts['walks'] / starts['pa_count'].replace(0, np.nan)
    starts['zone_rate'] = starts['in_zone'] / starts['pitches']
    starts['chase_rate'] = starts['out_zone_swings'] / starts['out_zone'].replace(0, np.nan)
    starts['barrel_rate'] = np.nan  # Simplified - skip barrel calc
    starts['hard_hit_rate'] = starts['hard_hit'] / starts['batted_balls'].replace(0, np.nan)

    rolling_stats = [
        'whiff_rate', 'csw_rate', 'k_rate', 'bb_rate',
        'zone_rate', 'chase_rate', 'avg_velo',
        'xwoba', 'avg_exit_velo', 'barrel_rate', 'hard_hit_rate'
    ]

    # Compute rolling stats for each window
    rolling_cols = []
    for window in windows:
        for stat in rolling_stats:
            col_name = f'p_roll{window}_{stat}'
            starts[col_name] = (
                starts.groupby('pitcher')[stat]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            rolling_cols.append(col_name)

    # Get the latest row per pitcher
    latest = starts.sort_values('game_date').groupby('pitcher').tail(1)
    output_cols = ['pitcher'] + rolling_cols
    result = latest[output_cols].rename(columns={'pitcher': 'pitcher_id'})

    return result


def compute_batter_rolling_latest(pitches_df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Compute latest rolling stats for batters using proper methodology.

    Returns one row per batter with their most recent rolling averages.
    """
    df = pitches_df  # No copy - work with original

    # Get unique batter-game combinations
    games = df.groupby(['batter', 'game_date']).agg(
        pitches=('release_speed', 'count'),
        whiffs=('description', lambda x: x.isin(['swinging_strike', 'swinging_strike_blocked']).sum()),
        swings=('description', lambda x: x.isin([
            'swinging_strike', 'swinging_strike_blocked',
            'foul', 'foul_tip', 'hit_into_play'
        ]).sum()),
        in_zone=('zone', lambda x: x.between(1, 9).sum()),
        out_zone=('zone', lambda x: (~x.between(1, 9)).sum()),
        strikeouts=('events', lambda x: x.isin(['strikeout', 'strikeout_double_play']).sum()),
        walks=('events', lambda x: x.isin(['walk']).sum()),
        pa_count=('events', lambda x: x.notna().sum()),
        avg_exit_velo=('launch_speed', 'mean'),
        xwoba=('estimated_woba_using_speedangle', 'mean'),
        batted_balls=('launch_speed', 'count'),
        hard_hit=('launch_speed', lambda x: (x >= 95).sum()),
    ).reset_index()

    # Compute zone/chase swings
    zone_swing_data = df[df['zone'].between(1, 9)].groupby(['batter', 'game_date']).agg(
        zone_swings=('description', lambda x: x.isin([
            'swinging_strike', 'swinging_strike_blocked',
            'foul', 'foul_tip', 'hit_into_play'
        ]).sum()),
    ).reset_index()

    chase_data = df[~df['zone'].between(1, 9)].groupby(['batter', 'game_date']).agg(
        chase_swings=('description', lambda x: x.isin([
            'swinging_strike', 'swinging_strike_blocked',
            'foul', 'foul_tip', 'hit_into_play'
        ]).sum()),
    ).reset_index()

    games = games.merge(zone_swing_data, on=['batter', 'game_date'], how='left')
    games = games.merge(chase_data, on=['batter', 'game_date'], how='left')
    games['zone_swings'] = games['zone_swings'].fillna(0)
    games['chase_swings'] = games['chase_swings'].fillna(0)
    games = games.sort_values(['batter', 'game_date'])

    # Compute rates
    games['whiff_rate'] = games['whiffs'] / games['swings'].replace(0, np.nan)
    games['contact_rate'] = 1 - games['whiff_rate']
    games['k_rate'] = games['strikeouts'] / games['pa_count'].replace(0, np.nan)
    games['bb_rate'] = games['walks'] / games['pa_count'].replace(0, np.nan)
    games['zone_swing_rate'] = games['zone_swings'] / games['in_zone'].replace(0, np.nan)
    games['chase_rate'] = games['chase_swings'] / games['out_zone'].replace(0, np.nan)
    games['barrel_rate'] = np.nan  # Simplified
    games['hard_hit_rate'] = games['hard_hit'] / games['batted_balls'].replace(0, np.nan)

    rolling_stats = [
        'whiff_rate', 'contact_rate', 'k_rate', 'bb_rate',
        'zone_swing_rate', 'chase_rate',
        'xwoba', 'avg_exit_velo', 'barrel_rate', 'hard_hit_rate'
    ]

    # Compute rolling stats for each window
    rolling_cols = []
    for window in windows:
        for stat in rolling_stats:
            col_name = f'b_roll{window}_{stat}'
            games[col_name] = (
                games.groupby('batter')[stat]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            rolling_cols.append(col_name)

    # Get the latest row per batter
    latest = games.sort_values('game_date').groupby('batter').tail(1)
    output_cols = ['batter'] + rolling_cols
    result = latest[output_cols].rename(columns={'batter': 'batter_id'})

    return result


def update_rolling_stats(pitches_df: pd.DataFrame = None):
    """Update rolling stats for pitchers and batters using proper methodology."""
    log("Updating rolling stats...")

    profiles_dir = PROJECT_ROOT / "data" / "profiles"

    # Always load from plate_appearances.parquet for rolling stats
    # This ensures we use the filtered PA data, not full pitches
    pa_path = profiles_dir / "plate_appearances.parquet"
    if not pa_path.exists():
        log("  No plate appearances data found, skipping rolling stats")
        return

    pitches_df = pd.read_parquet(pa_path)
    log(f"  Loaded {len(pitches_df):,} plate appearances")
    clear_mem()

    # Calculate pitcher rolling stats with proper windows
    log(f"  Calculating pitcher rolling stats (windows: {PITCHER_ROLLING_WINDOWS})...")
    pitcher_rolling = compute_pitcher_rolling_latest(pitches_df, PITCHER_ROLLING_WINDOWS)
    pitcher_rolling.to_csv(profiles_dir / "pitcher_rolling.csv", index=False)
    log(f"  Saved {len(pitcher_rolling)} pitcher rolling stats with {len(pitcher_rolling.columns)} columns")
    del pitcher_rolling
    clear_mem()

    # Calculate batter rolling stats with proper windows
    log(f"  Calculating batter rolling stats (windows: {BATTER_ROLLING_WINDOWS})...")
    batter_rolling = compute_batter_rolling_latest(pitches_df, BATTER_ROLLING_WINDOWS)
    batter_rolling.to_csv(profiles_dir / "batter_rolling.csv", index=False)
    log(f"  Saved {len(batter_rolling)} batter rolling stats with {len(batter_rolling.columns)} columns")
    del batter_rolling, pitches_df
    clear_mem()




def rebuild_preprocessor(data: pd.DataFrame) -> None:
    """Rebuild the matchup preprocessor from fresh data."""
    from src.data.preprocess import MatchupPreprocessor

    log("  Rebuilding preprocessor...")

    preprocessor = MatchupPreprocessor()
    preprocessor.fit(data)

    output_path = PROJECT_ROOT / "models" / "matchup_preprocessor.pkl"
    preprocessor.save(str(output_path))
    log(f"  Preprocessor saved to {output_path}")


def retrain_models():
    """Retrain the binary ensemble models (including preprocessor)."""
    log("Retraining models...")

    from sklearn.model_selection import train_test_split
    from src.model.train_binary_models import BinaryModelEnsemble, prepare_features
    from src.data.preprocess import MatchupPreprocessor

    # Load raw pitches - only columns needed for matchup building
    pitches_path = PROJECT_ROOT / "data" / "raw" / "pitches.parquet"
    if not pitches_path.exists():
        log("  No pitches data found, skipping model training")
        return

    log("  Loading raw pitches...")
    pitches = pd.read_parquet(pitches_path)
    log(f"  Loaded {len(pitches):,} pitches")
    clear_mem()

    # Load profiles
    profiles_dir = PROJECT_ROOT / "data" / "profiles"
    log("  Loading profiles...")
    pitcher_profiles = pd.read_csv(profiles_dir / "pitcher_arsenal.csv")
    batter_profiles = pd.read_csv(profiles_dir / "batter_profiles.csv")
    log(f"  Pitcher profiles: {len(pitcher_profiles)}, Batter profiles: {len(batter_profiles)}")

    # Build enriched matchup data using preprocessor
    log("  Building matchup data (this may take a while)...")
    preprocessor = MatchupPreprocessor()
    matchups = preprocessor.build_matchup_data(
        pitches_df=pitches,
        pitcher_profiles=pitcher_profiles,
        batter_profiles=batter_profiles,
        pitcher_rolling_windows=PITCHER_ROLLING_WINDOWS,
        batter_rolling_windows=BATTER_ROLLING_WINDOWS,
    )
    log(f"  Built {len(matchups):,} matchups with {len(matchups.columns)} columns")

    # Free memory - pitches and profiles no longer needed
    del pitches, pitcher_profiles, batter_profiles
    clear_mem()

    # Filter to valid outcomes
    valid_outcomes = ["strikeout", "walk", "single", "double", "triple", "home_run", "field_out", "grounded_into_double_play", "force_out", "sac_fly", "fielders_choice_out"]
    data = matchups[matchups['events'].isin(valid_outcomes)].copy()

    # Map events to outcome classes
    event_to_outcome = {
        "strikeout": "K",
        "walk": "BB",
        "single": "1B",
        "double": "2B",
        "triple": "3B",
        "home_run": "HR",
        "field_out": "OUT",
        "grounded_into_double_play": "OUT",
        "force_out": "OUT",
        "sac_fly": "OUT",
        "fielders_choice_out": "OUT",
    }
    data["outcome"] = data["events"].map(event_to_outcome)
    data = data.dropna(subset=["outcome"])

    log(f"  Training data: {len(data):,} plate appearances")

    # Free matchups - we now have filtered data
    del matchups
    clear_mem()

    # Rebuild preprocessor on enriched data
    rebuild_preprocessor(data)

    # Prepare features with new preprocessor
    log("  Preparing features...")
    try:
        X, y, feature_names, outcome_classes = prepare_features(
            data,
            preprocessor_path=str(PROJECT_ROOT / "models" / "matchup_preprocessor.pkl"),
        )
    except Exception as e:
        log(f"  Error preparing features: {e}")
        return

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    log(f"  Train: {len(X_train)}, Val: {len(X_val)}")

    # Train ensemble
    log("  Training binary ensemble (this may take a while)...")
    output_dir = PROJECT_ROOT / "models" / "binary_ensemble"
    ensemble = BinaryModelEnsemble()
    ensemble.fit(
        X_train, y_train,
        X_val, y_val,
        verbose=1,
        save_dir=output_dir,
    )

    # Save model
    output_dir = PROJECT_ROOT / "models" / "binary_ensemble"
    ensemble.save(output_dir)
    log(f"  Model saved to {output_dir}")
    log("  Model training complete")

    # Final cleanup
    del ensemble, X_train, X_val, y_train, y_val
    clear_mem()


def restart_streamlit_app():
    """Restart the Streamlit app to pick up new models."""
    import subprocess

    log("Restarting Streamlit app...")

    # Kill existing streamlit process
    subprocess.run(["pkill", "-f", "streamlit run src/betting"], capture_output=True)

    # Start new process
    subprocess.Popen(
        ["/Applications/miniconda3/envs/mlbenv/bin/streamlit", "run", "src/betting/streamlit_app.py"],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    log("  Streamlit app restarted")


def main():
    parser = argparse.ArgumentParser(description="Daily data pipeline")
    parser.add_argument("--retrain", action="store_true", help="Retrain models after data update")
    parser.add_argument("--skip-profiles", action="store_true", help="Skip Statcast profiles update")
    parser.add_argument("--skip-rolling", action="store_true", help="Skip rolling stats update")
    args = parser.parse_args()

    log("=" * 60)
    log("Starting daily pipeline")
    log("=" * 60)

    try:
        if not args.skip_profiles:
            update_statcast_profiles()
            clear_mem()

        if not args.skip_rolling:
            update_rolling_stats()
            clear_mem()

        if args.retrain:
            retrain_models()
            clear_mem()

        # Restart app to pick up new models
        restart_streamlit_app()

        log("=" * 60)
        log("Pipeline complete!")
        log("=" * 60)

    except Exception as e:
        log(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
