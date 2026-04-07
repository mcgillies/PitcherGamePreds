#!/usr/bin/env python3
"""
Daily data pipeline for pitcher predictions.

Updates:
1. Raw data (pitcher game logs, team batting)
2. Pitcher/batter profiles from Statcast
3. Rolling stats
4. Optionally retrains models

Run with: python scripts/daily_pipeline.py [--retrain]
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from pybaseball import statcast, cache

# Enable pybaseball cache
cache.enable()


def log(msg: str):
    """Print timestamped log message."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def update_raw_data(seasons: list[int] = None):
    """Update raw pitcher game logs and team batting data."""
    from src.data.collect import collect_pitcher_game_logs, collect_team_batting

    if seasons is None:
        seasons = [datetime.now().year]

    log(f"Updating raw data for seasons: {seasons}")

    # Pitcher game logs
    all_logs = []
    for season in seasons:
        start = f"{season}-03-01"
        end = f"{season}-11-01" if season < datetime.now().year else datetime.now().strftime("%Y-%m-%d")
        log(f"  Fetching pitcher logs {start} to {end}...")
        try:
            logs = collect_pitcher_game_logs(start, end)
            all_logs.append(logs)
        except Exception as e:
            log(f"  Warning: Could not fetch logs for {season}: {e}")

    if all_logs:
        pitcher_games = pd.concat(all_logs, ignore_index=True)
        output_path = PROJECT_ROOT / "data" / "raw" / "pitcher_games.csv"
        pitcher_games.to_csv(output_path, index=False)
        log(f"  Saved {len(pitcher_games)} pitcher game logs to {output_path}")

    # Team batting
    all_batting = []
    for season in seasons:
        log(f"  Fetching team batting for {season}...")
        try:
            batting = collect_team_batting(season)
            all_batting.append(batting)
        except Exception as e:
            log(f"  Warning: Could not fetch batting for {season}: {e}")

    if all_batting:
        team_batting = pd.concat(all_batting, ignore_index=True)
        output_path = PROJECT_ROOT / "data" / "raw" / "team_batting.csv"
        team_batting.to_csv(output_path, index=False)
        log(f"  Saved {len(team_batting)} team batting rows to {output_path}")


def update_statcast_profiles(start_date: str = None, end_date: str = None):
    """Update pitcher and batter profiles from Statcast."""
    log("Updating Statcast profiles...")

    if start_date is None:
        # Default to current season
        year = datetime.now().year
        start_date = f"{year}-03-01"
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    log(f"  Fetching Statcast data from {start_date} to {end_date}...")

    try:
        data = statcast(start_dt=start_date, end_dt=end_date)
        log(f"  Retrieved {len(data)} pitches")
    except Exception as e:
        log(f"  Error fetching Statcast data: {e}")
        return

    if data.empty:
        log("  No Statcast data available")
        return

    output_dir = PROJECT_ROOT / "data" / "profiles"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build pitcher profiles
    log("  Building pitcher profiles...")
    pitcher_profiles = build_pitcher_profiles(data)
    pitcher_profiles.to_csv(output_dir / "pitcher_arsenal.csv", index=False)
    log(f"  Saved {len(pitcher_profiles)} pitcher profiles")

    # Build batter profiles
    log("  Building batter profiles...")
    batter_profiles = build_batter_profiles(data)
    batter_profiles.to_csv(output_dir / "batter_profiles.csv", index=False)
    log(f"  Saved {len(batter_profiles)} batter profiles")

    # Save plate appearances
    log("  Saving plate appearances...")
    pa_data = data[data['events'].notna()].copy()
    pa_data.to_parquet(output_dir / "plate_appearances.parquet", index=False)
    log(f"  Saved {len(pa_data)} plate appearances")


def build_pitcher_profiles(data: pd.DataFrame) -> pd.DataFrame:
    """Build pitcher arsenal profiles from Statcast data."""
    # Group by pitcher and calculate stats
    pitcher_stats = data.groupby(['pitcher', 'player_name']).agg({
        'release_speed': 'mean',
        'release_spin_rate': 'mean',
        'pfx_x': 'mean',
        'pfx_z': 'mean',
        'plate_x': 'mean',
        'plate_z': 'mean',
        'pitch_type': 'count',
    }).reset_index()

    pitcher_stats.columns = [
        'pitcher_id', 'pitcher_name', 'avg_velo', 'avg_spin',
        'avg_h_break', 'avg_v_break', 'avg_plate_x', 'avg_plate_z', 'total_pitches'
    ]

    # Calculate whiff rate, etc.
    whiff_data = data.groupby('pitcher').agg({
        'description': lambda x: (x == 'swinging_strike').sum() / max((x.isin(['swinging_strike', 'foul', 'hit_into_play', 'foul_tip'])).sum(), 1),
    }).reset_index()
    whiff_data.columns = ['pitcher_id', 'whiff_pct']

    pitcher_stats = pitcher_stats.merge(whiff_data, on='pitcher_id', how='left')

    return pitcher_stats


def build_batter_profiles(data: pd.DataFrame) -> pd.DataFrame:
    """Build batter profiles from Statcast data."""
    # Group by batter and calculate stats
    batter_stats = data.groupby(['batter', 'stand']).agg({
        'pitch_type': 'count',
        'launch_speed': 'mean',
        'launch_angle': 'mean',
        'estimated_woba_using_speedangle': 'mean',
        'estimated_ba_using_speedangle': 'mean',
    }).reset_index()

    batter_stats.columns = [
        'batter_id', 'stand', 'total_pitches',
        'avg_exit_velo', 'avg_launch_angle', 'xwoba', 'xba'
    ]

    return batter_stats


def update_rolling_stats():
    """Update rolling stats for pitchers and batters."""
    log("Updating rolling stats...")

    profiles_dir = PROJECT_ROOT / "data" / "profiles"

    # Load plate appearances
    pa_path = profiles_dir / "plate_appearances.parquet"
    if not pa_path.exists():
        log("  No plate appearances data found, skipping rolling stats")
        return

    pa = pd.read_parquet(pa_path)
    log(f"  Loaded {len(pa)} plate appearances")

    # Calculate pitcher rolling stats
    log("  Calculating pitcher rolling stats...")
    pitcher_rolling = calculate_pitcher_rolling(pa)
    pitcher_rolling.to_csv(profiles_dir / "pitcher_rolling.csv", index=False)
    log(f"  Saved {len(pitcher_rolling)} pitcher rolling stats")

    # Calculate batter rolling stats
    log("  Calculating batter rolling stats...")
    batter_rolling = calculate_batter_rolling(pa)
    batter_rolling.to_csv(profiles_dir / "batter_rolling.csv", index=False)
    log(f"  Saved {len(batter_rolling)} batter rolling stats")


def calculate_pitcher_rolling(pa: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    """Calculate rolling stats for pitchers over last N plate appearances."""
    # Sort by game date
    pa = pa.sort_values('game_date')

    rolling_stats = []
    for pitcher_id, group in pa.groupby('pitcher'):
        recent = group.tail(window)

        stats = {
            'pitcher_id': pitcher_id,
            'rolling_pa': len(recent),
            'rolling_k_pct': (recent['events'] == 'strikeout').mean() if len(recent) > 0 else 0,
            'rolling_bb_pct': (recent['events'] == 'walk').mean() if len(recent) > 0 else 0,
            'rolling_hr_pct': (recent['events'] == 'home_run').mean() if len(recent) > 0 else 0,
        }
        rolling_stats.append(stats)

    return pd.DataFrame(rolling_stats)


def calculate_batter_rolling(pa: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    """Calculate rolling stats for batters over last N plate appearances."""
    pa = pa.sort_values('game_date')

    rolling_stats = []
    for batter_id, group in pa.groupby('batter'):
        recent = group.tail(window)

        stats = {
            'batter_id': batter_id,
            'rolling_pa': len(recent),
            'rolling_k_pct': (recent['events'] == 'strikeout').mean() if len(recent) > 0 else 0,
            'rolling_bb_pct': (recent['events'] == 'walk').mean() if len(recent) > 0 else 0,
            'rolling_hr_pct': (recent['events'] == 'home_run').mean() if len(recent) > 0 else 0,
        }
        rolling_stats.append(stats)

    return pd.DataFrame(rolling_stats)


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

    # Load training data
    data_path = PROJECT_ROOT / "data" / "profiles" / "plate_appearances.parquet"
    if not data_path.exists():
        log("  No training data found, skipping model training")
        return

    log("  Loading training data...")
    data = pd.read_parquet(data_path)

    # Filter to valid outcomes
    valid_outcomes = ["strikeout", "walk", "single", "double", "triple", "home_run", "field_out", "grounded_into_double_play", "force_out", "sac_fly", "fielders_choice_out"]
    data = data[data['events'].isin(valid_outcomes)].copy()

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

    log(f"  Training data: {len(data)} plate appearances")

    # Rebuild preprocessor first
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
    ensemble = BinaryModelEnsemble()
    ensemble.fit(
        X_train, y_train,
        X_val, y_val,
        time_budget=300,  # 5 min per model
    )

    # Save model
    output_dir = PROJECT_ROOT / "models" / "binary_ensemble"
    ensemble.save(output_dir)
    log(f"  Model saved to {output_dir}")
    log("  Model training complete")


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
    parser.add_argument("--seasons", nargs="+", type=int, help="Seasons to update (default: current year)")
    parser.add_argument("--skip-raw", action="store_true", help="Skip raw data update")
    parser.add_argument("--skip-profiles", action="store_true", help="Skip Statcast profiles update")
    parser.add_argument("--skip-rolling", action="store_true", help="Skip rolling stats update")
    args = parser.parse_args()

    log("=" * 60)
    log("Starting daily pipeline")
    log("=" * 60)

    seasons = args.seasons or [datetime.now().year]

    try:
        if not args.skip_raw:
            update_raw_data(seasons)

        if not args.skip_profiles:
            update_statcast_profiles()

        if not args.skip_rolling:
            update_rolling_stats()

        if args.retrain:
            retrain_models()

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
