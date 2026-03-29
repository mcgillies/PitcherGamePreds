"""
Make predictions for upcoming games.

Loads trained model and preprocessor, fetches current data, and predicts strikeouts.
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

from .data.collect import (
    collect_pitcher_game_logs,
    collect_pitcher_season_stats,
    collect_team_batting,
)
from .data.features import (
    add_home_away,
    compute_games_started_count,
    compute_rest_days,
    compute_rolling_stats,
    compute_season_to_date_stats,
    get_feature_columns,
)
from .data.preprocess import DataPreprocessor


class StrikeoutPredictor:
    """Make strikeout predictions for upcoming games."""

    def __init__(
        self,
        model_path: str,
        preprocessor_path: str,
    ):
        """
        Initialize predictor with trained model and preprocessor.

        Args:
            model_path: Path to saved Keras model
            preprocessor_path: Path to saved preprocessor
        """
        print(f"Loading model from {model_path}")
        self.model = keras.models.load_model(model_path)

        print(f"Loading preprocessor from {preprocessor_path}")
        self.preprocessor = DataPreprocessor.load(preprocessor_path)

        self.team_batting: pd.DataFrame | None = None
        self.pitcher_history: pd.DataFrame | None = None
        self.season: int | None = None

    def load_current_data(self, season: int | None = None) -> None:
        """
        Load current season data for predictions.

        Args:
            season: MLB season year (defaults to current year)
        """
        if season is None:
            season = datetime.now().year

        self.season = season

        print(f"Loading data for {season} season...")

        # Get team batting stats
        self.team_batting = collect_team_batting(season)

        # Get recent pitcher game logs (last 30 days for rolling stats)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')

        self.pitcher_history = collect_pitcher_game_logs(
            start_date=start_date,
            end_date=end_date,
            delay_seconds=2.0,
        )

        print(f"Loaded {len(self.team_batting)} teams, {len(self.pitcher_history)} pitcher games")

    def _prepare_pitcher_features(
        self,
        pitcher_name: str,
        opponent: str,
        is_home: bool = True,
    ) -> pd.DataFrame:
        """
        Prepare features for a single pitcher prediction.

        Args:
            pitcher_name: Full pitcher name
            opponent: Opponent team abbreviation (e.g., "BOS", "NYY")
            is_home: Whether pitcher is at home

        Returns:
            DataFrame with single row of features
        """
        if self.pitcher_history is None or self.team_batting is None:
            raise ValueError("Must call load_current_data() first")

        # Get pitcher's recent games
        pitcher_games = self.pitcher_history[
            self.pitcher_history['Name'] == pitcher_name
        ].copy()

        if pitcher_games.empty:
            raise ValueError(f"No recent game data found for {pitcher_name}")

        # Compute rolling features from history
        rolling_stats = ['SO', 'IP', 'H', 'BB', 'ER']
        pitcher_games = compute_rolling_stats(pitcher_games, rolling_stats, windows=[3, 5])
        pitcher_games = compute_season_to_date_stats(pitcher_games, rolling_stats)
        pitcher_games = compute_rest_days(pitcher_games)
        pitcher_games = compute_games_started_count(pitcher_games)

        # Get most recent row as baseline
        latest = pitcher_games.iloc[[-1]].copy()

        # Update for new game
        latest['opp_abbrev'] = opponent
        latest['is_home'] = int(is_home)

        # Estimate rest days (days since last game)
        if 'game_date' in latest.columns:
            from .data.features import parse_game_date
            last_game = parse_game_date(latest['game_date'].iloc[0])
            rest = (datetime.now() - last_game).days
            latest['rest_days'] = min(rest, 30)

        # Merge with opponent batting stats
        opp_stats = self.team_batting[self.team_batting['Team'] == opponent]
        if opp_stats.empty:
            raise ValueError(f"No batting data found for opponent: {opponent}. Use team abbreviation (e.g., 'BOS', 'NYY')")

        # Cross join (single row each)
        latest['_merge_key'] = 1
        opp_stats = opp_stats.copy()
        opp_stats['_merge_key'] = 1

        features = latest.merge(opp_stats, on='_merge_key', suffixes=('', '_opp'))
        features = features.drop(columns=['_merge_key'])

        return features

    def predict(
        self,
        pitcher_name: str,
        opponent: str,
        is_home: bool = True,
    ) -> float:
        """
        Predict strikeouts for a single game.

        Args:
            pitcher_name: Full pitcher name (e.g., "Gerrit Cole")
            opponent: Opponent team abbreviation (e.g., "BOS", "NYY", "LAD")
            is_home: Whether pitcher is at home

        Returns:
            Predicted strikeouts
        """
        features = self._prepare_pitcher_features(pitcher_name, opponent, is_home)

        # Transform features
        X = self.preprocessor.transform(features)

        # Predict
        prediction = self.model.predict(X, verbose=0)[0, 0]

        return float(prediction)

    def predict_batch(
        self,
        matchups: list[dict],
    ) -> pd.DataFrame:
        """
        Predict strikeouts for multiple games.

        Args:
            matchups: List of dicts with 'pitcher', 'opponent', 'is_home' keys

        Returns:
            DataFrame with predictions
        """
        results = []

        for matchup in matchups:
            pitcher = matchup['pitcher']
            opponent = matchup['opponent']
            is_home = matchup.get('is_home', True)

            try:
                pred = self.predict(pitcher, opponent, is_home)
                results.append({
                    'pitcher': pitcher,
                    'opponent': opponent,
                    'is_home': is_home,
                    'predicted_k': pred,
                    'predicted_k_rounded': round(pred * 2) / 2,
                    'error': None,
                })
            except Exception as e:
                results.append({
                    'pitcher': pitcher,
                    'opponent': opponent,
                    'is_home': is_home,
                    'predicted_k': None,
                    'predicted_k_rounded': None,
                    'error': str(e),
                })

        return pd.DataFrame(results)


def predict_game(
    pitcher_name: str,
    opponent: str,
    model_path: str = "models/strikeout_model.keras",
    preprocessor_path: str = "models/preprocessor.pkl",
    is_home: bool = True,
) -> float:
    """
    Convenience function to predict strikeouts for a single game.

    Args:
        pitcher_name: Full pitcher name
        opponent: Opponent team abbreviation (e.g., "BOS", "NYY")
        model_path: Path to saved model
        preprocessor_path: Path to saved preprocessor
        is_home: Whether pitcher is at home

    Returns:
        Predicted strikeouts
    """
    predictor = StrikeoutPredictor(model_path, preprocessor_path)
    predictor.load_current_data()
    return predictor.predict(pitcher_name, opponent, is_home)


if __name__ == "__main__":
    # Example usage
    predictor = StrikeoutPredictor(
        model_path="models/strikeout_model.keras",
        preprocessor_path="models/preprocessor.pkl",
    )

    predictor.load_current_data(season=2024)

    # Single prediction (use team abbreviations)
    pred = predictor.predict(
        pitcher_name="Gerrit Cole",
        opponent="BOS",  # Boston Red Sox
        is_home=True,
    )
    print(f"Predicted strikeouts: {pred:.1f}")

    # Batch predictions
    matchups = [
        {"pitcher": "Gerrit Cole", "opponent": "BOS", "is_home": True},
        {"pitcher": "Zack Wheeler", "opponent": "ATL", "is_home": False},
    ]

    results = predictor.predict_batch(matchups)
    print(results)
