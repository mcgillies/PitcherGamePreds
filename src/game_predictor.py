"""
Game-level prediction for starting pitchers.

Aggregates PA-level predictions to full game statlines.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from src.data.mlb_api import get_pitcher_game_logs, get_games_with_lineups
from src.model.train_flaml import MatchupModelTrainer, prepare_features
from src.data.preprocess import MatchupPreprocessor


# Linear weights for expected runs per event
# Calibrated to match MLB average ERA (~4.2)
# Raw RE24 weights produce ERA ~6.6, so scaled by 0.64
RUN_VALUES = {
    "1B": 0.29,   # Single
    "2B": 0.48,   # Double
    "3B": 0.67,   # Triple
    "HR": 1.00,   # Home run (minimum 1 run)
    "BB": 0.19,   # Walk
    "K": 0.00,    # Strikeout
    "OUT": 0.00,  # Out
}


class GamePredictor:
    """
    Predicts full game statlines for starting pitchers.

    Uses:
    - PA-level model for individual matchup predictions
    - Rolling average batters faced for starter duration
    - MLB API for lineups
    """

    def __init__(
        self,
        trainer_path: str = "models/flaml_trainer.pkl",
        preprocessor_path: str = "models/matchup_preprocessor.pkl",
        pitcher_profiles_path: str = "data/profiles/pitcher_arsenal.csv",
        batter_profiles_path: str = "data/profiles/batter_profiles.csv",
        pitcher_rolling_path: str = "data/profiles/pitcher_rolling.csv",
        batter_rolling_path: str = "data/profiles/batter_rolling.csv",
        rolling_starts: int = 10,
    ):
        """
        Initialize game predictor.

        Args:
            trainer_path: Path to trained FLAML model
            preprocessor_path: Path to fitted preprocessor
            pitcher_profiles_path: Path to pitcher profiles
            batter_profiles_path: Path to batter profiles
            pitcher_rolling_path: Path to pitcher rolling stats
            batter_rolling_path: Path to batter rolling stats
            rolling_starts: Number of recent starts for BF average
        """
        self.trainer = MatchupModelTrainer.load(trainer_path)
        self.preprocessor = MatchupPreprocessor.load(preprocessor_path)
        self.rolling_starts = rolling_starts

        # Load profiles
        self.pitcher_profiles = pd.read_csv(pitcher_profiles_path)
        self.batter_profiles = pd.read_csv(batter_profiles_path)

        # Load rolling stats (pre-computed from training data)
        try:
            self.pitcher_rolling = pd.read_csv(pitcher_rolling_path)
            self.batter_rolling = pd.read_csv(batter_rolling_path)
        except FileNotFoundError:
            print("Warning: Rolling stats files not found. Predictions will have null rolling features.")
            self.pitcher_rolling = None
            self.batter_rolling = None

        # Get outcome classes and feature names
        self.outcome_classes = list(self.preprocessor.label_encoder.classes_)
        self.feature_names = self.preprocessor.get_feature_names()

    def get_expected_batters_faced(
        self,
        pitcher_id: int,
        season: int | None = None,
        default_bf: int = 24,
        min_starts: int = 5,
    ) -> float:
        """
        Get expected batters faced using median of recent starts.

        Combines current + previous season if not enough current season data.

        Args:
            pitcher_id: MLB pitcher ID
            season: Season year
            default_bf: Default if no data (roughly 6 IP)
            min_starts: Minimum starts needed before using data

        Returns:
            Expected batters faced (median of last N starts)
        """
        try:
            # Get current season logs
            game_logs = get_pitcher_game_logs(
                pitcher_id,
                season=season,
                limit=self.rolling_starts,
            )

            bf_values = [g["batters_faced"] for g in game_logs if g.get("batters_faced")]

            # If not enough current season data, add previous season
            if len(bf_values) < min_starts and season:
                prev_logs = get_pitcher_game_logs(
                    pitcher_id,
                    season=season - 1,
                    limit=self.rolling_starts - len(bf_values),
                )
                bf_values.extend([g["batters_faced"] for g in prev_logs if g.get("batters_faced")])

            if not bf_values:
                return default_bf

            # Use median (more robust to outliers like short starts)
            return float(np.median(bf_values))

        except Exception as e:
            print(f"Warning: Could not get game logs for pitcher {pitcher_id}: {e}")
            return default_bf

    def build_matchup_features(
        self,
        pitcher_id: int,
        batter_id: int,
        p_throws: str,
        stand: str,
        season: int,
    ) -> pd.DataFrame | None:
        """
        Build feature vector for a single matchup.

        Args:
            pitcher_id: Pitcher MLB ID
            batter_id: Batter MLB ID
            p_throws: Pitcher throwing hand (L/R)
            stand: Batter stance (L/R)
            season: Season year

        Returns:
            DataFrame with features or None if missing data
        """
        # Get pitcher profile
        # Check if profiles have season column
        has_season = "season" in self.pitcher_profiles.columns

        if has_season:
            pitcher_mask = (
                (self.pitcher_profiles["pitcher_id"] == pitcher_id) &
                (self.pitcher_profiles["season"] == season)
            )
            pitcher_data = self.pitcher_profiles[pitcher_mask]

            if pitcher_data.empty:
                # Try previous season
                pitcher_mask = (
                    (self.pitcher_profiles["pitcher_id"] == pitcher_id) &
                    (self.pitcher_profiles["season"] == season - 1)
                )
                pitcher_data = self.pitcher_profiles[pitcher_mask]
        else:
            # No season column - just match on pitcher_id
            pitcher_mask = self.pitcher_profiles["pitcher_id"] == pitcher_id
            pitcher_data = self.pitcher_profiles[pitcher_mask]

        if pitcher_data.empty:
            return None

        # Get batter profile
        has_batter_season = "season" in self.batter_profiles.columns

        if has_batter_season:
            batter_mask = (
                (self.batter_profiles["batter_id"] == batter_id) &
                (self.batter_profiles["season"] == season)
            )
            batter_data = self.batter_profiles[batter_mask]

            if batter_data.empty:
                # Try previous season
                batter_mask = (
                    (self.batter_profiles["batter_id"] == batter_id) &
                    (self.batter_profiles["season"] == season - 1)
                )
                batter_data = self.batter_profiles[batter_mask]
        else:
            # No season column - just match on batter_id
            batter_mask = self.batter_profiles["batter_id"] == batter_id
            batter_data = self.batter_profiles[batter_mask]

        if batter_data.empty:
            return None

        # Build feature row
        features = {}

        # Add pitcher features with prefix
        for col in pitcher_data.columns:
            if col not in ["pitcher_id", "pitcher_name", "season"]:
                features[f"p_{col}"] = pitcher_data[col].values[0]

        # Add batter features with prefix
        for col in batter_data.columns:
            if col not in ["batter_id", "batter_name", "season"]:
                features[f"b_{col}"] = batter_data[col].values[0]

        # Add pitcher rolling stats
        if self.pitcher_rolling is not None:
            p_roll_mask = self.pitcher_rolling["pitcher_id"] == pitcher_id
            p_roll_data = self.pitcher_rolling[p_roll_mask]
            if not p_roll_data.empty:
                for col in p_roll_data.columns:
                    if col != "pitcher_id":
                        features[col] = p_roll_data[col].values[0]

        # Add batter rolling stats
        if self.batter_rolling is not None:
            b_roll_mask = self.batter_rolling["batter_id"] == batter_id
            b_roll_data = self.batter_rolling[b_roll_mask]
            if not b_roll_data.empty:
                for col in b_roll_data.columns:
                    if col != "batter_id":
                        features[col] = b_roll_data[col].values[0]

        # Add handedness features
        features["p_throws_L"] = 1 if p_throws == "L" else 0
        features["p_throws_R"] = 1 if p_throws == "R" else 0
        features["stand_L"] = 1 if stand == "L" else 0
        features["stand_R"] = 1 if stand == "R" else 0
        features["matchup_LvL"] = 1 if p_throws == "L" and stand == "L" else 0
        features["matchup_LvR"] = 1 if p_throws == "L" and stand == "R" else 0
        features["matchup_RvL"] = 1 if p_throws == "R" and stand == "L" else 0
        features["matchup_RvR"] = 1 if p_throws == "R" and stand == "R" else 0
        features["same_hand"] = 1 if p_throws == stand else 0

        # Create DataFrame with correct column order
        df = pd.DataFrame([features])

        # Ensure all feature columns exist (fill missing with NaN)
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = np.nan

        return df[self.feature_names]

    def predict_game(
        self,
        pitcher_id: int,
        pitcher_name: str,
        p_throws: str,
        lineup: list[dict],
        season: int,
        expected_bf: float | None = None,
    ) -> dict[str, Any]:
        """
        Predict full game statline for a starting pitcher.

        Args:
            pitcher_id: Pitcher MLB ID
            pitcher_name: Pitcher name
            p_throws: Pitcher throwing hand
            lineup: List of batter dicts with batter_id, batter_name, batting_order
            season: Season year
            expected_bf: Override expected batters faced (uses rolling avg if None)

        Returns:
            Prediction dict with expected stats and per-batter breakdown
        """
        if expected_bf is None:
            expected_bf = self.get_expected_batters_faced(pitcher_id, season)

        # Initialize prediction accumulators
        predictions = {cls: 0.0 for cls in self.outcome_classes}
        batter_predictions = []
        missing_batters = []

        # Cycle through lineup until expected BF reached
        bf_count = 0
        lineup_idx = 0

        while bf_count < expected_bf:
            batter = lineup[lineup_idx % len(lineup)]
            times_through = lineup_idx // len(lineup) + 1

            # Get batter stance (default to R if unknown)
            stand = batter.get("stand", "R")

            # Build features
            features = self.build_matchup_features(
                pitcher_id=pitcher_id,
                batter_id=batter["batter_id"],
                p_throws=p_throws,
                stand=stand,
                season=season,
            )

            if features is not None:
                # Predict probabilities
                proba = self.trainer.predict_proba(features)[0]

                # Weight for partial last batter
                weight = min(1.0, expected_bf - bf_count)

                # Accumulate predictions
                for i, cls in enumerate(self.outcome_classes):
                    predictions[cls] += proba[i] * weight

                batter_predictions.append({
                    "batter_id": batter["batter_id"],
                    "batter_name": batter.get("batter_name", "Unknown"),
                    "batting_order": batter.get("batting_order", lineup_idx % 9 + 1),
                    "times_through": times_through,
                    "probabilities": {cls: float(proba[i]) for i, cls in enumerate(self.outcome_classes)},
                })
            else:
                missing_batters.append(batter.get("batter_name", str(batter["batter_id"])))

            bf_count += 1
            lineup_idx += 1

        # Calculate expected runs using linear weights
        expected_runs = sum(
            predictions.get(outcome, 0) * RUN_VALUES.get(outcome, 0)
            for outcome in self.outcome_classes
        )

        # Build result
        result = {
            "pitcher_id": pitcher_id,
            "pitcher_name": pitcher_name,
            "expected_bf": round(expected_bf, 1),
            "actual_bf_modeled": bf_count,
            "missing_batters": missing_batters,
            "expected_stats": {
                "K": round(predictions.get("K", 0), 2),
                "BB": round(predictions.get("BB", 0), 2),
                "H": round(
                    predictions.get("1B", 0) +
                    predictions.get("2B", 0) +
                    predictions.get("3B", 0) +
                    predictions.get("HR", 0),
                    2
                ),
                "1B": round(predictions.get("1B", 0), 2),
                "2B": round(predictions.get("2B", 0), 2),
                "3B": round(predictions.get("3B", 0), 2),
                "HR": round(predictions.get("HR", 0), 2),
                "OUT": round(predictions.get("OUT", 0), 2),
                "ER": round(expected_runs, 2),  # Expected runs allowed
            },
            "batter_predictions": batter_predictions,
        }

        # Estimate innings (rough: 3 outs per inning)
        outs = predictions.get("OUT", 0) + predictions.get("K", 0)
        result["expected_stats"]["IP_approx"] = round(outs / 3, 1)

        # ERA proxy (expected runs per 9 innings)
        if outs > 0:
            result["expected_stats"]["ERA_approx"] = round(expected_runs / (outs / 3) * 9, 2)
        else:
            result["expected_stats"]["ERA_approx"] = None

        return result

    def predict_day(
        self,
        game_date: str,
        season: int | None = None,
    ) -> list[dict]:
        """
        Predict all starting pitcher statlines for a day.

        Args:
            game_date: Date string (YYYY-MM-DD)
            season: Season year (inferred from date if None)

        Returns:
            List of game predictions
        """
        if season is None:
            season = int(game_date[:4])

        games = get_games_with_lineups(game_date)
        predictions = []

        for game in games:
            game_pred = {
                "game_pk": game["game_pk"],
                "game_time": game["game_time"],
                "away_team": game["away_team"]["abbreviation"],
                "home_team": game["home_team"]["abbreviation"],
                "away_prediction": None,
                "home_prediction": None,
                "status": "pending",
                "errors": [],
            }

            # Away pitcher vs Home lineup
            if game["away_pitcher"] and game["home_lineup"]:
                try:
                    # Need to get pitcher's throwing hand
                    p_throws = self._get_pitcher_hand(game["away_pitcher"]["pitcher_id"])

                    # Add stance to lineup batters
                    lineup_with_stance = self._add_batter_stances(game["home_lineup"])

                    game_pred["away_prediction"] = self.predict_game(
                        pitcher_id=game["away_pitcher"]["pitcher_id"],
                        pitcher_name=game["away_pitcher"]["pitcher_name"],
                        p_throws=p_throws,
                        lineup=lineup_with_stance,
                        season=season,
                    )
                except Exception as e:
                    game_pred["errors"].append(f"Away pitcher error: {e}")

            # Home pitcher vs Away lineup
            if game["home_pitcher"] and game["away_lineup"]:
                try:
                    p_throws = self._get_pitcher_hand(game["home_pitcher"]["pitcher_id"])
                    lineup_with_stance = self._add_batter_stances(game["away_lineup"])

                    game_pred["home_prediction"] = self.predict_game(
                        pitcher_id=game["home_pitcher"]["pitcher_id"],
                        pitcher_name=game["home_pitcher"]["pitcher_name"],
                        p_throws=p_throws,
                        lineup=lineup_with_stance,
                        season=season,
                    )
                except Exception as e:
                    game_pred["errors"].append(f"Home pitcher error: {e}")

            # Update status
            if game_pred["away_prediction"] and game_pred["home_prediction"]:
                game_pred["status"] = "complete"
            elif game_pred["away_prediction"] or game_pred["home_prediction"]:
                game_pred["status"] = "partial"
            elif not game["lineups_available"]:
                game_pred["status"] = "awaiting_lineups"
            else:
                game_pred["status"] = "failed"

            predictions.append(game_pred)

        return predictions

    def _get_pitcher_hand(self, pitcher_id: int) -> str:
        """Get pitcher throwing hand from profiles."""
        mask = self.pitcher_profiles["pitcher_id"] == pitcher_id
        if mask.any():
            hand = self.pitcher_profiles.loc[mask, "p_throws"].values[0]
            if pd.notna(hand):
                return hand
        return "R"  # Default to right-handed

    def _add_batter_stances(self, lineup: list[dict]) -> list[dict]:
        """Add batting stance to lineup from profiles."""
        result = []
        for batter in lineup:
            batter_copy = batter.copy()
            mask = self.batter_profiles["batter_id"] == batter["batter_id"]
            if mask.any():
                stand = self.batter_profiles.loc[mask, "stand"].values[0]
                if pd.notna(stand):
                    batter_copy["stand"] = stand
                else:
                    batter_copy["stand"] = "R"
            else:
                batter_copy["stand"] = "R"
            result.append(batter_copy)
        return result


def format_prediction_summary(prediction: dict) -> str:
    """Format a game prediction as readable text."""
    if prediction is None:
        return "No prediction available"

    stats = prediction["expected_stats"]
    lines = [
        f"{prediction['pitcher_name']}",
        f"  Expected BF: {prediction['expected_bf']}, IP~: {stats['IP_approx']}",
        f"  K: {stats['K']:.1f}, BB: {stats['BB']:.1f}, H: {stats['H']:.1f}, HR: {stats['HR']:.1f}",
        f"  Expected Runs: {stats['ER']:.2f}",
    ]

    if stats.get("ERA_approx"):
        lines.append(f"  ERA (this start): {stats['ERA_approx']:.2f}")

    if prediction["missing_batters"]:
        lines.append(f"  Missing data for: {', '.join(prediction['missing_batters'][:3])}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    from datetime import date

    predictor = GamePredictor()
    today = date.today().strftime("%Y-%m-%d")

    print(f"Predicting games for {today}...")
    predictions = predictor.predict_day(today)

    for game in predictions:
        print(f"\n{game['away_team']} @ {game['home_team']} ({game['status']})")
        if game["away_prediction"]:
            print(format_prediction_summary(game["away_prediction"]))
        if game["home_prediction"]:
            print(format_prediction_summary(game["home_prediction"]))
