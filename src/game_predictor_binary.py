"""
Game-level prediction for starting pitchers using binary model ensemble.

Aggregates PA-level predictions from individual outcome models to full game statlines.
Uses Markov chain simulation for runs/ERA estimation.
"""

import gc
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from src.data.mlb_api import get_pitcher_game_logs, get_games_with_lineups
from src.model.train_binary_models import BinaryModelEnsemble, OUTCOME_CLASSES
from src.model.markov_sim import expected_game_stats, OUTCOMES
from src.data.preprocess import MatchupPreprocessor


class GamePredictorBinary:
    """
    Predicts full game statlines for starting pitchers using binary ensemble.

    Uses:
    - Binary model ensemble (7 separate models) for individual matchup predictions
    - Rolling average batters faced for starter duration
    - MLB API for lineups
    """

    def __init__(
        self,
        ensemble_dir: str = "models/binary_ensemble",
        preprocessor_path: str = "models/matchup_preprocessor.pkl",
        pitcher_profiles_path: str = "data/profiles/pitcher_arsenal.csv",
        batter_profiles_path: str = "data/profiles/batter_profiles.csv",
        pitcher_rolling_path: str = "data/profiles/pitcher_rolling.csv",
        batter_rolling_path: str = "data/profiles/batter_rolling.csv",
        park_factors_path: str = "data/profiles/park_factors.csv",
        rolling_starts: int = 10,
        xwoba_sensitivity: float = 0.3,
    ):
        """
        Initialize game predictor.

        Args:
            ensemble_dir: Directory containing binary ensemble models
            preprocessor_path: Path to fitted preprocessor
            pitcher_profiles_path: Path to pitcher profiles
            batter_profiles_path: Path to batter profiles
            pitcher_rolling_path: Path to pitcher rolling stats
            batter_rolling_path: Path to batter rolling stats
            park_factors_path: Path to park factors CSV
            rolling_starts: Number of recent starts for BF average
            xwoba_sensitivity: How much lineup xwOBA affects IP projection (0-1, default 0.5)
        """
        self.ensemble_dir = Path(ensemble_dir)
        self.ensemble = BinaryModelEnsemble.load(self.ensemble_dir)
        self.preprocessor = MatchupPreprocessor.load(preprocessor_path)
        self.rolling_starts = rolling_starts
        self.xwoba_sensitivity = xwoba_sensitivity

        # Load profiles
        self.pitcher_profiles = pd.read_csv(pitcher_profiles_path)
        self.batter_profiles = pd.read_csv(batter_profiles_path)

        # Load rolling stats
        try:
            self.pitcher_rolling = pd.read_csv(pitcher_rolling_path)
            self.batter_rolling = pd.read_csv(batter_rolling_path)
        except FileNotFoundError:
            print("Warning: Rolling stats files not found.")
            self.pitcher_rolling = None
            self.batter_rolling = None

        # Load park factors
        try:
            park_df = pd.read_csv(park_factors_path)
            self.park_factors = dict(zip(park_df["home_team"], park_df["park_factor"]))
        except FileNotFoundError:
            print("Warning: Park factors file not found. Using neutral factors.")
            self.park_factors = {}

        # Get outcome classes and feature names
        self.outcome_classes = OUTCOME_CLASSES
        self.feature_names = self.ensemble.feature_names

    def get_expected_batters_faced(
        self,
        pitcher_id: int,
        season: int | None = None,
        default_bf: int = 24,
        default_bf_per_ip: float = 4.3,
        min_starts: int = 5,
        starter_pitch_threshold: int = 65,
    ) -> tuple[float, float]:
        """
        Get expected batters faced and pitcher-specific BF/IP ratio.

        Uses pitch count to identify starter appearances (>= starter_pitch_threshold).
        Falls back to all appearances if no starter games found (for true relievers).

        Returns:
            Tuple of (expected_bf, bf_per_ip_ratio)
        """
        try:
            game_logs = get_pitcher_game_logs(
                pitcher_id,
                season=season,
                limit=self.rolling_starts,
            )

            # First, try to find starter appearances (high pitch count)
            starter_games = [
                g for g in game_logs
                if g.get("batters_faced") and g.get("pitches_thrown", 0) >= starter_pitch_threshold
            ]

            # Check previous season for more starter appearances if needed
            if len(starter_games) < min_starts and season:
                prev_logs = get_pitcher_game_logs(
                    pitcher_id,
                    season=season - 1,
                    limit=self.rolling_starts,
                )
                starter_games.extend([
                    g for g in prev_logs
                    if g.get("batters_faced") and g.get("pitches_thrown", 0) >= starter_pitch_threshold
                ])

            # If we found starter appearances, use those
            if starter_games:
                bf_values = [g["batters_faced"] for g in starter_games]
                expected_bf = float(np.median(bf_values))

                # Calculate pitcher-specific BF/IP ratio
                bf_per_ip = self._calculate_bf_per_ip(starter_games, default_bf_per_ip)

                return expected_bf, bf_per_ip

            # Fallback: use all appearances (for true relievers/openers)
            all_games = [g for g in game_logs if g.get("batters_faced")]

            if len(all_games) < min_starts and season:
                prev_logs = get_pitcher_game_logs(
                    pitcher_id,
                    season=season - 1,
                    limit=self.rolling_starts - len(all_games),
                )
                all_games.extend([g for g in prev_logs if g.get("batters_faced")])

            if not all_games:
                return default_bf, default_bf_per_ip

            bf_values = [g["batters_faced"] for g in all_games]
            expected_bf = float(np.median(bf_values))
            bf_per_ip = self._calculate_bf_per_ip(all_games, default_bf_per_ip)

            return expected_bf, bf_per_ip

        except Exception as e:
            print(f"Warning: Could not get game logs for pitcher {pitcher_id}: {e}")
            return default_bf, default_bf_per_ip

    def _calculate_bf_per_ip(
        self,
        games: list[dict],
        default: float = 4.3,
    ) -> float:
        """Calculate pitcher's BF/IP ratio from game logs."""
        total_bf = 0
        total_ip = 0.0

        for g in games:
            bf = g.get("batters_faced")
            ip_str = g.get("innings_pitched")

            if bf and ip_str:
                total_bf += bf
                # Convert IP string (e.g., "5.2" means 5 2/3) to float
                try:
                    ip = float(ip_str)
                    # Handle fractional innings (5.1 = 5 1/3, 5.2 = 5 2/3)
                    whole = int(ip)
                    frac = ip - whole
                    if frac > 0:
                        ip = whole + (frac * 10 / 3)
                    total_ip += ip
                except (ValueError, TypeError):
                    continue

        if total_ip > 0:
            return total_bf / total_ip

        return default

    def get_lineup_xwoba_factor(
        self,
        lineup: list[dict],
        season: int,
        league_avg_xwoba: float = 0.315,
    ) -> float:
        """
        Calculate lineup's xwOBA factor relative to league average.

        Higher xwOBA = more baserunners = more batters faced per inning.
        """
        xwoba_values = []

        for batter in lineup[:9]:
            batter_id = batter.get("batter_id")
            if not batter_id:
                continue

            # Look up batter's xwOBA from profiles
            has_season = "season" in self.batter_profiles.columns

            if has_season:
                mask = (
                    (self.batter_profiles["batter_id"] == batter_id) &
                    (self.batter_profiles["season"] == season)
                )
                batter_data = self.batter_profiles[mask]

                if batter_data.empty:
                    mask = (
                        (self.batter_profiles["batter_id"] == batter_id) &
                        (self.batter_profiles["season"] == season - 1)
                    )
                    batter_data = self.batter_profiles[mask]
            else:
                mask = self.batter_profiles["batter_id"] == batter_id
                batter_data = self.batter_profiles[mask]

            if not batter_data.empty and "xwoba" in batter_data.columns:
                xwoba = batter_data["xwoba"].values[0]
                if pd.notna(xwoba):
                    xwoba_values.append(xwoba)

        if not xwoba_values:
            return 1.0  # Neutral factor if no data

        lineup_xwoba = np.mean(xwoba_values)
        return lineup_xwoba / league_avg_xwoba

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
        """
        # Get pitcher profile
        has_season = "season" in self.pitcher_profiles.columns

        if has_season:
            pitcher_mask = (
                (self.pitcher_profiles["pitcher_id"] == pitcher_id) &
                (self.pitcher_profiles["season"] == season)
            )
            pitcher_data = self.pitcher_profiles[pitcher_mask]

            if pitcher_data.empty:
                pitcher_mask = (
                    (self.pitcher_profiles["pitcher_id"] == pitcher_id) &
                    (self.pitcher_profiles["season"] == season - 1)
                )
                pitcher_data = self.pitcher_profiles[pitcher_mask]
        else:
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
                batter_mask = (
                    (self.batter_profiles["batter_id"] == batter_id) &
                    (self.batter_profiles["season"] == season - 1)
                )
                batter_data = self.batter_profiles[batter_mask]
        else:
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

        # Ensure all feature columns exist
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
        target_innings: float | None = None,
        n_simulations: int = 100,
        seed: int | None = None,
        park_factor: float = 1.0,
    ) -> dict[str, Any]:
        """
        Predict full game statline for a starting pitcher.

        Uses Markov chain simulation for runs/ERA estimation.

        Args:
            pitcher_id: Pitcher MLB ID
            pitcher_name: Pitcher name
            p_throws: Pitcher throwing hand
            lineup: List of batter dicts with batter_id, batter_name, batting_order
            season: Season year
            expected_bf: Override expected batters faced (used to estimate target_innings)
            target_innings: Target innings to simulate (default: derived from expected_bf)
            n_simulations: Number of Markov simulations to average
            seed: Random seed for reproducibility
            park_factor: Park factor multiplier for runs (default 1.0)

        Returns:
            Prediction dict with expected stats and per-batter breakdown
        """
        # Get pitcher's baseline expected BF and their personal BF/IP ratio
        baseline_bf, bf_per_ip = self.get_expected_batters_faced(pitcher_id, season)
        # Apply efficiency adjustment: assume slightly fewer batters per IP than historical avg
        # This adds ~0.25-0.30 IP to projections (historical BF/IP includes pulled-early games)
        bf_per_ip = bf_per_ip * 0.95
        xwoba_factor = self.get_lineup_xwoba_factor(lineup, season)

        if expected_bf is None:
            expected_bf = baseline_bf  # Pitcher's workload stays constant

        # Convert expected_bf to target_innings
        # Adjust BF/IP ratio for lineup quality: tougher lineup = more BF per inning = fewer IP
        # Sensitivity dampens the effect (0 = no effect, 1 = full effect)
        if target_innings is None:
            dampened_factor = 1 + (xwoba_factor - 1) * self.xwoba_sensitivity
            adjusted_bf_per_ip = bf_per_ip * dampened_factor
            target_innings = expected_bf / adjusted_bf_per_ip

        # Get predictions for each lineup spot (first time through)
        # Build per-batter probability arrays for Markov simulation
        lineup_proba = []  # List of 9 probability arrays
        batter_predictions = []
        missing_batters = []

        # Collect features for first 9 batters (one time through)
        all_features = []
        batter_info = []

        for idx, batter in enumerate(lineup[:9]):
            stand = batter.get("stand", "R")

            features = self.build_matchup_features(
                pitcher_id=pitcher_id,
                batter_id=batter["batter_id"],
                p_throws=p_throws,
                stand=stand,
                season=season,
            )

            if features is not None:
                all_features.append(features)
                batter_info.append({
                    "batter": batter,
                    "lineup_slot": idx,
                })
            else:
                missing_batters.append(batter.get("batter_name", str(batter["batter_id"])))
                # Use league average probabilities for missing batters
                batter_info.append({
                    "batter": batter,
                    "lineup_slot": idx,
                    "use_default": True,
                })

        # Batch predict for available batters
        proba_dict = {}  # lineup_slot -> proba array

        if all_features:
            X = pd.concat(all_features, ignore_index=True)
            proba = self.ensemble.predict_proba(X, save_dir=self.ensemble_dir)

            feature_idx = 0
            for info in batter_info:
                if info.get("use_default"):
                    continue

                batter = info["batter"]
                slot = info["lineup_slot"]

                # Map ensemble outcome order to Markov OUTCOMES order
                # OUTCOMES = ["1B", "2B", "3B", "BB", "HR", "K", "OUT"]
                # self.outcome_classes order may differ
                proba_row = proba[feature_idx]
                markov_proba = np.zeros(7)
                for i, outcome in enumerate(OUTCOMES):
                    if outcome in self.outcome_classes:
                        oc_idx = list(self.outcome_classes).index(outcome)
                        markov_proba[i] = proba_row[oc_idx]

                proba_dict[slot] = markov_proba

                batter_predictions.append({
                    "batter_id": batter["batter_id"],
                    "batter_name": batter.get("batter_name", "Unknown"),
                    "batting_order": batter.get("batting_order", slot + 1),
                    "times_through": 1,
                    "probabilities": {cls: float(proba_row[j]) for j, cls in enumerate(self.outcome_classes)},
                })

                feature_idx += 1

        # Build lineup_proba array (9 slots)
        # Use league average for missing batters
        league_avg = np.array([0.143, 0.044, 0.004, 0.083, 0.031, 0.231, 0.464])  # 1B,2B,3B,BB,HR,K,OUT

        for slot in range(9):
            if slot in proba_dict:
                lineup_proba.append(proba_dict[slot])
            else:
                lineup_proba.append(league_avg.copy())

        # Run Markov simulation
        # Use pitcher_id as seed for reproducibility if no seed provided
        sim_seed = seed if seed is not None else pitcher_id
        sim_stats = expected_game_stats(
            lineup_proba=lineup_proba,
            target_innings=target_innings,
            n_simulations=n_simulations,
            seed=sim_seed,
        )

        # Apply park factor to runs
        adjusted_runs = sim_stats["runs"] * park_factor

        # Build result using simulation stats
        dampened_factor = 1 + (xwoba_factor - 1) * self.xwoba_sensitivity
        adjusted_bf_per_ip = bf_per_ip * dampened_factor
        result = {
            "pitcher_id": pitcher_id,
            "pitcher_name": pitcher_name,
            "baseline_bf": round(baseline_bf, 1),
            "bf_per_ip": round(bf_per_ip, 2),
            "xwoba_factor": round(xwoba_factor, 3),
            "adjusted_bf_per_ip": round(adjusted_bf_per_ip, 2),
            "expected_bf": round(expected_bf, 1),
            "target_innings": round(target_innings, 1),
            "actual_bf_modeled": len([b for b in batter_info if not b.get("use_default")]),
            "missing_batters": missing_batters,
            "expected_stats": {
                "K": round(sim_stats["strikeouts"], 2),
                "BB": round(sim_stats["walks"], 2),
                "H": round(sim_stats["hits"], 2),
                "1B": round(sim_stats["singles"], 2),
                "2B": round(sim_stats["doubles"], 2),
                "3B": round(sim_stats["triples"], 2),
                "HR": round(sim_stats["home_runs"], 2),
                "ER": round(adjusted_runs, 2),
                "IP_approx": round(sim_stats["ip"], 1),
            },
            "batter_predictions": batter_predictions,
            "simulation": {
                "n_simulations": n_simulations,
                "batters_faced": round(sim_stats["batters_faced"], 1),
            },
            "park_factor": round(park_factor, 3),
        }

        # ERA from simulation (with park factor)
        if sim_stats["ip"] > 0:
            adjusted_era = (adjusted_runs / sim_stats["ip"]) * 9
            result["expected_stats"]["ERA_approx"] = round(adjusted_era, 2)
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
        """
        if season is None:
            season = int(game_date[:4])

        games = get_games_with_lineups(game_date)
        predictions = []

        for game in games:
            home_team = game["home_team"]["abbreviation"]
            park_factor = self.park_factors.get(home_team, 1.0)

            game_pred = {
                "game_pk": game["game_pk"],
                "game_time": game["game_time"],
                "away_team": game["away_team"]["abbreviation"],
                "home_team": home_team,
                "park_factor": round(park_factor, 3),
                "away_prediction": None,
                "home_prediction": None,
                "status": "pending",
                "errors": [],
            }

            # Away pitcher vs Home lineup
            if game["away_pitcher"] and game["home_lineup"]:
                try:
                    p_throws = self._get_pitcher_hand(game["away_pitcher"]["pitcher_id"])
                    lineup_with_stance = self._add_batter_stances(game["home_lineup"])

                    game_pred["away_prediction"] = self.predict_game(
                        pitcher_id=game["away_pitcher"]["pitcher_id"],
                        pitcher_name=game["away_pitcher"]["pitcher_name"],
                        p_throws=p_throws,
                        lineup=lineup_with_stance,
                        season=season,
                        park_factor=park_factor,
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
                        park_factor=park_factor,
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
        return "R"

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
    target_ip = prediction.get("target_innings", stats.get("IP_approx", "?"))

    lines = [
        f"{prediction['pitcher_name']}",
        f"  Target IP: {target_ip}, Sim IP: {stats['IP_approx']}",
        f"  K: {stats['K']:.1f}, BB: {stats['BB']:.1f}, H: {stats['H']:.1f}, HR: {stats['HR']:.1f}",
        f"  Expected Runs: {stats['ER']:.2f}",
    ]

    if stats.get("ERA_approx"):
        lines.append(f"  ERA (this start): {stats['ERA_approx']:.2f}")

    if prediction["missing_batters"]:
        lines.append(f"  Missing data for: {', '.join(prediction['missing_batters'][:3])}")

    return "\n".join(lines)


if __name__ == "__main__":
    from datetime import date

    predictor = GamePredictorBinary()
    today = date.today().strftime("%Y-%m-%d")

    print(f"Predicting games for {today}...")
    predictions = predictor.predict_day(today)

    for game in predictions:
        print(f"\n{game['away_team']} @ {game['home_team']} ({game['status']})")
        if game["away_prediction"]:
            print(format_prediction_summary(game["away_prediction"]))
        if game["home_prediction"]:
            print(format_prediction_summary(game["home_prediction"]))
