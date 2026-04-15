"""
First Inning Runs Prediction App.

Predicts probability of any runs scored in the first inning (YRFI/NRFI).

Run with: streamlit run src/betting/first_inning_app.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from dataclasses import dataclass

from src.data.mlb_api import get_games_with_lineups
from src.model.train_binary_models import BinaryModelEnsemble, OUTCOME_CLASSES
from src.model.markov_sim import simulate_inning, OUTCOMES
from src.data.preprocess import MatchupPreprocessor


@dataclass
class FirstInningPrediction:
    """First inning run prediction for a game."""
    game_pk: int
    game_time: str
    away_team: str
    home_team: str
    away_pitcher_name: str
    home_pitcher_name: str

    # Expected runs
    top_1st_expected_runs: float
    bottom_1st_expected_runs: float
    total_1st_expected_runs: float

    # Probabilities
    top_1st_runs_prob: float  # P(runs in top 1st)
    bottom_1st_runs_prob: float  # P(runs in bottom 1st)
    any_runs_prob: float  # P(runs in either half)
    no_runs_prob: float  # P(no runs in 1st inning)

    # Status
    status: str = "complete"
    errors: list = None


def prob_to_american_odds(prob: float) -> int:
    """Convert probability to American odds."""
    if prob <= 0:
        return 0
    if prob >= 1:
        return -10000

    if prob >= 0.5:
        # Favorite: negative odds
        return int(-100 * prob / (1 - prob))
    else:
        # Underdog: positive odds
        return int(100 * (1 - prob) / prob)


def format_american_odds(odds: int) -> str:
    """Format American odds with +/- sign."""
    if odds >= 0:
        return f"+{odds}"
    return str(odds)


class FirstInningPredictor:
    """Predicts first inning runs using binary ensemble and Markov simulation."""

    def __init__(
        self,
        ensemble_dir: str = "models/binary_ensemble",
        preprocessor_path: str = "models/matchup_preprocessor.pkl",
        pitcher_profiles_path: str = "data/profiles/pitcher_arsenal.csv",
        batter_profiles_path: str = "data/profiles/batter_profiles.csv",
        pitcher_rolling_path: str = "data/profiles/pitcher_rolling.csv",
        batter_rolling_path: str = "data/profiles/batter_rolling.csv",
        n_simulations: int = 1000,
    ):
        self.ensemble_dir = Path(ensemble_dir)
        self.ensemble = BinaryModelEnsemble.load(self.ensemble_dir)
        self.preprocessor = MatchupPreprocessor.load(preprocessor_path)
        self.n_simulations = n_simulations

        # Load profiles
        self.pitcher_profiles = pd.read_csv(pitcher_profiles_path)
        self.batter_profiles = pd.read_csv(batter_profiles_path)

        # Load rolling stats
        try:
            self.pitcher_rolling = pd.read_csv(pitcher_rolling_path)
            self.batter_rolling = pd.read_csv(batter_rolling_path)
        except FileNotFoundError:
            self.pitcher_rolling = None
            self.batter_rolling = None

        self.outcome_classes = OUTCOME_CLASSES
        self.feature_names = self.ensemble.feature_names

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
                batter_copy["stand"] = stand if pd.notna(stand) else "R"
            else:
                batter_copy["stand"] = "R"
            result.append(batter_copy)
        return result

    def build_matchup_features(
        self,
        pitcher_id: int,
        batter_id: int,
        p_throws: str,
        stand: str,
        season: int,
    ) -> pd.DataFrame | None:
        """Build feature vector for a single matchup."""
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

        for col in pitcher_data.columns:
            if col not in ["pitcher_id", "pitcher_name", "season"]:
                features[f"p_{col}"] = pitcher_data[col].values[0]

        for col in batter_data.columns:
            if col not in ["batter_id", "batter_name", "season"]:
                features[f"b_{col}"] = batter_data[col].values[0]

        # Add rolling stats
        if self.pitcher_rolling is not None:
            p_roll_mask = self.pitcher_rolling["pitcher_id"] == pitcher_id
            p_roll_data = self.pitcher_rolling[p_roll_mask]
            if not p_roll_data.empty:
                for col in p_roll_data.columns:
                    if col != "pitcher_id":
                        features[col] = p_roll_data[col].values[0]

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

        df = pd.DataFrame([features])

        for col in self.feature_names:
            if col not in df.columns:
                df[col] = np.nan

        return df[self.feature_names]

    def get_lineup_probabilities(
        self,
        pitcher_id: int,
        p_throws: str,
        lineup: list[dict],
        season: int,
    ) -> list[np.ndarray]:
        """Get outcome probabilities for each batter in lineup vs pitcher."""
        league_avg = np.array([0.143, 0.044, 0.004, 0.083, 0.031, 0.231, 0.464])

        # Build features for all batters
        all_features = []
        batter_slots = []

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
                batter_slots.append(idx)

        # Batch predict
        proba_dict = {}

        if all_features:
            X = pd.concat(all_features, ignore_index=True)
            proba = self.ensemble.predict_proba(X, save_dir=self.ensemble_dir)

            for i, slot in enumerate(batter_slots):
                proba_row = proba[i]
                markov_proba = np.zeros(7)
                for j, outcome in enumerate(OUTCOMES):
                    if outcome in self.outcome_classes:
                        oc_idx = list(self.outcome_classes).index(outcome)
                        markov_proba[j] = proba_row[oc_idx]
                proba_dict[slot] = markov_proba

        # Build lineup proba array
        lineup_proba = []
        for slot in range(9):
            if slot in proba_dict:
                lineup_proba.append(proba_dict[slot])
            else:
                lineup_proba.append(league_avg.copy())

        return lineup_proba

    def simulate_first_inning(
        self,
        lineup_proba: list[np.ndarray],
        seed: int | None = None,
    ) -> tuple[float, float]:
        """
        Simulate first inning multiple times.

        Returns:
            Tuple of (expected_runs, probability_of_any_runs)
        """
        rng = np.random.default_rng(seed)

        total_runs = 0
        innings_with_runs = 0

        for _ in range(self.n_simulations):
            sim_seed = rng.integers(0, 2**31)
            sim_rng = np.random.default_rng(sim_seed)

            state, _, _ = simulate_inning(lineup_proba, lineup_idx=0, rng=sim_rng)
            total_runs += state.runs
            if state.runs > 0:
                innings_with_runs += 1

        expected_runs = total_runs / self.n_simulations
        runs_prob = innings_with_runs / self.n_simulations

        return expected_runs, runs_prob

    def predict_game(
        self,
        game: dict,
        season: int,
    ) -> FirstInningPrediction | None:
        """Predict first inning runs for a game."""
        errors = []

        # Check we have both pitchers and lineups
        away_pitcher = game.get("away_pitcher")
        home_pitcher = game.get("home_pitcher")
        away_lineup = game.get("away_lineup")
        home_lineup = game.get("home_lineup")

        if not all([away_pitcher, home_pitcher, away_lineup, home_lineup]):
            return None

        # Top 1st: Away lineup vs Home pitcher
        home_p_throws = self._get_pitcher_hand(home_pitcher["pitcher_id"])
        away_lineup_with_stance = self._add_batter_stances(away_lineup)

        try:
            away_lineup_proba = self.get_lineup_probabilities(
                pitcher_id=home_pitcher["pitcher_id"],
                p_throws=home_p_throws,
                lineup=away_lineup_with_stance,
                season=season,
            )
            top_exp_runs, top_runs_prob = self.simulate_first_inning(
                away_lineup_proba,
                seed=home_pitcher["pitcher_id"],
            )
        except Exception as e:
            errors.append(f"Top 1st error: {e}")
            top_exp_runs, top_runs_prob = 0.5, 0.3

        # Bottom 1st: Home lineup vs Away pitcher
        away_p_throws = self._get_pitcher_hand(away_pitcher["pitcher_id"])
        home_lineup_with_stance = self._add_batter_stances(home_lineup)

        try:
            home_lineup_proba = self.get_lineup_probabilities(
                pitcher_id=away_pitcher["pitcher_id"],
                p_throws=away_p_throws,
                lineup=home_lineup_with_stance,
                season=season,
            )
            bottom_exp_runs, bottom_runs_prob = self.simulate_first_inning(
                home_lineup_proba,
                seed=away_pitcher["pitcher_id"],
            )
        except Exception as e:
            errors.append(f"Bottom 1st error: {e}")
            bottom_exp_runs, bottom_runs_prob = 0.5, 0.3

        # Calculate combined probabilities
        # P(no runs) = P(no runs top) * P(no runs bottom)
        no_runs_prob = (1 - top_runs_prob) * (1 - bottom_runs_prob)
        any_runs_prob = 1 - no_runs_prob

        return FirstInningPrediction(
            game_pk=game["game_pk"],
            game_time=game["game_time"],
            away_team=game["away_team"]["abbreviation"],
            home_team=game["home_team"]["abbreviation"],
            away_pitcher_name=away_pitcher["pitcher_name"],
            home_pitcher_name=home_pitcher["pitcher_name"],
            top_1st_expected_runs=round(top_exp_runs, 3),
            bottom_1st_expected_runs=round(bottom_exp_runs, 3),
            total_1st_expected_runs=round(top_exp_runs + bottom_exp_runs, 3),
            top_1st_runs_prob=round(top_runs_prob, 3),
            bottom_1st_runs_prob=round(bottom_runs_prob, 3),
            any_runs_prob=round(any_runs_prob, 3),
            no_runs_prob=round(no_runs_prob, 3),
            status="complete" if not errors else "partial",
            errors=errors if errors else None,
        )

    def predict_day(self, game_date: str) -> tuple[list[FirstInningPrediction], dict]:
        """Predict first inning runs for all games on a date.

        Returns:
            Tuple of (predictions, stats) where stats contains game counts by status
        """
        season = int(game_date[:4])
        games = get_games_with_lineups(game_date)

        predictions = []
        stats = {
            "total": len(games),
            "scheduled": 0,
            "in_progress": 0,
            "final": 0,
            "no_lineups": 0,
        }

        for game in games:
            status = game.get("status", "").lower()

            # Skip in-progress and completed games (1st inning already happened)
            if any(s in status for s in ["progress", "final", "over", "completed"]):
                if "progress" in status:
                    stats["in_progress"] += 1
                else:
                    stats["final"] += 1
                continue

            stats["scheduled"] += 1

            pred = self.predict_game(game, season)
            if pred is not None:
                predictions.append(pred)
            else:
                stats["no_lineups"] += 1

        return predictions, stats


def get_model_version():
    """Get model version based on file modification times."""
    model_dir = Path("models/binary_ensemble")
    if not model_dir.exists():
        return "0"

    # Get latest modification time from model files
    mod_times = []
    for pkl_file in model_dir.glob("*.pkl"):
        mod_times.append(pkl_file.stat().st_mtime)

    if mod_times:
        return str(max(mod_times))
    return "0"


@st.cache_resource
def get_predictor(_model_version: str):
    """Load predictor (cached, invalidated when models update)."""
    return FirstInningPredictor(
        ensemble_dir="models/binary_ensemble",
        preprocessor_path="models/matchup_preprocessor.pkl",
        pitcher_profiles_path="data/profiles/pitcher_arsenal.csv",
        batter_profiles_path="data/profiles/batter_profiles.csv",
        pitcher_rolling_path="data/profiles/pitcher_rolling.csv",
        batter_rolling_path="data/profiles/batter_rolling.csv",
        n_simulations=1000,
    )


def render_game_card(pred: FirstInningPrediction):
    """Render a single game's first inning prediction."""

    # Game header
    st.subheader(f"{pred.away_team} @ {pred.home_team}")
    st.caption(f"Game time: {pred.game_time}")

    # Pitchers
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Away SP:** {pred.away_pitcher_name}")
        st.metric(
            "Bottom 1st (vs home lineup)",
            f"{pred.bottom_1st_expected_runs:.2f} runs",
            f"{pred.bottom_1st_runs_prob*100:.1f}% runs",
        )
    with col2:
        st.write(f"**Home SP:** {pred.home_pitcher_name}")
        st.metric(
            "Top 1st (vs away lineup)",
            f"{pred.top_1st_expected_runs:.2f} runs",
            f"{pred.top_1st_runs_prob*100:.1f}% runs",
        )

    st.divider()

    # Main prediction
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Expected 1st Inning Runs",
            f"{pred.total_1st_expected_runs:.2f}",
        )

    with col2:
        yrfi_odds = prob_to_american_odds(pred.any_runs_prob)
        st.metric(
            "YRFI (Yes Runs First Inning)",
            f"{pred.any_runs_prob*100:.1f}%",
            format_american_odds(yrfi_odds),
        )

    with col3:
        nrfi_odds = prob_to_american_odds(pred.no_runs_prob)
        st.metric(
            "NRFI (No Runs First Inning)",
            f"{pred.no_runs_prob*100:.1f}%",
            format_american_odds(nrfi_odds),
        )

    if pred.errors:
        st.warning(f"Warnings: {', '.join(pred.errors)}")


def main():
    st.set_page_config(
        page_title="First Inning Runs Predictor",
        page_icon="1",
        layout="wide",
    )

    st.title("1st Inning Runs Predictor (YRFI/NRFI)")
    st.caption(f"Today: {date.today().isoformat()}")

    # Load predictions
    with st.spinner("Loading predictions..."):
        try:
            model_version = get_model_version()
            predictor = get_predictor(model_version)
            predictions, stats = predictor.predict_day(date.today().isoformat())
        except Exception as e:
            st.error(f"Error loading predictions: {e}")
            predictions = []
            stats = {}

    # Show game status breakdown
    if stats:
        status_parts = []
        if stats.get("in_progress", 0) > 0:
            status_parts.append(f"{stats['in_progress']} in progress")
        if stats.get("final", 0) > 0:
            status_parts.append(f"{stats['final']} final")
        if stats.get("no_lineups", 0) > 0:
            status_parts.append(f"{stats['no_lineups']} awaiting lineups")
        if status_parts:
            st.caption(f"Games today: {stats.get('total', 0)} total ({', '.join(status_parts)})")

    if not predictions:
        if stats.get("in_progress", 0) > 0 or stats.get("final", 0) > 0:
            st.info("All scheduled games have started or completed. Check back tomorrow!")
        elif stats.get("no_lineups", 0) > 0:
            st.info("Games found but lineups not yet available. Check back closer to game time.")
        else:
            st.info("No games scheduled for today.")
        return

    st.write(f"**{len(predictions)} games** with predictions")

    # Summary table
    st.subheader("Summary")

    summary_data = []
    for pred in predictions:
        yrfi_odds = prob_to_american_odds(pred.any_runs_prob)
        nrfi_odds = prob_to_american_odds(pred.no_runs_prob)

        summary_data.append({
            "Game": f"{pred.away_team} @ {pred.home_team}",
            "Away SP": pred.away_pitcher_name,
            "Home SP": pred.home_pitcher_name,
            "Exp Runs": pred.total_1st_expected_runs,
            "YRFI %": f"{pred.any_runs_prob*100:.1f}%",
            "YRFI Odds": format_american_odds(yrfi_odds),
            "NRFI %": f"{pred.no_runs_prob*100:.1f}%",
            "NRFI Odds": format_american_odds(nrfi_odds),
        })

    df = pd.DataFrame(summary_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Individual game tabs
    st.subheader("Game Details")

    game_labels = [f"{p.away_team} @ {p.home_team}" for p in predictions]
    tabs = st.tabs(game_labels)

    for tab, pred in zip(tabs, predictions):
        with tab:
            render_game_card(pred)


if __name__ == "__main__":
    main()
