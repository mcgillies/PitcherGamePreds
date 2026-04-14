"""
Simple predictions-only app for viewing model predictions.

Run with: streamlit run src/betting/predictions_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
from datetime import date

from src.game_predictor_binary import GamePredictorBinary


def get_model_version():
    """Get model version based on file modification times."""
    model_dir = Path("models/binary_ensemble")
    if not model_dir.exists():
        return "0"
    mod_times = []
    for pkl_file in model_dir.glob("*.pkl"):
        mod_times.append(pkl_file.stat().st_mtime)
    return str(max(mod_times)) if mod_times else "0"


@st.cache_resource
def get_predictor(_model_version: str):
    """Load predictor (cached, invalidated when models update)."""
    return GamePredictorBinary(
        ensemble_dir="models/binary_ensemble",
        preprocessor_path="models/matchup_preprocessor.pkl",
        pitcher_profiles_path="data/profiles/pitcher_arsenal.csv",
        batter_profiles_path="data/profiles/batter_profiles.csv",
        pitcher_rolling_path="data/profiles/pitcher_rolling.csv",
        batter_rolling_path="data/profiles/batter_rolling.csv",
        park_factors_path="data/profiles/park_factors.csv",
    )


def main():
    st.set_page_config(
        page_title="MLB Pitcher Predictions",
        page_icon="baseball",
        layout="wide",
    )

    st.title("MLB Pitcher Predictions")
    st.caption(f"Today: {date.today().isoformat()}")

    # Load predictions
    with st.spinner("Loading predictions..."):
        try:
            model_version = get_model_version()
            predictor = get_predictor(model_version)
            predictions = predictor.predict_day(date.today().isoformat())
        except Exception as e:
            st.error(f"Error loading predictions: {e}")
            return

    if not predictions:
        st.info("No games found for today.")
        return

    # Summary table
    st.subheader("Summary")

    summary_data = []
    for game in predictions:
        row = {
            "Game": f"{game['away_team']} @ {game['home_team']}",
            "Time": game.get("game_time", ""),
            "Park Factor": game.get("park_factor", 1.0),
        }

        if game.get("away_prediction"):
            p = game["away_prediction"]
            row["Away SP"] = p["pitcher_name"]
            row["Away K"] = p["expected_stats"]["K"]
            row["Away H"] = p["expected_stats"]["H"]
            row["Away ER"] = p["expected_stats"]["ER"]
        else:
            row["Away SP"] = "-"
            row["Away K"] = "-"
            row["Away H"] = "-"
            row["Away ER"] = "-"

        if game.get("home_prediction"):
            p = game["home_prediction"]
            row["Home SP"] = p["pitcher_name"]
            row["Home K"] = p["expected_stats"]["K"]
            row["Home H"] = p["expected_stats"]["H"]
            row["Home ER"] = p["expected_stats"]["ER"]
        else:
            row["Home SP"] = "-"
            row["Home K"] = "-"
            row["Home H"] = "-"
            row["Home ER"] = "-"

        summary_data.append(row)

    df = pd.DataFrame(summary_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Game details
    st.subheader("Game Details")

    for game in predictions:
        status_icon = ""
        if game["status"] == "complete":
            status_icon = ""
        elif game["status"] == "partial":
            status_icon = " (partial)"
        elif game["status"] == "awaiting_lineups":
            status_icon = " (awaiting lineups)"

        with st.expander(f"{game['away_team']} @ {game['home_team']}{status_icon}"):
            col1, col2 = st.columns(2)

            # Away pitcher
            with col1:
                pred = game.get("away_prediction")
                if pred:
                    st.markdown(f"**{pred['pitcher_name']}** (Away)")
                    st.write(f"Target IP: {pred.get('target_innings', '-')}")

                    stats = pred["expected_stats"]
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    with metrics_col1:
                        st.metric("K", f"{stats['K']:.1f}")
                    with metrics_col2:
                        st.metric("H", f"{stats['H']:.1f}")
                    with metrics_col3:
                        st.metric("ER", f"{stats['ER']:.1f}")

                    st.caption(f"BB: {stats['BB']:.1f} | HR: {stats['HR']:.1f} | ERA: {stats.get('ERA_approx', '-')}")
                else:
                    st.write("Away SP: No prediction")

            # Home pitcher
            with col2:
                pred = game.get("home_prediction")
                if pred:
                    st.markdown(f"**{pred['pitcher_name']}** (Home)")
                    st.write(f"Target IP: {pred.get('target_innings', '-')}")

                    stats = pred["expected_stats"]
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    with metrics_col1:
                        st.metric("K", f"{stats['K']:.1f}")
                    with metrics_col2:
                        st.metric("H", f"{stats['H']:.1f}")
                    with metrics_col3:
                        st.metric("ER", f"{stats['ER']:.1f}")

                    st.caption(f"BB: {stats['BB']:.1f} | HR: {stats['HR']:.1f} | ERA: {stats.get('ERA_approx', '-')}")
                else:
                    st.write("Home SP: No prediction")

            st.caption(f"Park Factor: {game.get('park_factor', 1.0):.3f}")


if __name__ == "__main__":
    main()
