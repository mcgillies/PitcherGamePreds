"""
Streamlit app for pitcher props betting simulation.

Run with: streamlit run src/betting/streamlit_app.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
from datetime import date, datetime

from src.betting import espn_odds, database, value
from src.betting.database import Bet, BetStatus, BetSide
from src.betting.auto_bet import place_auto_bets, get_auto_bet_summary
from src.game_predictor_binary import GamePredictorBinary

st.set_page_config(
    page_title="Pitcher Props Betting Sim",
    page_icon="⚾",
    layout="wide",
)


def get_model_version():
    """Get model version based on file modification times."""
    model_dir = Path("models/binary_ensemble")
    if not model_dir.exists():
        return "0"

    mod_times = []
    for pkl_file in model_dir.glob("*.pkl"):
        mod_times.append(pkl_file.stat().st_mtime)

    if mod_times:
        return str(max(mod_times))
    return "0"


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


def render_dashboard():
    """Render main dashboard with stats (manual bets only)."""
    st.header("Dashboard")
    st.caption("Manual bets only (see Auto Bets tab for automated tracking)")

    # Bankroll info from main stats (already manual-only in bankroll_history)
    bankroll_stats = database.get_stats()
    # Bet record from manual bets only
    manual_stats = database.get_stats_by_type(is_auto=False)

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Current Bankroll",
            f"${bankroll_stats['current_bankroll']:.2f}",
            f"{bankroll_stats['bankroll_change']:+.2f}",
        )

    with col2:
        st.metric(
            "Total P&L",
            f"${manual_stats['total_pnl']:+.2f}",
            f"{manual_stats['roi']:.1f}% ROI",
        )

    with col3:
        st.metric(
            "Win Rate",
            f"{manual_stats['win_rate']*100:.1f}%",
            f"{manual_stats['wins']}W-{manual_stats['losses']}L-{manual_stats['pushes']}P",
        )

    with col4:
        st.metric(
            "Total Bets",
            manual_stats['total_bets'],
            f"{manual_stats['pending']} pending",
        )

    # Bankroll chart
    st.subheader("Bankroll History")
    history = database.get_bankroll_history()
    if history:
        df = pd.DataFrame(history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        st.line_chart(df.set_index('timestamp')['balance'])

    # Recent bets (manual only)
    st.subheader("Recent Bets")
    recent = database.get_bets(is_auto=False, limit=10)
    if recent:
        df = pd.DataFrame([b.to_dict() for b in recent])
        df = df[['game_date', 'pitcher_name', 'prop_type', 'side', 'line', 'odds', 'stake', 'status', 'pnl']]
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No manual bets yet. Go to Today's Games to find value!")


def render_today():
    """Render today's games with predictions and odds."""
    st.header("Today's Games")
    st.caption(date.today().isoformat())

    # Load predictions
    with st.spinner("Loading predictions..."):
        model_version = get_model_version()
        predictor = get_predictor(model_version)
        predictions = predictor.predict_day(date.today().isoformat())

    # Try to load odds
    props = []
    value_bets = []

    # Odds fetching section (ESPN - free, no API key needed)
    st.subheader("Odds (DraftKings via ESPN)")

    # Count games with lineups
    games_with_lineups = [g for g in predictions if g.get('status') == 'complete']
    total_games = len([g for g in predictions if g.get('away_prediction') or g.get('home_prediction')])

    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(f"**{total_games} games** today ({len(games_with_lineups)} with full lineups)")

    with col2:
        if st.button("Fetch Odds", type="primary"):
            with st.spinner("Fetching odds from ESPN..."):
                try:
                    props = espn_odds.get_todays_pitcher_props()
                    st.success(f"Loaded {len(props)} props")
                    st.session_state['props'] = props
                    st.rerun()
                except Exception as e:
                    st.error(f"Error fetching odds: {e}")

    # Use cached props if available
    if 'props' in st.session_state:
        props = st.session_state['props']
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"Using cached odds ({len(props)} props loaded)")
        with col2:
            if st.button("Clear Cache", type="secondary"):
                del st.session_state['props']
                st.rerun()

    # Props are all from DraftKings via ESPN
    filtered_props = props

    # Find value bets
    if filtered_props:
        value_bets = value.find_value_bets(
            predictions=predictions,
            props=filtered_props,
            min_edge=0.3,
            min_ev=0.0,
        )

    # Value bets section
    if value_bets:
        st.subheader(f"Value Opportunities ({len(value_bets)})")

        for idx, vb in enumerate(value_bets):
            with st.expander(
                f"{'🟢' if vb.expected_value > 0.05 else '🟡'} {vb.pitcher_name} - {vb.prop_type.upper()} {vb.side.upper()} {vb.line} (EV: {vb.expected_value*100:+.1f}%)"
            ):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write(f"**Game:** {vb.away_team} @ {vb.home_team}")
                    st.write(f"**Odds:** {vb.odds:+d}" if vb.odds > 0 else f"**Odds:** {vb.odds}")
                    st.write(f"**Book:** {vb.bookmaker}")

                with col2:
                    st.write(f"**Model Prediction:** {vb.model_prediction:.1f}")
                    st.write(f"**Edge:** {vb.edge:+.1f}")
                    st.write(f"**Model Prob:** {vb.model_prob*100:.1f}%")

                with col3:
                    st.write(f"**Implied Prob:** {vb.implied_prob*100:.1f}%")
                    st.write(f"**Expected Value:** {vb.expected_value*100:+.1f}%")

                # Place bet form
                st.divider()
                with st.form(key=f"bet_{idx}_{vb.pitcher_name}_{vb.prop_type}_{vb.side}_{vb.line}"):
                    stake = st.number_input("Stake ($)", min_value=0.01, value=1.0, step=0.25)
                    submitted = st.form_submit_button("Place Bet")

                    if submitted:
                        bet = Bet(
                            id=None,
                            created_at=datetime.now(),
                            game_date=date.today().isoformat(),
                            pitcher_name=vb.pitcher_name,
                            prop_type=vb.prop_type,
                            line=vb.line,
                            side=BetSide(vb.side),
                            odds=vb.odds,
                            stake=stake,
                            model_prediction=vb.model_prediction,
                            model_edge=vb.edge,
                            bookmaker=vb.bookmaker,
                            status=BetStatus.PENDING,
                            actual_result=None,
                            pnl=None,
                            home_team=vb.home_team,
                            away_team=vb.away_team,
                        )
                        bet_id = database.add_bet(bet)
                        st.success(f"Bet #{bet_id} placed! ${stake:.2f} on {vb.pitcher_name} {vb.prop_type} {vb.side} {vb.line}")
                        st.rerun()

    elif filtered_props:
        st.info("No value bets found with current criteria.")

    # Manual bet entry
    st.subheader("Place Custom Bet")

    # Build pitcher options from predictions
    pitcher_options = {}
    for game in predictions:
        if game.get("away_prediction"):
            p = game["away_prediction"]
            label = f"{p['pitcher_name']} ({game['away_team']} @ {game['home_team']})"
            pitcher_options[label] = {
                "pitcher_name": p["pitcher_name"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "prediction": p,
            }
        if game.get("home_prediction"):
            p = game["home_prediction"]
            label = f"{p['pitcher_name']} ({game['away_team']} @ {game['home_team']})"
            pitcher_options[label] = {
                "pitcher_name": p["pitcher_name"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "prediction": p,
            }

    if pitcher_options:
        with st.form(key="custom_bet_form"):
            col1, col2 = st.columns(2)

            with col1:
                selected_pitcher = st.selectbox("Pitcher", list(pitcher_options.keys()))
                prop_type = st.selectbox("Prop Type", ["strikeouts", "hits_allowed", "earned_runs"])
                line = st.number_input("Line", min_value=0.0, value=5.5, step=0.5)

            with col2:
                side = st.selectbox("Side", ["over", "under"])
                odds = st.number_input("Odds (American)", value=-110, step=5)
                stake = st.number_input("Stake ($)", min_value=0.01, value=1.0, step=0.25)

            # Show model prediction for context
            if selected_pitcher:
                info = pitcher_options[selected_pitcher]
                pred = info["prediction"]
                stat_map = {"strikeouts": "K", "hits_allowed": "H", "earned_runs": "ER"}
                model_val = pred["expected_stats"].get(stat_map.get(prop_type, "K"), "?")
                st.caption(f"Model prediction: {model_val}")

            submitted = st.form_submit_button("Place Bet", type="primary")

            if submitted:
                info = pitcher_options[selected_pitcher]
                pred = info["prediction"]
                stat_map = {"strikeouts": "K", "hits_allowed": "H", "earned_runs": "ER"}
                model_pred = pred["expected_stats"].get(stat_map.get(prop_type, "K"), 0)

                # Calculate edge
                if side == "over":
                    edge = model_pred - line
                else:
                    edge = line - model_pred

                bet = Bet(
                    id=None,
                    created_at=datetime.now(),
                    game_date=date.today().isoformat(),
                    pitcher_name=info["pitcher_name"],
                    prop_type=prop_type,
                    line=line,
                    side=BetSide(side),
                    odds=odds,
                    stake=stake,
                    model_prediction=model_pred,
                    model_edge=edge,
                    bookmaker="manual",
                    status=BetStatus.PENDING,
                    actual_result=None,
                    pnl=None,
                    home_team=info["home_team"],
                    away_team=info["away_team"],
                )
                bet_id = database.add_bet(bet)
                st.success(f"Bet #{bet_id} placed! ${stake:.2f} on {info['pitcher_name']} {prop_type} {side} {line}")
                st.rerun()
    else:
        st.info("Load predictions first to place custom bets.")

    # All predictions with odds
    st.subheader("All Predictions")

    # Build props lookup by pitcher name for quick access
    props_by_pitcher = {}
    for prop in filtered_props:
        pitcher = prop.get('pitcher_name', '').lower()
        if pitcher not in props_by_pitcher:
            props_by_pitcher[pitcher] = []
        props_by_pitcher[pitcher].append(prop)

    for game in predictions:
        with st.expander(f"{game['away_team']} @ {game['home_team']} (PF: {game.get('park_factor', 1.0):.2f})"):
            col1, col2 = st.columns(2)

            for col, (side, pred) in zip([col1, col2], [('away', game['away_prediction']), ('home', game['home_prediction'])]):
                with col:
                    if pred:
                        st.write(f"**{pred['pitcher_name']}** ({side.upper()})")
                        st.write(f"Target IP: {pred.get('target_innings', '-')}")

                        stats_df = pd.DataFrame([{
                            'K': pred['expected_stats']['K'],
                            'H': pred['expected_stats']['H'],
                            'ER': pred['expected_stats']['ER'],
                            'ERA': pred['expected_stats'].get('ERA_approx', '-'),
                        }])
                        st.dataframe(stats_df, hide_index=True)

                        # Show ESPN odds if available
                        pitcher_name = pred['pitcher_name'].lower()
                        pitcher_props = props_by_pitcher.get(pitcher_name, [])
                        if not pitcher_props:
                            # Try partial match
                            for pname, plist in props_by_pitcher.items():
                                name_parts = set(pitcher_name.split())
                                prop_parts = set(pname.split())
                                if len(name_parts & prop_parts) >= 2:
                                    pitcher_props = plist
                                    break

                        if pitcher_props:
                            st.write("**DraftKings Lines:**")
                            for prop in pitcher_props:
                                prop_type = prop['prop_type']
                                line = prop['line']
                                over_odds = prop.get('over_odds')
                                under_odds = prop.get('under_odds')

                                # Format odds strings
                                over_str = f"{over_odds:+d}" if over_odds is not None else "-"
                                under_str = f"{under_odds:+d}" if under_odds is not None else "-"

                                # Get model prediction for comparison
                                stat_map = {"strikeouts": "K", "hits_allowed": "H"}
                                model_val = pred['expected_stats'].get(stat_map.get(prop_type, ''), None)

                                if model_val:
                                    edge = model_val - line
                                    edge_str = f" (Edge: {edge:+.1f})"
                                else:
                                    edge_str = ""

                                st.caption(f"{prop_type}: O/U {line} (O:{over_str}, U:{under_str}){edge_str}")
                    else:
                        st.write(f"**{side.upper()}** - No prediction")


def render_history():
    """Render bet history and settlement."""
    st.header("Bet History")

    # Filter
    status_filter = st.selectbox(
        "Filter by status",
        ["All", "Pending", "Won", "Lost", "Push"],
    )

    status = None
    if status_filter != "All":
        status = BetStatus(status_filter.lower())

    bets = database.get_bets(status=status, limit=100)

    if not bets:
        st.info("No bets found.")
        return

    # Pending bets - allow settlement
    pending = [b for b in bets if b.status == BetStatus.PENDING]
    if pending:
        st.subheader(f"Pending Bets ({len(pending)})")

        for bet in pending:
            with st.expander(f"#{bet.id} - {bet.pitcher_name} {bet.prop_type} {bet.side.value.upper()} {bet.line}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Game:** {bet.away_team} @ {bet.home_team}")
                    st.write(f"**Date:** {bet.game_date}")
                    st.write(f"**Odds:** {bet.odds:+d}" if bet.odds > 0 else f"**Odds:** {bet.odds}")
                    st.write(f"**Stake:** ${bet.stake:.2f}")

                with col2:
                    st.write(f"**Model Prediction:** {bet.model_prediction:.1f}")
                    st.write(f"**Edge:** {bet.model_edge:+.1f}")
                    st.write(f"**Book:** {bet.bookmaker}")

                # Settlement form
                st.divider()
                col_settle, col_cancel = st.columns(2)

                with col_settle:
                    with st.form(key=f"settle_{bet.id}"):
                        actual = st.number_input(
                            f"Actual {bet.prop_type} result",
                            min_value=0.0,
                            step=1.0,
                            key=f"actual_{bet.id}"
                        )
                        submitted = st.form_submit_button("Settle Bet")

                        if submitted:
                            result = database.settle_bet(bet.id, actual)
                            st.success(f"Bet settled: {result['status'].upper()} (P&L: ${result['pnl']:+.2f})")
                            st.rerun()

                with col_cancel:
                    with st.form(key=f"cancel_{bet.id}"):
                        st.write("Pitcher scratched?")
                        cancel_submitted = st.form_submit_button("Cancel & Refund")

                        if cancel_submitted:
                            refund = database.cancel_bet(bet.id)
                            st.success(f"Bet cancelled. ${refund:.2f} refunded.")
                            st.rerun()

    # All bets table
    st.subheader("All Bets")
    df = pd.DataFrame([b.to_dict() for b in bets])
    df = df[['id', 'game_date', 'pitcher_name', 'prop_type', 'side', 'line', 'odds', 'stake', 'model_prediction', 'status', 'actual_result', 'pnl']]
    st.dataframe(df, use_container_width=True)


def is_betting_hours() -> bool:
    """Check if current time is within betting hours (10am-6pm Mountain)."""
    import pytz
    mountain = pytz.timezone('America/Denver')
    now = datetime.now(mountain)
    return 10 <= now.hour < 18


def auto_place_bets_on_load():
    """Automatically try to place bets when page loads (within betting hours)."""
    # Track last auto-check time to avoid hammering API
    last_check_key = 'last_auto_bet_check'
    now = datetime.now()

    # Only check once per 30 minutes
    if last_check_key in st.session_state:
        last_check = st.session_state[last_check_key]
        minutes_since = (now - last_check).total_seconds() / 60
        if minutes_since < 30:
            return None, False  # Too soon, skip

    # Only during betting hours
    if not is_betting_hours():
        return None, False

    st.session_state[last_check_key] = now

    try:
        bets = place_auto_bets()
        return bets, True
    except Exception as e:
        return str(e), False


def render_auto_bets():
    """Render auto-bets tracking tab."""
    st.header("Auto Bets")
    st.caption("Automatic $1 bets on all matching props to track model performance")

    # Auto-refresh toggle (reloads page every 30 min during betting hours)
    auto_refresh = st.checkbox("Auto-refresh every 30 min", value=True, key="auto_refresh_toggle")

    if auto_refresh and is_betting_hours():
        # Use meta refresh tag to reload page after 30 minutes
        st.markdown(
            '<meta http-equiv="refresh" content="1800">',
            unsafe_allow_html=True
        )
        st.caption("Page will auto-refresh in 30 min to check for new props")

    # Auto-place bets on page load (during betting hours)
    auto_result, did_check = auto_place_bets_on_load()
    if did_check:
        if isinstance(auto_result, list) and auto_result:
            st.success(f"Auto-placed {len(auto_result)} new bets on page load!")
            for bet in auto_result:
                st.write(f"- {bet['pitcher']}: {bet['prop_type']} {bet['side'].upper()} {bet['line']} @ {bet['odds']:+d}")
        elif isinstance(auto_result, str):
            st.warning(f"Auto-bet check failed: {auto_result}")

    # Show betting hours status
    if is_betting_hours():
        st.info("Within betting hours (10am-6pm MT) - checking for new props on each page load")
    else:
        st.caption("Outside betting hours (10am-6pm MT) - auto-refresh paused")

    # Auto bet summary
    summary = get_auto_bet_summary()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Auto Bets",
            summary['total_bets'],
            f"{summary['pending']} pending",
        )

    with col2:
        st.metric(
            "Record",
            f"{summary['wins']}W-{summary['losses']}L-{summary['pushes']}P",
            f"{summary['win_rate']*100:.1f}% win rate",
        )

    with col3:
        st.metric(
            "Total P&L",
            f"${summary['total_pnl']:+.2f}",
            f"{summary['roi']:.1f}% ROI",
        )

    with col4:
        # Calculate units profit (each bet is $1)
        units = summary['total_pnl']
        st.metric(
            "Units",
            f"{units:+.1f}u",
            f"per {summary['total_bets'] - summary['pending']} settled",
        )

    # Place auto bets button
    st.subheader("Place Today's Auto Bets")
    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("Check & Place Bets Now", type="primary"):
            with st.spinner("Checking for props and placing bets..."):
                try:
                    bets = place_auto_bets()
                    if bets:
                        st.success(f"Placed {len(bets)} auto bets!")
                        for bet in bets:
                            st.write(f"- {bet['pitcher']}: {bet['prop_type']} {bet['side'].upper()} {bet['line']} @ {bet['odds']:+d}")
                        st.session_state['last_auto_bet_check'] = datetime.now()
                        st.rerun()
                    else:
                        st.info("No new auto bets to place (already placed or no props available)")
                except Exception as e:
                    st.error(f"Error placing auto bets: {e}")

    with col2:
        dry_run = st.checkbox("Dry run (show what would be bet)")
        if dry_run:
            with st.spinner("Simulating auto bets..."):
                try:
                    bets = place_auto_bets(dry_run=True)
                    if bets:
                        st.write(f"**Would place {len(bets)} bets:**")
                        for bet in bets:
                            st.write(f"- {bet['pitcher']}: {bet['prop_type']} {bet['side'].upper()} {bet['line']} (Model: {bet['model_prediction']:.1f}, Edge: {bet['edge']:.2f})")
                    else:
                        st.info("No matching props found")
                except Exception as e:
                    st.error(f"Error: {e}")

    # Cumulative P&L chart
    st.subheader("Cumulative P&L")

    pnl_data = database.get_cumulative_pnl(is_auto=True)
    if pnl_data:
        df = pd.DataFrame(pnl_data)
        df['game_date'] = pd.to_datetime(df['game_date'])
        st.line_chart(df.set_index('game_date')['cumulative_pnl'])
    else:
        st.info("No settled auto bets yet")

    # Compare manual vs auto
    st.subheader("Manual vs Auto Performance")

    manual_stats = database.get_stats_by_type(is_auto=False)
    auto_stats = database.get_stats_by_type(is_auto=True)

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Manual Bets**")
        st.write(f"Record: {manual_stats['wins']}W-{manual_stats['losses']}L-{manual_stats['pushes']}P")
        st.write(f"Win Rate: {manual_stats['win_rate']*100:.1f}%")
        st.write(f"P&L: ${manual_stats['total_pnl']:+.2f}")
        st.write(f"ROI: {manual_stats['roi']:.1f}%")

    with col2:
        st.write("**Auto Bets**")
        st.write(f"Record: {auto_stats['wins']}W-{auto_stats['losses']}L-{auto_stats['pushes']}P")
        st.write(f"Win Rate: {auto_stats['win_rate']*100:.1f}%")
        st.write(f"P&L: ${auto_stats['total_pnl']:+.2f}")
        st.write(f"ROI: {auto_stats['roi']:.1f}%")

    # Side-by-side P&L charts
    st.subheader("Cumulative P&L Comparison")

    manual_pnl = database.get_cumulative_pnl(is_auto=False)
    auto_pnl = database.get_cumulative_pnl(is_auto=True)

    if manual_pnl or auto_pnl:
        # Merge data for comparison chart
        chart_data = {}
        if manual_pnl:
            for row in manual_pnl:
                chart_data[row['game_date']] = {'Manual': row['cumulative_pnl']}
        if auto_pnl:
            for row in auto_pnl:
                if row['game_date'] in chart_data:
                    chart_data[row['game_date']]['Auto'] = row['cumulative_pnl']
                else:
                    chart_data[row['game_date']] = {'Auto': row['cumulative_pnl']}

        if chart_data:
            df = pd.DataFrame.from_dict(chart_data, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df = df.ffill().fillna(0)
            st.line_chart(df)

    # Recent auto bets
    st.subheader("Recent Auto Bets")
    auto_bets = database.get_bets(is_auto=True, limit=20)

    if auto_bets:
        df = pd.DataFrame([b.to_dict() for b in auto_bets])
        df = df[['game_date', 'pitcher_name', 'prop_type', 'side', 'line', 'odds', 'model_prediction', 'model_edge', 'status', 'actual_result', 'pnl']]
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No auto bets yet")


def main():
    st.title("⚾ Pitcher Props Betting Sim")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Today's Games", "Auto Bets", "Bet History"])

    with tab1:
        render_dashboard()

    with tab2:
        render_today()

    with tab3:
        render_auto_bets()

    with tab4:
        render_history()


if __name__ == "__main__":
    main()
