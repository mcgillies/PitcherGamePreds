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

from src.betting import odds, database, value
from src.betting.database import Bet, BetStatus, BetSide
from src.game_predictor_binary import GamePredictorBinary

st.set_page_config(
    page_title="Pitcher Props Betting Sim",
    page_icon="⚾",
    layout="wide",
)


@st.cache_resource
def get_predictor():
    """Load predictor (cached)."""
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
    """Render main dashboard with stats."""
    st.header("Dashboard")

    stats = database.get_stats()

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Current Bankroll",
            f"${stats['current_bankroll']:.2f}",
            f"{stats['bankroll_change']:+.2f}",
        )

    with col2:
        st.metric(
            "Total P&L",
            f"${stats['total_pnl']:+.2f}",
            f"{stats['roi']:.1f}% ROI",
        )

    with col3:
        st.metric(
            "Win Rate",
            f"{stats['win_rate']*100:.1f}%",
            f"{stats['wins']}W-{stats['losses']}L-{stats['pushes']}P",
        )

    with col4:
        st.metric(
            "Total Bets",
            stats['total_bets'],
            f"{stats['pending']} pending",
        )

    # Bankroll chart
    st.subheader("Bankroll History")
    history = database.get_bankroll_history()
    if history:
        df = pd.DataFrame(history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        st.line_chart(df.set_index('timestamp')['balance'])

    # Recent bets
    st.subheader("Recent Bets")
    recent = database.get_bets(limit=10)
    if recent:
        df = pd.DataFrame([b.to_dict() for b in recent])
        df = df[['game_date', 'pitcher_name', 'prop_type', 'side', 'line', 'odds', 'stake', 'status', 'pnl']]
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No bets yet. Go to Today's Games to find value!")


def render_today():
    """Render today's games with predictions and odds."""
    st.header("Today's Games")
    st.caption(date.today().isoformat())

    # Load predictions
    with st.spinner("Loading predictions..."):
        predictor = get_predictor()
        predictions = predictor.predict_day(date.today().isoformat())

    # Try to load odds
    props = []
    value_bets = []

    # Odds fetching section
    st.subheader("Odds")

    # Show current quota if we have it
    if 'quota' in st.session_state:
        quota = st.session_state['quota']
        st.info(f"API Requests: {quota['requests_remaining']} remaining this month")

    # Count games with lineups
    games_with_lineups = [g for g in predictions if g.get('status') == 'complete']
    total_games = len([g for g in predictions if g.get('away_prediction') or g.get('home_prediction')])

    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(f"**{total_games} games** today ({len(games_with_lineups)} with full lineups)")
        st.caption(f"Fetching odds will use ~{total_games + 1} API requests")

    with col2:
        if st.button("Fetch All Odds", type="primary"):
            with st.spinner(f"Fetching odds for {total_games} games..."):
                try:
                    props = odds.get_todays_pitcher_props()
                    quota = odds.get_remaining_requests()
                    st.success(f"Loaded {len(props)} props")
                    st.session_state['props'] = props
                    st.session_state['quota'] = quota
                    st.rerun()
                except Exception as e:
                    st.error(f"Error fetching odds: {e}")

    # Single game fetch option (uses only 2 requests)
    with st.expander("Fetch odds for single game (saves API requests)"):
        try:
            events = odds.get_events()
            today = date.today().isoformat()
            todays_events = [e for e in events if e.get('commence_time', '').startswith(today)]

            game_options = {f"{e['away_team']} @ {e['home_team']}": e['id'] for e in todays_events}

            selected_game = st.selectbox("Select game", list(game_options.keys()))

            if st.button("Fetch This Game"):
                event_id = game_options[selected_game]
                with st.spinner("Fetching..."):
                    props_data = odds.get_player_props(event_id)
                    new_props = odds.parse_pitcher_props(props_data)

                    # Add game context
                    event = next(e for e in todays_events if e['id'] == event_id)
                    for prop in new_props:
                        prop['home_team'] = event.get('home_team')
                        prop['away_team'] = event.get('away_team')
                        prop['commence_time'] = event.get('commence_time')

                    # Merge with existing props
                    existing = st.session_state.get('props', [])
                    # Remove old props for this game
                    existing = [p for p in existing if p.get('event_id') != event_id]
                    existing.extend(new_props)

                    st.session_state['props'] = existing
                    st.session_state['quota'] = odds.get_remaining_requests()
                    st.success(f"Loaded {len(new_props)} props for {selected_game}")
                    st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

    # Use cached props if available
    if 'props' in st.session_state:
        props = st.session_state['props']
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"Using cached odds ({len(props)} props loaded)")
        with col2:
            if st.button("Clear Cache", type="secondary"):
                del st.session_state['props']
                if 'quota' in st.session_state:
                    del st.session_state['quota']
                st.rerun()

    # Book filter
    if props:
        available_books = sorted(set(p['bookmaker'] for p in props))
        selected_book = st.selectbox("Select Bookmaker", available_books, index=0)

        # Filter props by selected book
        filtered_props = [p for p in props if p['bookmaker'] == selected_book]
        st.caption(f"Showing {len(filtered_props)} props from {selected_book}")
    else:
        filtered_props = []
        selected_book = None

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

    # All predictions
    st.subheader("All Predictions")

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


def main():
    st.title("⚾ Pitcher Props Betting Sim")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Dashboard", "Today's Games", "Bet History"])

    with tab1:
        render_dashboard()

    with tab2:
        render_today()

    with tab3:
        render_history()


if __name__ == "__main__":
    main()
