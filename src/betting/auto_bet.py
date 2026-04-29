"""
Auto-betting system for tracking model performance against actual odds.

Places simulated $1 bets on all matching props every day.
"""

import sys
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.betting import espn_odds, database
from src.betting.database import Bet, BetStatus, BetSide
from src.game_predictor_binary import GamePredictorBinary


def get_predictor():
    """Load the game predictor."""
    return GamePredictorBinary(
        ensemble_dir="models/binary_ensemble",
        preprocessor_path="models/matchup_preprocessor.pkl",
        pitcher_profiles_path="data/profiles/pitcher_arsenal.csv",
        batter_profiles_path="data/profiles/batter_profiles.csv",
        pitcher_rolling_path="data/profiles/pitcher_rolling.csv",
        batter_rolling_path="data/profiles/batter_rolling.csv",
        park_factors_path="data/profiles/park_factors.csv",
    )


def match_prediction_to_prop(prediction: dict, prop: dict) -> bool:
    """Check if a prediction matches a prop (same pitcher)."""
    pred_name = prediction.get("pitcher_name", "").lower()
    prop_name = prop.get("pitcher_name", "").lower()

    # Handle name variations (e.g., "Jacob deGrom" vs "Jacob DeGrom")
    pred_parts = set(pred_name.split())
    prop_parts = set(prop_name.split())

    # Match if significant overlap in name parts
    overlap = pred_parts & prop_parts
    return len(overlap) >= 2 or pred_name == prop_name


def determine_bet_side(model_prediction: float, line: float, prop_type: str) -> tuple[BetSide, float]:
    """
    Determine which side to bet (over/under) based on model prediction.

    Returns:
        Tuple of (side, edge) where edge is how far off the line is
    """
    edge = model_prediction - line

    if edge > 0:
        return BetSide.OVER, edge
    else:
        return BetSide.UNDER, abs(edge)


def place_auto_bets(game_date: str | None = None, dry_run: bool = False) -> list[dict]:
    """
    Place auto bets for all matching predictions and props.

    Args:
        game_date: Date to place bets for (default: today)
        dry_run: If True, don't actually save bets to DB

    Returns:
        List of bet details that were placed
    """
    if game_date is None:
        game_date = date.today().isoformat()

    # Check if we already placed auto bets for this date
    if database.auto_bets_exist_for_date(game_date):
        print(f"Auto bets already exist for {game_date}")
        return []

    # Get predictions
    predictor = get_predictor()
    predictions = predictor.predict_day(game_date)

    # Flatten predictions to list of pitcher predictions
    pitcher_preds = []
    for game in predictions:
        for side in ["away_prediction", "home_prediction"]:
            pred = game.get(side)
            if pred:
                pitcher_preds.append({
                    **pred,
                    "home_team": game.get("home_team"),
                    "away_team": game.get("away_team"),
                })

    # Get ESPN odds
    props = espn_odds.get_todays_pitcher_props()

    # Match and place bets
    bets_placed = []

    for prop in props:
        prop_type = prop.get("prop_type")
        if prop_type not in ["strikeouts", "hits_allowed"]:
            continue  # Only bet on K and H for now

        line = prop.get("line")
        over_odds = prop.get("over_odds")
        under_odds = prop.get("under_odds")

        if line is None:
            continue

        # Find matching prediction
        for pred in pitcher_preds:
            if not match_prediction_to_prop(pred, prop):
                continue

            # Get model's prediction for this prop type
            stats = pred.get("expected_stats", {})
            if prop_type == "strikeouts":
                model_pred = stats.get("K")
            elif prop_type == "hits_allowed":
                model_pred = stats.get("H")
            else:
                continue

            if model_pred is None:
                continue

            # Determine bet side
            side, edge = determine_bet_side(model_pred, line, prop_type)

            # Get odds for our side
            if side == BetSide.OVER:
                odds = over_odds
            else:
                odds = under_odds

            if odds is None:
                continue

            # Create bet
            bet = Bet(
                id=None,
                created_at=None,
                game_date=game_date,
                pitcher_name=pred.get("pitcher_name"),
                prop_type=prop_type,
                line=line,
                side=side,
                odds=odds,
                stake=1.0,  # $1 per bet
                model_prediction=model_pred,
                model_edge=edge,
                bookmaker=prop.get("bookmaker", "draftkings"),
                status=BetStatus.PENDING,
                actual_result=None,
                pnl=None,
                home_team=prop.get("home_team") or pred.get("home_team"),
                away_team=prop.get("away_team") or pred.get("away_team"),
                is_auto=True,
            )

            bet_info = {
                "pitcher": bet.pitcher_name,
                "prop_type": bet.prop_type,
                "line": bet.line,
                "side": bet.side.value,
                "odds": bet.odds,
                "model_prediction": bet.model_prediction,
                "edge": bet.model_edge,
            }

            if not dry_run:
                bet_id = database.add_bet(bet, track_bankroll=False)
                bet_info["bet_id"] = bet_id

            bets_placed.append(bet_info)
            break  # Only one bet per pitcher per prop type

    return bets_placed


def get_auto_bet_summary() -> dict:
    """Get summary of auto-bet performance."""
    stats = database.get_stats_by_type(is_auto=True)
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Place auto bets for today")
    parser.add_argument("--date", help="Date to place bets for (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="Don't save bets to DB")
    args = parser.parse_args()

    game_date = args.date or date.today().isoformat()

    print(f"Placing auto bets for {game_date}...")
    bets = place_auto_bets(game_date, dry_run=args.dry_run)

    if bets:
        print(f"\nPlaced {len(bets)} auto bets:")
        for bet in bets:
            print(f"  {bet['pitcher']}: {bet['prop_type']} {bet['side'].upper()} {bet['line']} @ {bet['odds']}")
            print(f"    Model: {bet['model_prediction']:.2f}, Edge: {bet['edge']:.2f}")
    else:
        print("No bets placed")

    # Show summary
    print("\n--- Auto Bet Summary ---")
    summary = get_auto_bet_summary()
    print(f"Total bets: {summary['total_bets']}")
    print(f"Record: {summary['wins']}W-{summary['losses']}L-{summary['pushes']}P")
    print(f"Win rate: {summary['win_rate']*100:.1f}%")
    print(f"Total P&L: ${summary['total_pnl']:.2f}")
    print(f"ROI: {summary['roi']:.1f}%")
