#!/usr/bin/env python
"""
Poll MLB API for lineup availability and trigger predictions.

Usage:
    python scripts/poll_lineups.py                    # Poll today's games
    python scripts/poll_lineups.py --date 2024-04-15  # Specific date
    python scripts/poll_lineups.py --interval 300     # Poll every 5 mins
    python scripts/poll_lineups.py --once             # Single check, no loop

Lineups typically become available ~2 hours before game time.
"""

import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.mlb_api import check_lineup_status
from src.game_predictor import GamePredictor, format_prediction_summary


def run_predictions(game_date: str, output_dir: str = "data") -> dict:
    """Run predictions and save results."""
    print(f"\n{'='*60}")
    print(f"Running predictions for {game_date}")
    print(f"{'='*60}")

    predictor = GamePredictor()
    predictions = predictor.predict_day(game_date)

    # Save to JSON
    output_path = Path(output_dir) / f"predictions_{game_date}.json"
    with open(output_path, "w") as f:
        # Convert to JSON-serializable format
        json_data = []
        for p in predictions:
            game_data = {
                "game_pk": p["game_pk"],
                "away_team": p["away_team"],
                "home_team": p["home_team"],
                "status": p["status"],
            }
            if p["away_prediction"]:
                game_data["away_prediction"] = {
                    "pitcher_name": p["away_prediction"]["pitcher_name"],
                    "expected_stats": p["away_prediction"]["expected_stats"],
                }
            if p["home_prediction"]:
                game_data["home_prediction"] = {
                    "pitcher_name": p["home_prediction"]["pitcher_name"],
                    "expected_stats": p["home_prediction"]["expected_stats"],
                }
            json_data.append(game_data)

        json.dump(json_data, f, indent=2)

    print(f"Saved predictions to {output_path}")

    # Print summary
    for game in predictions:
        print(f"\n{game['away_team']} @ {game['home_team']}")
        if game["away_prediction"]:
            print(f"  [Away] {format_prediction_summary(game['away_prediction'])}")
        if game["home_prediction"]:
            print(f"  [Home] {format_prediction_summary(game['home_prediction'])}")

    return {
        "predictions": predictions,
        "output_path": str(output_path),
    }


def poll_for_lineups(
    game_date: str,
    interval: int = 300,
    min_lineups_pct: float = 0.8,
    max_wait_hours: float = 6,
) -> bool:
    """
    Poll until lineups are available, then run predictions.

    Args:
        game_date: Date string (YYYY-MM-DD)
        interval: Seconds between polls
        min_lineups_pct: Minimum % of games with lineups to trigger
        max_wait_hours: Maximum hours to wait

    Returns:
        True if predictions were run, False if timed out
    """
    start_time = datetime.now()
    max_wait = max_wait_hours * 3600

    print(f"Polling for lineups on {game_date}")
    print(f"Interval: {interval}s, Min coverage: {min_lineups_pct:.0%}")
    print(f"Max wait: {max_wait_hours} hours")
    print("-" * 40)

    while True:
        elapsed = (datetime.now() - start_time).total_seconds()
        if elapsed > max_wait:
            print(f"\nTimeout after {max_wait_hours} hours")
            return False

        status = check_lineup_status(game_date)
        total = status["total_games"]
        with_lineups = status["games_with_lineups"]

        if total == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] No games scheduled")
            return False

        coverage = with_lineups / total
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            f"Lineups: {with_lineups}/{total} ({coverage:.0%})"
        )

        if coverage >= min_lineups_pct:
            print(f"\nLineup threshold reached!")
            run_predictions(game_date)
            return True

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(
        description="Poll for lineups and run predictions"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=date.today().strftime("%Y-%m-%d"),
        help="Game date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Poll interval in seconds (default: 300)",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.8,
        help="Minimum lineup coverage to trigger (default: 0.8)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once without polling",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run predictions immediately regardless of lineup status",
    )

    args = parser.parse_args()

    if args.force:
        run_predictions(args.date)
    elif args.once:
        status = check_lineup_status(args.date)
        print(f"Games: {status['total_games']}")
        print(f"With lineups: {status['games_with_lineups']}")
        print(f"With pitchers: {status['games_with_probable_pitchers']}")

        if status["games_with_lineups"] > 0:
            run_predictions(args.date)
        else:
            print("No lineups available yet")
    else:
        poll_for_lineups(
            args.date,
            interval=args.interval,
            min_lineups_pct=args.min_coverage,
        )


if __name__ == "__main__":
    main()
