#!/usr/bin/env python3
"""
Cron runner for automated betting operations.

Usage:
    python src/betting/cron_runner.py hourly    # Check lineups, place auto bets
    python src/betting/cron_runner.py daily     # Settle yesterday's bets
    python src/betting/cron_runner.py both      # Do both

Cron examples:
    # Every hour from 10am-6pm ET during baseball season
    0 10-18 * 3-10 * cd /path/to/PitcherGamePreds && python src/betting/cron_runner.py hourly

    # Daily at 6am ET to settle yesterday's bets (after overnight data update)
    0 6 * 3-10 * cd /path/to/PitcherGamePreds && python src/betting/cron_runner.py daily
"""

import sys
from pathlib import Path
from datetime import date, datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.betting.auto_bet import place_auto_bets, get_auto_bet_summary
from src.betting.settle_bets import settle_yesterday, settle_all_pending


def log(msg: str):
    """Print timestamped log message."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def run_hourly():
    """
    Hourly task: Check for lineups and place auto bets.

    - Only runs if we haven't already placed bets for today
    - Tries to place bets on all available matchups
    """
    log("Running hourly auto-bet check...")

    today = date.today().isoformat()

    try:
        bets = place_auto_bets(game_date=today)

        if bets:
            log(f"Placed {len(bets)} auto bets:")
            for bet in bets:
                log(f"  {bet['pitcher']}: {bet['prop_type']} {bet['side'].upper()} {bet['line']} @ {bet['odds']:+d}")
                log(f"    Model: {bet['model_prediction']:.2f}, Edge: {bet['edge']:.2f}")
        else:
            log("No new auto bets placed (may already exist or no props available)")

    except Exception as e:
        log(f"Error placing auto bets: {e}")
        import traceback
        traceback.print_exc()

    # Show current summary
    summary = get_auto_bet_summary()
    log(f"Auto-bet record: {summary['wins']}W-{summary['losses']}L-{summary['pushes']}P ({summary['pending']} pending)")
    log(f"Total P&L: ${summary['total_pnl']:+.2f} ({summary['roi']:.1f}% ROI)")


def run_daily():
    """
    Daily task: Settle yesterday's bets.

    - Runs after overnight data update
    - Settles all pending bets that have stats available
    """
    log("Running daily bet settlement...")

    try:
        # Settle all pending bets (not just yesterday, in case some were missed)
        results = settle_all_pending()

        if results:
            wins = sum(1 for r in results if r['status'] == 'won')
            losses = sum(1 for r in results if r['status'] == 'lost')
            pushes = sum(1 for r in results if r['status'] == 'push')
            total_pnl = sum(r['pnl'] for r in results)

            log(f"Settled {len(results)} bets: {wins}W-{losses}L-{pushes}P, P&L: ${total_pnl:+.2f}")
        else:
            log("No bets to settle")

    except Exception as e:
        log(f"Error settling bets: {e}")
        import traceback
        traceback.print_exc()

    # Show updated summary
    summary = get_auto_bet_summary()
    log(f"Auto-bet total: {summary['wins']}W-{summary['losses']}L-{summary['pushes']}P")
    log(f"Total P&L: ${summary['total_pnl']:+.2f} ({summary['roi']:.1f}% ROI)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Cron runner for betting automation")
    parser.add_argument("task", choices=["hourly", "daily", "both"],
                       help="Which task to run")
    args = parser.parse_args()

    log("=" * 50)

    if args.task in ["hourly", "both"]:
        run_hourly()

    if args.task in ["daily", "both"]:
        run_daily()

    log("=" * 50)


if __name__ == "__main__":
    main()
