#!/usr/bin/env python3
"""
Auto-bet daemon - runs in background, places bets automatically.

Start with: python src/betting/auto_bet_daemon.py &
Or: nohup python src/betting/auto_bet_daemon.py > logs/auto_bet_daemon.log 2>&1 &

Checks every 30 minutes during betting hours (10am-6pm Mountain Time).
Sleeps overnight, resumes in the morning.
"""

import sys
import time
import gc
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytz

from src.betting.auto_bet import place_auto_bets, get_auto_bet_summary
from src.betting.settle_bets import settle_all_pending


def clear_memory():
    """Force garbage collection to free memory."""
    gc.collect()


MOUNTAIN = pytz.timezone('America/Denver')
CHECK_INTERVAL_MINUTES = 30
BETTING_START_HOUR = 10  # 10am MT
BETTING_END_HOUR = 18    # 6pm MT


def log(msg: str):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}", flush=True)
    # Also write to a dedicated log file for reliability
    try:
        log_path = Path(__file__).parent.parent.parent / "logs" / "auto_bet_daemon.log"
        with open(log_path, "a") as f:
            f.write(f"[{timestamp}] {msg}\n")
            f.flush()
    except:
        pass


def is_betting_hours() -> bool:
    """Check if current time is within betting hours."""
    now = datetime.now(MOUNTAIN)
    return BETTING_START_HOUR <= now.hour < BETTING_END_HOUR


def minutes_until_betting_hours() -> int:
    """Calculate minutes until betting hours start."""
    now = datetime.now(MOUNTAIN)

    if now.hour >= BETTING_END_HOUR:
        # After 6pm - wait until 10am tomorrow
        tomorrow_10am = now.replace(hour=BETTING_START_HOUR, minute=0, second=0, microsecond=0)
        tomorrow_10am = tomorrow_10am + timedelta(days=1)
        delta = tomorrow_10am - now
    elif now.hour < BETTING_START_HOUR:
        # Before 10am - wait until 10am today
        today_10am = now.replace(hour=BETTING_START_HOUR, minute=0, second=0, microsecond=0)
        delta = today_10am - now
    else:
        # During betting hours
        return 0

    return int(delta.total_seconds() / 60)


def settle_previous_bets():
    """Settle any pending bets from previous days."""
    log("Settling pending bets from previous days...")

    try:
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
    finally:
        clear_memory()


def check_and_place_bets():
    """Check for props and place auto bets."""
    log("Checking for props...")

    try:
        bets = place_auto_bets()

        if bets:
            log(f"Placed {len(bets)} auto bets:")
            for bet in bets:
                log(f"  {bet['pitcher']}: {bet['prop_type']} {bet['side'].upper()} {bet['line']} @ {bet['odds']:+d}")
        else:
            log("No new bets (already placed or no props available)")

        # Show current stats
        summary = get_auto_bet_summary()
        log(f"Record: {summary['wins']}W-{summary['losses']}L-{summary['pushes']}P | P&L: ${summary['total_pnl']:+.2f} ({summary['roi']:.1f}% ROI)")

    except Exception as e:
        log(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        clear_memory()


def run_daemon():
    """Main daemon loop."""
    log("=" * 50)
    log("Auto-bet daemon starting")
    log(f"Betting hours: {BETTING_START_HOUR}:00 - {BETTING_END_HOUR}:00 Mountain Time")
    log(f"Check interval: {CHECK_INTERVAL_MINUTES} minutes")
    log("Auto-settles previous day's bets at first check each day")
    log("=" * 50)

    last_settle_date = None  # Track when we last settled

    while True:
        try:
            if is_betting_hours():
                # Check if we need to settle (first check of the day)
                today = datetime.now(MOUNTAIN).strftime('%Y-%m-%d')
                if last_settle_date != today:
                    settle_previous_bets()
                    last_settle_date = today

                check_and_place_bets()
                log(f"Sleeping {CHECK_INTERVAL_MINUTES} minutes...")
                time.sleep(CHECK_INTERVAL_MINUTES * 60)
            else:
                wait_minutes = minutes_until_betting_hours()
                log(f"Outside betting hours. Sleeping {wait_minutes} minutes until {BETTING_START_HOUR}:00 MT...")
                time.sleep(wait_minutes * 60)

        except KeyboardInterrupt:
            log("Daemon stopped by user")
            break
        except Exception as e:
            log(f"Daemon error: {e}")
            log("Retrying in 5 minutes...")
            time.sleep(300)


if __name__ == "__main__":
    run_daemon()
