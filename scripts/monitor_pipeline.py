#!/usr/bin/env python3
"""
Monitor pipeline health and alert if stale.

Checks:
1. pipeline_last_success.txt - alerts if older than threshold
2. pipeline_failure.txt - alerts if exists
3. pitches.parquet - alerts if corrupted

Run via cron: 0 9,21 * * * /path/to/python scripts/monitor_pipeline.py

Or run manually: python scripts/monitor_pipeline.py --check
"""

import argparse
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data" / "raw"

STALE_HOURS = 36  # Alert if no success in this many hours


def send_notification(title: str, message: str):
    """Send macOS notification."""
    subprocess.run([
        "osascript", "-e",
        f'display notification "{message}" with title "{title}" sound name "Basso"'
    ], capture_output=True)


def check_last_success() -> tuple[bool, str]:
    """Check if pipeline ran successfully recently."""
    success_file = LOGS_DIR / "pipeline_last_success.txt"

    if not success_file.exists():
        return False, "No success file found - pipeline may have never completed"

    try:
        last_success = datetime.fromisoformat(success_file.read_text().strip())
        age = datetime.now() - last_success

        if age > timedelta(hours=STALE_HOURS):
            hours_ago = age.total_seconds() / 3600
            return False, f"Last success was {hours_ago:.1f} hours ago ({last_success.strftime('%Y-%m-%d %H:%M')})"

        return True, f"Pipeline healthy - last success {age.total_seconds()/3600:.1f} hours ago"
    except Exception as e:
        return False, f"Error reading success file: {e}"


def check_failure_marker() -> tuple[bool, str]:
    """Check if there's a failure marker file."""
    failure_file = LOGS_DIR / "pipeline_failure.txt"

    if failure_file.exists():
        content = failure_file.read_text()[:200]
        return False, f"Failure marker exists:\n{content}"

    return True, "No failure marker"


def check_parquet_health() -> tuple[bool, str]:
    """Check if pitches.parquet is readable."""
    parquet_file = DATA_DIR / "pitches.parquet"

    if not parquet_file.exists():
        return False, "pitches.parquet does not exist"

    try:
        import pandas as pd
        df = pd.read_parquet(parquet_file, columns=['game_date'])
        latest = pd.to_datetime(df['game_date']).max()
        return True, f"Parquet healthy - data through {latest.strftime('%Y-%m-%d')}"
    except Exception as e:
        return False, f"Parquet corrupted or unreadable: {e}"


def run_checks(verbose: bool = False) -> bool:
    """Run all health checks. Returns True if all pass."""
    checks = [
        ("Last Success", check_last_success),
        ("Failure Marker", check_failure_marker),
        ("Parquet Health", check_parquet_health),
    ]

    all_passed = True
    alerts = []

    for name, check_fn in checks:
        try:
            passed, message = check_fn()
        except Exception as e:
            passed, message = False, f"Check error: {e}"

        if verbose:
            status = "✓" if passed else "✗"
            print(f"{status} {name}: {message}")

        if not passed:
            all_passed = False
            alerts.append(f"{name}: {message}")

    return all_passed, alerts


def main():
    parser = argparse.ArgumentParser(description="Monitor pipeline health")
    parser.add_argument("--check", action="store_true", help="Run checks and print status")
    parser.add_argument("--quiet", action="store_true", help="Only output on failure")
    parser.add_argument("--no-notify", action="store_true", help="Skip desktop notification")
    args = parser.parse_args()

    all_passed, alerts = run_checks(verbose=not args.quiet)

    if not all_passed:
        alert_text = "\n".join(alerts)

        if not args.quiet:
            print(f"\n⚠️  PIPELINE ISSUES DETECTED:\n{alert_text}")

        if not args.no_notify:
            send_notification(
                "Pipeline Alert",
                alerts[0][:100]  # First alert, truncated for notification
            )

        sys.exit(1)
    else:
        if not args.quiet:
            print("\n✓ All checks passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
