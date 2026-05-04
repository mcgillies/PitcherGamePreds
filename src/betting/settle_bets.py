"""
Settle pending bets using actual pitcher stats.

Pulls actual results from MLB Stats API boxscores.
"""

import sys
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.betting import database
from src.betting.database import BetStatus
from src.data.mlb_api import get_schedule, get_game_boxscore


def get_pitcher_stats_from_boxscore(boxscore: dict, team_type: str) -> list[dict]:
    """Extract pitcher stats from a boxscore."""
    pitchers = []
    team_data = boxscore.get("teams", {}).get(team_type, {})
    players = team_data.get("players", {})

    for player_id, player_data in players.items():
        stats = player_data.get("stats", {}).get("pitching", {})
        if not stats:
            continue

        pitchers.append({
            "name": player_data.get("person", {}).get("fullName", ""),
            "id": player_data.get("person", {}).get("id"),
            "strikeouts": int(stats.get("strikeOuts", 0)),
            "hits_allowed": int(stats.get("hits", 0)),
            "earned_runs": int(stats.get("earnedRuns", 0)),
            "innings_pitched": stats.get("inningsPitched", "0"),
        })

    return pitchers


def load_actual_stats(game_date: str) -> list[dict]:
    """Load actual pitcher stats for a given date from MLB API."""
    print(f"Fetching boxscores for {game_date}...")

    try:
        games = get_schedule(game_date)
    except Exception as e:
        print(f"Error fetching schedule: {e}")
        return []

    all_pitchers = []

    for game in games:
        game_pk = game.get("gamePk")
        status = game.get("status", {}).get("abstractGameState", "")

        # Only get stats from completed games
        if status != "Final":
            continue

        try:
            boxscore = get_game_boxscore(game_pk)

            # Get pitchers from both teams
            for team_type in ["away", "home"]:
                pitchers = get_pitcher_stats_from_boxscore(boxscore, team_type)
                all_pitchers.extend(pitchers)

        except Exception as e:
            print(f"  Error fetching boxscore for game {game_pk}: {e}")

    print(f"  Found stats for {len(all_pitchers)} pitchers")
    return all_pitchers


def match_pitcher_name(bet_name: str, pitchers: list[dict]) -> dict | None:
    """
    Match a bet's pitcher name to stats data.

    Returns dict with strikeouts, hits_allowed, earned_runs if found.
    """
    bet_name_lower = bet_name.lower()
    bet_parts = set(bet_name_lower.split())

    for pitcher in pitchers:
        stats_name = pitcher.get('name', '').lower()
        stats_parts = set(stats_name.split())

        # Match if 2+ name parts overlap
        if len(bet_parts & stats_parts) >= 2:
            return pitcher

        # Also try exact match
        if bet_name_lower == stats_name:
            return pitcher

    return None


def settle_bets_for_date(game_date: str, dry_run: bool = False) -> list[dict]:
    """
    Settle all pending bets for a given date.

    Args:
        game_date: Date to settle (YYYY-MM-DD)
        dry_run: If True, don't actually update DB

    Returns:
        List of settlement results
    """
    # Get pending bets for this date
    pending = database.get_bets(status=BetStatus.PENDING)
    date_bets = [b for b in pending if b.game_date == game_date]

    if not date_bets:
        print(f"No pending bets for {game_date}")
        return []

    print(f"Found {len(date_bets)} pending bets for {game_date}")

    # Load actual stats from MLB API
    pitchers = load_actual_stats(game_date)

    if not pitchers:
        print(f"No stats available for {game_date} yet")
        return []

    results = []

    for bet in date_bets:
        # Find matching stats
        actual_stats = match_pitcher_name(bet.pitcher_name, pitchers)

        if actual_stats is None:
            print(f"  No stats found for {bet.pitcher_name}")
            continue

        # Get actual result for this prop type
        actual_result = actual_stats.get(bet.prop_type)

        if actual_result is None:
            print(f"  No {bet.prop_type} stat for {bet.pitcher_name}")
            continue

        result_info = {
            'bet_id': bet.id,
            'pitcher': bet.pitcher_name,
            'prop_type': bet.prop_type,
            'line': bet.line,
            'side': bet.side.value,
            'actual': actual_result,
            'model_prediction': bet.model_prediction,
            'is_auto': bet.is_auto,
        }

        if not dry_run:
            # Settle the bet
            outcome = database.settle_bet(bet.id, actual_result)
            result_info['status'] = outcome['status']
            result_info['pnl'] = outcome['pnl']
        else:
            # Simulate outcome
            if actual_result == bet.line:
                result_info['status'] = 'push'
                result_info['pnl'] = 0
            elif bet.side.value == 'over':
                if actual_result > bet.line:
                    result_info['status'] = 'won'
                    result_info['pnl'] = bet.stake * (bet.odds / 100 if bet.odds > 0 else 100 / abs(bet.odds))
                else:
                    result_info['status'] = 'lost'
                    result_info['pnl'] = -bet.stake
            else:  # under
                if actual_result < bet.line:
                    result_info['status'] = 'won'
                    result_info['pnl'] = bet.stake * (bet.odds / 100 if bet.odds > 0 else 100 / abs(bet.odds))
                else:
                    result_info['status'] = 'lost'
                    result_info['pnl'] = -bet.stake

        results.append(result_info)
        print(f"  {bet.pitcher_name} {bet.prop_type} {bet.side.value} {bet.line}: actual={actual_result} -> {result_info['status']} (${result_info['pnl']:+.2f})")

    return results


def settle_yesterday(dry_run: bool = False) -> list[dict]:
    """Settle all pending bets from yesterday."""
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    return settle_bets_for_date(yesterday, dry_run)


def settle_all_pending(dry_run: bool = False) -> list[dict]:
    """Settle all pending bets that have stats available."""
    pending = database.get_bets(status=BetStatus.PENDING)

    # Group by date
    dates = sorted(set(b.game_date for b in pending))

    all_results = []
    for game_date in dates:
        # Don't settle today's bets (games may not be over)
        if game_date >= date.today().isoformat():
            continue

        results = settle_bets_for_date(game_date, dry_run)
        all_results.extend(results)

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Settle pending bets")
    parser.add_argument("--date", help="Specific date to settle (YYYY-MM-DD)")
    parser.add_argument("--yesterday", action="store_true", help="Settle yesterday's bets")
    parser.add_argument("--all", action="store_true", help="Settle all pending bets with available stats")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually update DB")
    args = parser.parse_args()

    if args.date:
        results = settle_bets_for_date(args.date, dry_run=args.dry_run)
    elif args.yesterday:
        results = settle_yesterday(dry_run=args.dry_run)
    elif args.all:
        results = settle_all_pending(dry_run=args.dry_run)
    else:
        # Default: settle yesterday
        results = settle_yesterday(dry_run=args.dry_run)

    if results:
        wins = sum(1 for r in results if r['status'] == 'won')
        losses = sum(1 for r in results if r['status'] == 'lost')
        pushes = sum(1 for r in results if r['status'] == 'push')
        total_pnl = sum(r['pnl'] for r in results)

        print(f"\n--- Settlement Summary ---")
        print(f"Settled: {len(results)} bets")
        print(f"Record: {wins}W-{losses}L-{pushes}P")
        print(f"P&L: ${total_pnl:+.2f}")
    else:
        print("No bets settled")
