"""
Fetch betting odds from ESPN's hidden API (free, no auth required).

Uses DraftKings odds via ESPN's propBets endpoint.
"""

import requests
from datetime import date, datetime
from typing import Any


BASE_URL = "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb"
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"

# ESPN prop type IDs
PROP_TYPES = {
    "46": "strikeouts",
    "47": "pitcher_win",
    "48": "hits_allowed",  # singles
    "49": "doubles_allowed",
    "52": "total_bases",
}

# Map our prop types to ESPN IDs
PROP_TYPE_TO_ID = {
    "strikeouts": "46",
    "hits_allowed": "48",
    "earned_runs": None,  # Not available directly
}


def get_todays_events() -> list[dict]:
    """
    Get all MLB events (games) for today.

    Returns:
        List of event dicts with id, home_team, away_team, status
    """
    resp = requests.get(SCOREBOARD_URL)
    resp.raise_for_status()
    data = resp.json()

    events = []
    for event in data.get("events", []):
        event_id = event.get("id")
        name = event.get("shortName", "")
        status = event.get("status", {}).get("type", {}).get("state", "")

        # Parse teams from shortName (e.g., "TB @ CLE")
        if " @ " in name:
            away, home = name.split(" @ ")
        else:
            away, home = "", ""

        events.append({
            "id": event_id,
            "away_team": away,
            "home_team": home,
            "status": status,
            "name": name,
        })

    return events


def get_event_odds(event_id: str) -> dict:
    """
    Get betting odds for a specific event.

    Returns:
        Dict with moneyline, spread, over/under
    """
    url = f"{BASE_URL}/events/{event_id}/competitions/{event_id}/odds"
    resp = requests.get(url, params={"lang": "en", "region": "us"})
    resp.raise_for_status()
    return resp.json()


def get_prop_bets(event_id: str, provider_id: str = "100") -> list[dict]:
    """
    Get player prop bets for a specific event.

    Args:
        event_id: ESPN event ID
        provider_id: Betting provider (100=DraftKings, 200=DraftKings Live)

    Returns:
        List of prop bet dicts
    """
    url = f"{BASE_URL}/events/{event_id}/competitions/{event_id}/odds/{provider_id}/propBets"
    params = {"lang": "en", "region": "us", "limit": 200}

    resp = requests.get(url, params=params)
    if resp.status_code == 404:
        return []  # No props available for this game
    resp.raise_for_status()

    data = resp.json()
    return data.get("items", [])


def get_athlete_name(athlete_ref: str) -> str:
    """Fetch athlete name from ESPN API reference."""
    try:
        resp = requests.get(athlete_ref)
        resp.raise_for_status()
        return resp.json().get("displayName", "Unknown")
    except:
        return "Unknown"


def parse_prop_bets(prop_bets: list[dict], event_info: dict) -> list[dict]:
    """
    Parse raw ESPN prop bets into clean format.

    ESPN returns two entries per prop line - one for over, one for under.
    We group them by athlete/prop_type/line.

    Returns:
        List of prop dicts with pitcher_name, prop_type, line, odds
    """
    # Group props by athlete/type/line
    grouped = {}
    athlete_cache = {}

    for prop in prop_bets:
        prop_type_id = prop.get("type", {}).get("id")
        prop_type_name = PROP_TYPES.get(prop_type_id)

        if not prop_type_name:
            continue

        # Get athlete name (with caching)
        athlete_ref = prop.get("athlete", {}).get("$ref", "")
        if athlete_ref not in athlete_cache:
            athlete_cache[athlete_ref] = get_athlete_name(athlete_ref)
        athlete_name = athlete_cache[athlete_ref]

        # Parse odds
        odds_data = prop.get("odds", {})
        line = float(odds_data.get("total", {}).get("value", 0))
        american_odds = odds_data.get("american", {}).get("value", "0")

        try:
            odds_int = int(american_odds.replace("+", ""))
        except:
            odds_int = 0

        # Create unique key for grouping
        key = (athlete_name, prop_type_name, line)

        if key not in grouped:
            grouped[key] = {
                "event_id": event_info.get("id"),
                "pitcher_name": athlete_name,
                "prop_type": prop_type_name,
                "line": line,
                "odds_list": [],
                "bookmaker": "draftkings",
                "home_team": event_info.get("home_team"),
                "away_team": event_info.get("away_team"),
            }
        grouped[key]["odds_list"].append(odds_int)

    # Convert grouped data to final format
    props = []
    for key, data in grouped.items():
        odds_list = sorted(data["odds_list"])

        # Usually: negative odds = favored side, positive = underdog
        # For strikeouts, if over is favored it means high K expectation
        if len(odds_list) >= 2:
            # Lower odds (more negative) is the favored side
            under_odds = odds_list[0]  # More negative = under favored
            over_odds = odds_list[-1]  # Less negative/positive = over
        elif len(odds_list) == 1:
            over_odds = odds_list[0]
            under_odds = None
        else:
            continue

        props.append({
            "event_id": data["event_id"],
            "pitcher_name": data["pitcher_name"],
            "prop_type": data["prop_type"],
            "line": data["line"],
            "over_odds": over_odds,
            "under_odds": under_odds,
            "bookmaker": data["bookmaker"],
            "home_team": data["home_team"],
            "away_team": data["away_team"],
        })

    return props


def get_todays_pitcher_props() -> list[dict]:
    """
    Get all pitcher props for today's games.

    Returns:
        List of parsed prop dicts
    """
    events = get_todays_events()
    all_props = []

    for event in events:
        event_id = event.get("id")
        if not event_id:
            continue

        # Skip games that have already started
        if event.get("status") in ["in", "post"]:
            continue

        try:
            prop_bets = get_prop_bets(event_id)
            if prop_bets:
                props = parse_prop_bets(prop_bets, event)
                all_props.extend(props)
        except Exception as e:
            print(f"Warning: Could not get props for event {event_id}: {e}")

    return all_props


def get_strikeout_props() -> list[dict]:
    """Get only strikeout props for today."""
    all_props = get_todays_pitcher_props()
    return [p for p in all_props if p["prop_type"] == "strikeouts"]


def american_to_implied_prob(american_odds: int) -> float:
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def american_to_decimal(american_odds: int) -> float:
    """Convert American odds to decimal odds."""
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


if __name__ == "__main__":
    # Test the ESPN API
    print("Testing ESPN API for betting odds...")

    events = get_todays_events()
    print(f"Found {len(events)} MLB events today")

    for event in events[:3]:
        print(f"  {event['away_team']} @ {event['home_team']} ({event['status']})")

    print("\nFetching pitcher props...")
    props = get_todays_pitcher_props()
    print(f"Found {len(props)} pitcher props")

    # Show strikeout props
    k_props = [p for p in props if p["prop_type"] == "strikeouts"]
    print(f"\nStrikeout props ({len(k_props)}):")
    for prop in k_props[:10]:
        print(f"  {prop['pitcher_name']}: O/U {prop['line']} K (O:{prop['over_odds']}, U:{prop['under_odds']})")
