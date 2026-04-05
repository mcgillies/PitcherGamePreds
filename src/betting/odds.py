"""
Fetch betting odds from The-Odds-API.

API docs: https://the-odds-api.com/liveapi/guides/v4/
"""

import os
import requests
from datetime import datetime, date
from typing import Any
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("THEODDS_APIKEY")
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "baseball_mlb"

# Prop markets we care about
PROP_MARKETS = [
    "pitcher_strikeouts",
    "pitcher_hits_allowed",
    "pitcher_earned_runs",
]


def get_remaining_requests() -> dict[str, int] | None:
    """Check remaining API requests from last response headers."""
    # This gets populated after any API call
    return getattr(get_remaining_requests, "_last_quota", None)


def _update_quota(headers: dict):
    """Update quota info from response headers."""
    get_remaining_requests._last_quota = {
        "requests_remaining": int(headers.get("x-requests-remaining", 0)),
        "requests_used": int(headers.get("x-requests-used", 0)),
    }


def get_events() -> list[dict]:
    """
    Get all MLB events (games) for today and upcoming.

    Returns:
        List of event dicts with id, home_team, away_team, commence_time
    """
    if not API_KEY:
        raise ValueError("THEODDS_APIKEY not found in environment")

    url = f"{BASE_URL}/sports/{SPORT}/events"
    params = {"apiKey": API_KEY}

    resp = requests.get(url, params=params)
    _update_quota(resp.headers)
    resp.raise_for_status()

    return resp.json()


def get_player_props(event_id: str, markets: list[str] | None = None) -> dict[str, Any]:
    """
    Get player prop odds for a specific event.

    Args:
        event_id: The event ID from get_events()
        markets: List of markets to fetch (default: all PROP_MARKETS)

    Returns:
        Dict with event info and bookmaker odds
    """
    if not API_KEY:
        raise ValueError("THEODDS_APIKEY not found in environment")

    if markets is None:
        markets = PROP_MARKETS

    url = f"{BASE_URL}/sports/{SPORT}/events/{event_id}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": ",".join(markets),
        "oddsFormat": "american",
    }

    resp = requests.get(url, params=params)
    _update_quota(resp.headers)
    resp.raise_for_status()

    return resp.json()


def parse_pitcher_props(odds_data: dict) -> list[dict]:
    """
    Parse raw odds data into clean pitcher prop format.

    Returns:
        List of prop dicts:
        {
            "event_id": str,
            "pitcher_name": str,
            "prop_type": str,  # "strikeouts", "hits_allowed", "earned_runs"
            "line": float,
            "over_odds": int,
            "under_odds": int,
            "bookmaker": str,
        }
    """
    props = []
    event_id = odds_data.get("id")

    for bookmaker in odds_data.get("bookmakers", []):
        book_name = bookmaker.get("key")

        for market in bookmaker.get("markets", []):
            market_key = market.get("key", "")

            # Map market key to our prop type
            if "strikeouts" in market_key:
                prop_type = "strikeouts"
            elif "hits" in market_key:
                prop_type = "hits_allowed"
            elif "earned_runs" in market_key or "runs" in market_key:
                prop_type = "earned_runs"
            else:
                continue

            for outcome in market.get("outcomes", []):
                pitcher_name = outcome.get("description", "")
                line = outcome.get("point")
                odds = outcome.get("price")
                over_under = outcome.get("name", "").lower()

                if not pitcher_name or line is None:
                    continue

                # Find or create prop entry for this pitcher/prop
                existing = next(
                    (p for p in props
                     if p["pitcher_name"] == pitcher_name
                     and p["prop_type"] == prop_type
                     and p["bookmaker"] == book_name
                     and p["line"] == line),
                    None
                )

                if existing:
                    if over_under == "over":
                        existing["over_odds"] = odds
                    else:
                        existing["under_odds"] = odds
                else:
                    props.append({
                        "event_id": event_id,
                        "pitcher_name": pitcher_name,
                        "prop_type": prop_type,
                        "line": line,
                        "over_odds": odds if over_under == "over" else None,
                        "under_odds": odds if over_under == "under" else None,
                        "bookmaker": book_name,
                    })

    return props


def get_todays_pitcher_props() -> list[dict]:
    """
    Get all pitcher props for today's games.

    Returns:
        List of parsed prop dicts
    """
    events = get_events()
    today = date.today().isoformat()

    all_props = []

    for event in events:
        # Check if game is today
        commence = event.get("commence_time", "")
        if not commence.startswith(today):
            continue

        event_id = event.get("id")
        if not event_id:
            continue

        try:
            odds_data = get_player_props(event_id)
            props = parse_pitcher_props(odds_data)

            # Add game context
            for prop in props:
                prop["home_team"] = event.get("home_team")
                prop["away_team"] = event.get("away_team")
                prop["commence_time"] = commence

            all_props.extend(props)
        except Exception as e:
            print(f"Warning: Could not get props for event {event_id}: {e}")

    return all_props


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
    # Test the API
    print("Testing The-Odds-API...")

    try:
        events = get_events()
        print(f"Found {len(events)} MLB events")

        quota = get_remaining_requests()
        print(f"API quota: {quota}")

        if events:
            print(f"\nFirst event: {events[0].get('away_team')} @ {events[0].get('home_team')}")

            # Try to get props for first event
            event_id = events[0].get("id")
            props_data = get_player_props(event_id)
            props = parse_pitcher_props(props_data)
            print(f"Found {len(props)} pitcher props")

            for prop in props[:5]:
                print(f"  {prop['pitcher_name']}: {prop['prop_type']} {prop['line']} (O:{prop['over_odds']}, U:{prop['under_odds']})")

            quota = get_remaining_requests()
            print(f"\nAPI quota after: {quota}")
    except Exception as e:
        print(f"Error: {e}")
