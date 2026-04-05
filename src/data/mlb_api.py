"""
MLB Stats API client for fetching schedules, lineups, and game data.

API Documentation: https://statsapi.mlb.com/docs/
"""

import requests
from datetime import datetime, date
from typing import Any

BASE_URL = "https://statsapi.mlb.com/api/v1"


def get_schedule(
    game_date: str | date,
    sport_id: int = 1,  # 1 = MLB
) -> list[dict]:
    """
    Get scheduled games for a date.

    Args:
        game_date: Date string (YYYY-MM-DD) or date object
        sport_id: Sport ID (1 = MLB)

    Returns:
        List of game dictionaries
    """
    if isinstance(game_date, date):
        game_date = game_date.strftime("%Y-%m-%d")

    url = f"{BASE_URL}/schedule"
    params = {
        "sportId": sport_id,
        "date": game_date,
        "hydrate": "probablePitcher,lineups,team",
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            games.append(game)

    return games


def get_game_feed(game_pk: int) -> dict:
    """
    Get live game feed with full details.

    Args:
        game_pk: Game primary key

    Returns:
        Game feed dictionary
    """
    url = f"{BASE_URL}.1/game/{game_pk}/feed/live"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def get_game_boxscore(game_pk: int) -> dict:
    """
    Get game boxscore.

    Args:
        game_pk: Game primary key

    Returns:
        Boxscore dictionary
    """
    url = f"{BASE_URL}/game/{game_pk}/boxscore"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def get_lineup_from_boxscore(game_pk: int, team_type: str = "away") -> list[dict] | None:
    """
    Get lineup from boxscore (works for started/completed games).

    Args:
        game_pk: Game primary key
        team_type: "away" or "home"

    Returns:
        List of batter dicts with id, name, position, batting_order
        None if not available
    """
    try:
        boxscore = get_game_boxscore(game_pk)
        team = boxscore.get("teams", {}).get(team_type, {})

        batting_order = team.get("battingOrder", [])
        players = team.get("players", {})

        if not batting_order:
            return None

        lineup = []
        for i, player_id in enumerate(batting_order):
            player_key = f"ID{player_id}"
            player_data = players.get(player_key, {})
            person = player_data.get("person", {})
            position = player_data.get("position", {})

            lineup.append({
                "batter_id": player_id,
                "batter_name": person.get("fullName", f"Unknown ({player_id})"),
                "position": position.get("abbreviation", ""),
                "batting_order": i + 1,
            })

        return lineup if lineup else None

    except Exception as e:
        print(f"Warning: Could not get boxscore lineup for game {game_pk}: {e}")
        return None


def parse_lineup(game: dict, team_type: str = "away") -> list[dict] | None:
    """
    Parse lineup from game data.

    Args:
        game: Game dictionary from schedule
        team_type: "away" or "home"

    Returns:
        List of batter dicts with id, name, position, batting_order
        None if lineup not available
    """
    # Lineups are at game['lineups']['awayPlayers'] or game['lineups']['homePlayers']
    lineups = game.get("lineups", {})
    key = "awayPlayers" if team_type == "away" else "homePlayers"
    lineup_data = lineups.get(key, [])

    if not lineup_data:
        # Fallback: check old location under teams
        teams = game.get("teams", {})
        team = teams.get(team_type, {})
        lineup_data = team.get("lineup", [])

    if not lineup_data:
        return None

    lineup = []
    for i, player in enumerate(lineup_data):
        lineup.append({
            "batter_id": player.get("id"),
            "batter_name": player.get("fullName"),
            "position": player.get("primaryPosition", {}).get("abbreviation"),
            "batting_order": i + 1,
        })

    return lineup


def parse_probable_pitcher(game: dict, team_type: str = "away") -> dict | None:
    """
    Parse probable pitcher from game data.

    Args:
        game: Game dictionary from schedule
        team_type: "away" or "home"

    Returns:
        Pitcher dict with id, name, throws
        None if not available
    """
    teams = game.get("teams", {})
    team = teams.get(team_type, {})
    pitcher = team.get("probablePitcher")

    if not pitcher:
        return None

    return {
        "pitcher_id": pitcher.get("id"),
        "pitcher_name": pitcher.get("fullName"),
    }


def get_games_with_lineups(game_date: str | date) -> list[dict]:
    """
    Get all games for a date with parsed lineups and pitchers.

    For pre-game: lineups from schedule endpoint (if available)
    For started/completed: lineups from boxscore endpoint

    Args:
        game_date: Date string (YYYY-MM-DD) or date object

    Returns:
        List of parsed game dicts with lineup availability status
    """
    games = get_schedule(game_date)
    parsed_games = []

    for game in games:
        game_pk = game.get("gamePk")
        game_time = game.get("gameDate")
        status = game.get("status", {}).get("detailedState", "Unknown")

        away_team = game.get("teams", {}).get("away", {}).get("team", {})
        home_team = game.get("teams", {}).get("home", {}).get("team", {})

        # Try schedule endpoint first
        away_lineup = parse_lineup(game, "away")
        home_lineup = parse_lineup(game, "home")

        # If game has started/completed, get lineup from boxscore
        game_started = status in ["In Progress", "Final", "Game Over", "Completed Early"]
        if game_started and (away_lineup is None or home_lineup is None):
            if away_lineup is None:
                away_lineup = get_lineup_from_boxscore(game_pk, "away")
            if home_lineup is None:
                home_lineup = get_lineup_from_boxscore(game_pk, "home")

        parsed = {
            "game_pk": game_pk,
            "game_time": game_time,
            "status": status,
            "away_team": {
                "id": away_team.get("id"),
                "name": away_team.get("name"),
                "abbreviation": away_team.get("abbreviation"),
            },
            "home_team": {
                "id": home_team.get("id"),
                "name": home_team.get("name"),
                "abbreviation": home_team.get("abbreviation"),
            },
            "away_pitcher": parse_probable_pitcher(game, "away"),
            "home_pitcher": parse_probable_pitcher(game, "home"),
            "away_lineup": away_lineup,
            "home_lineup": home_lineup,
        }

        # Lineup availability flags
        parsed["away_lineup_available"] = parsed["away_lineup"] is not None
        parsed["home_lineup_available"] = parsed["home_lineup"] is not None
        parsed["lineups_available"] = (
            parsed["away_lineup_available"] and parsed["home_lineup_available"]
        )

        parsed_games.append(parsed)

    return parsed_games


def get_pitcher_game_logs(
    pitcher_id: int,
    season: int | None = None,
    limit: int = 20,
) -> list[dict]:
    """
    Get recent game logs for a pitcher.

    Args:
        pitcher_id: MLB player ID
        season: Season year (defaults to current)
        limit: Max games to return

    Returns:
        List of game log dicts
    """
    if season is None:
        season = datetime.now().year

    url = f"{BASE_URL}/people/{pitcher_id}/stats"
    params = {
        "stats": "gameLog",
        "season": season,
        "group": "pitching",
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    game_logs = []
    for stat_group in data.get("stats", []):
        for split in stat_group.get("splits", []):
            stat = split.get("stat", {})
            game_logs.append({
                "date": split.get("date"),
                "opponent": split.get("opponent", {}).get("name"),
                "is_home": split.get("isHome"),
                "innings_pitched": stat.get("inningsPitched"),
                "batters_faced": stat.get("battersFaced"),
                "pitches_thrown": stat.get("numberOfPitches"),
                "strikeouts": stat.get("strikeOuts"),
                "walks": stat.get("baseOnBalls"),
                "hits": stat.get("hits"),
                "home_runs": stat.get("homeRuns"),
                "earned_runs": stat.get("earnedRuns"),
            })

    # Sort by date descending, limit
    game_logs.sort(key=lambda x: x["date"] or "", reverse=True)
    return game_logs[:limit]


def check_lineup_status(game_date: str | date) -> dict[str, Any]:
    """
    Check lineup availability status for all games on a date.

    Useful for determining when to run predictions.

    Args:
        game_date: Date string (YYYY-MM-DD) or date object

    Returns:
        Summary dict with counts and game details
    """
    games = get_games_with_lineups(game_date)

    total = len(games)
    with_lineups = sum(1 for g in games if g["lineups_available"])
    with_pitchers = sum(
        1 for g in games
        if g["away_pitcher"] and g["home_pitcher"]
    )

    return {
        "date": game_date if isinstance(game_date, str) else game_date.strftime("%Y-%m-%d"),
        "total_games": total,
        "games_with_lineups": with_lineups,
        "games_with_probable_pitchers": with_pitchers,
        "all_lineups_available": with_lineups == total and total > 0,
        "games": games,
    }


if __name__ == "__main__":
    # Example usage
    from datetime import date

    today = date.today().strftime("%Y-%m-%d")
    print(f"Checking games for {today}...")

    status = check_lineup_status(today)
    print(f"Total games: {status['total_games']}")
    print(f"With lineups: {status['games_with_lineups']}")
    print(f"With probable pitchers: {status['games_with_probable_pitchers']}")

    for game in status["games"]:
        away = game["away_team"]["abbreviation"]
        home = game["home_team"]["abbreviation"]
        lineups = "✓" if game["lineups_available"] else "✗"
        print(f"  {away} @ {home} - Lineups: {lineups}")
