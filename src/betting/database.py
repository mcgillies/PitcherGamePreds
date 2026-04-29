"""
SQLite database for tracking bets and bankroll.
"""

import sqlite3
from datetime import datetime, date
from pathlib import Path
from typing import Any
from dataclasses import dataclass
from enum import Enum


DB_PATH = Path(__file__).parent.parent.parent / "data" / "betting.db"


class BetStatus(str, Enum):
    PENDING = "pending"
    WON = "won"
    LOST = "lost"
    PUSH = "push"
    CANCELLED = "cancelled"


class BetSide(str, Enum):
    OVER = "over"
    UNDER = "under"


@dataclass
class Bet:
    id: int | None
    created_at: datetime
    game_date: str
    pitcher_name: str
    prop_type: str  # strikeouts, hits_allowed, earned_runs
    line: float
    side: BetSide
    odds: int  # American odds
    stake: float
    model_prediction: float
    model_edge: float  # How much we think line is off
    bookmaker: str
    status: BetStatus
    actual_result: float | None
    pnl: float | None
    home_team: str
    away_team: str
    is_auto: bool = False  # True for auto-bets, False for manual bets

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "game_date": self.game_date,
            "pitcher_name": self.pitcher_name,
            "prop_type": self.prop_type,
            "line": self.line,
            "side": self.side.value if isinstance(self.side, BetSide) else self.side,
            "odds": self.odds,
            "stake": self.stake,
            "model_prediction": self.model_prediction,
            "model_edge": self.model_edge,
            "bookmaker": self.bookmaker,
            "status": self.status.value if isinstance(self.status, BetStatus) else self.status,
            "actual_result": self.actual_result,
            "pnl": self.pnl,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "is_auto": self.is_auto,
        }


def get_connection() -> sqlite3.Connection:
    """Get database connection."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables."""
    conn = get_connection()
    cursor = conn.cursor()

    # Bets table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            game_date TEXT NOT NULL,
            pitcher_name TEXT NOT NULL,
            prop_type TEXT NOT NULL,
            line REAL NOT NULL,
            side TEXT NOT NULL,
            odds INTEGER NOT NULL,
            stake REAL NOT NULL,
            model_prediction REAL NOT NULL,
            model_edge REAL NOT NULL,
            bookmaker TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            actual_result REAL,
            pnl REAL,
            home_team TEXT,
            away_team TEXT,
            is_auto INTEGER DEFAULT 0
        )
    """)

    # Add is_auto column if it doesn't exist (migration for existing DBs)
    try:
        cursor.execute("ALTER TABLE bets ADD COLUMN is_auto INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Bankroll history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bankroll_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            balance REAL NOT NULL,
            change_amount REAL,
            change_reason TEXT
        )
    """)

    # Settings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    # Initialize starting bankroll if not exists
    cursor.execute("SELECT value FROM settings WHERE key = 'starting_bankroll'")
    if cursor.fetchone() is None:
        cursor.execute("INSERT INTO settings (key, value) VALUES ('starting_bankroll', '10.0')")
        cursor.execute("INSERT INTO bankroll_history (balance, change_amount, change_reason) VALUES (10.0, 10.0, 'Initial bankroll')")

    conn.commit()
    conn.close()


def get_setting(key: str, default: str = "") -> str:
    """Get a setting value."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
    row = cursor.fetchone()
    conn.close()
    return row["value"] if row else default


def set_setting(key: str, value: str):
    """Set a setting value."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
        (key, value)
    )
    conn.commit()
    conn.close()


def get_current_bankroll() -> float:
    """Get current bankroll balance."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT balance FROM bankroll_history ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    return row["balance"] if row else 10.0


def update_bankroll(new_balance: float, change_amount: float, reason: str, conn: sqlite3.Connection | None = None):
    """Update bankroll with new balance."""
    should_close = conn is None
    if conn is None:
        conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO bankroll_history (balance, change_amount, change_reason) VALUES (?, ?, ?)",
        (new_balance, change_amount, reason)
    )
    if should_close:
        conn.commit()
        conn.close()


def add_bet(bet: Bet, track_bankroll: bool = True) -> int:
    """Add a new bet to the database. Returns bet ID.

    Args:
        bet: The bet to add
        track_bankroll: If True, deduct stake from bankroll. Set False for auto-bets.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO bets (
            game_date, pitcher_name, prop_type, line, side, odds, stake,
            model_prediction, model_edge, bookmaker, status, home_team, away_team, is_auto
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        bet.game_date,
        bet.pitcher_name,
        bet.prop_type,
        bet.line,
        bet.side.value if isinstance(bet.side, BetSide) else bet.side,
        bet.odds,
        bet.stake,
        bet.model_prediction,
        bet.model_edge,
        bet.bookmaker,
        bet.status.value if isinstance(bet.status, BetStatus) else bet.status,
        bet.home_team,
        bet.away_team,
        1 if bet.is_auto else 0,
    ))

    bet_id = cursor.lastrowid

    # Update bankroll (subtract stake) - skip for auto bets
    if track_bankroll and not bet.is_auto:
        cursor.execute("SELECT balance FROM bankroll_history ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        current = row["balance"] if row else 10.0
        new_balance = current - bet.stake
        update_bankroll(new_balance, -bet.stake, f"Bet #{bet_id} placed", conn)

    conn.commit()
    conn.close()

    return bet_id


def settle_bet(bet_id: int, actual_result: float):
    """
    Settle a bet with the actual result.

    Determines win/loss/push and updates bankroll.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Get bet details
    cursor.execute("SELECT * FROM bets WHERE id = ?", (bet_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise ValueError(f"Bet {bet_id} not found")

    line = row["line"]
    side = row["side"]
    odds = row["odds"]
    stake = row["stake"]

    # Determine outcome
    if actual_result == line:
        status = BetStatus.PUSH
        pnl = 0
    elif side == "over":
        if actual_result > line:
            status = BetStatus.WON
            # Calculate winnings from American odds
            if odds > 0:
                pnl = stake * (odds / 100)
            else:
                pnl = stake * (100 / abs(odds))
        else:
            status = BetStatus.LOST
            pnl = -stake
    else:  # under
        if actual_result < line:
            status = BetStatus.WON
            if odds > 0:
                pnl = stake * (odds / 100)
            else:
                pnl = stake * (100 / abs(odds))
        else:
            status = BetStatus.LOST
            pnl = -stake

    # Update bet
    cursor.execute("""
        UPDATE bets SET status = ?, actual_result = ?, pnl = ?
        WHERE id = ?
    """, (status.value, actual_result, pnl, bet_id))

    # Update bankroll (add back stake + pnl for wins, just stake for push)
    cursor.execute("SELECT balance FROM bankroll_history ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    current = row["balance"] if row else 10.0

    if status == BetStatus.WON:
        change = stake + pnl  # Return stake plus winnings
        reason = f"Bet #{bet_id} won (+{pnl:.2f})"
    elif status == BetStatus.PUSH:
        change = stake  # Return stake
        reason = f"Bet #{bet_id} pushed"
    else:
        change = 0  # Already subtracted stake when bet placed
        reason = f"Bet #{bet_id} lost"

    if change != 0:
        update_bankroll(current + change, change, reason, conn)

    conn.commit()
    conn.close()

    return {"status": status.value, "pnl": pnl}


def get_bet(bet_id: int) -> Bet | None:
    """Get a bet by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM bets WHERE id = ?", (bet_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return Bet(
        id=row["id"],
        created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
        game_date=row["game_date"],
        pitcher_name=row["pitcher_name"],
        prop_type=row["prop_type"],
        line=row["line"],
        side=BetSide(row["side"]),
        odds=row["odds"],
        stake=row["stake"],
        model_prediction=row["model_prediction"],
        model_edge=row["model_edge"],
        bookmaker=row["bookmaker"],
        status=BetStatus(row["status"]),
        actual_result=row["actual_result"],
        pnl=row["pnl"],
        home_team=row["home_team"],
        away_team=row["away_team"],
        is_auto=bool(row["is_auto"]) if "is_auto" in row.keys() else False,
    )


def get_bets(
    status: BetStatus | None = None,
    game_date: str | None = None,
    is_auto: bool | None = None,
    limit: int = 100,
) -> list[Bet]:
    """Get bets with optional filters."""
    conn = get_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM bets WHERE 1=1"
    params = []

    if status:
        query += " AND status = ?"
        params.append(status.value)
    if game_date:
        query += " AND game_date = ?"
        params.append(game_date)
    if is_auto is not None:
        query += " AND is_auto = ?"
        params.append(1 if is_auto else 0)

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return [
        Bet(
            id=row["id"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            game_date=row["game_date"],
            pitcher_name=row["pitcher_name"],
            prop_type=row["prop_type"],
            line=row["line"],
            side=BetSide(row["side"]),
            odds=row["odds"],
            stake=row["stake"],
            model_prediction=row["model_prediction"],
            model_edge=row["model_edge"],
            bookmaker=row["bookmaker"],
            status=BetStatus(row["status"]),
            actual_result=row["actual_result"],
            pnl=row["pnl"],
            home_team=row["home_team"],
            away_team=row["away_team"],
            is_auto=bool(row["is_auto"]) if "is_auto" in row.keys() else False,
        )
        for row in rows
    ]


def get_pending_bets() -> list[Bet]:
    """Get all pending bets."""
    return get_bets(status=BetStatus.PENDING)


def cancel_bet(bet_id: int) -> float:
    """Cancel a pending bet and refund the stake. Returns refunded amount."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM bets WHERE id = ?", (bet_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise ValueError(f"Bet {bet_id} not found")

    if row["status"] != "pending":
        conn.close()
        raise ValueError(f"Bet {bet_id} is not pending (status: {row['status']})")

    stake = row["stake"]

    # Update bet status
    cursor.execute("UPDATE bets SET status = ? WHERE id = ?", (BetStatus.CANCELLED.value, bet_id))

    # Refund stake
    cursor.execute("SELECT balance FROM bankroll_history ORDER BY id DESC LIMIT 1")
    current = cursor.fetchone()["balance"]
    update_bankroll(current + stake, stake, f"Bet #{bet_id} cancelled (refund)", conn)

    conn.commit()
    conn.close()

    return stake


def get_bankroll_history() -> list[dict]:
    """Get bankroll history for charting."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM bankroll_history ORDER BY id")
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "timestamp": row["timestamp"],
            "balance": row["balance"],
            "change_amount": row["change_amount"],
            "change_reason": row["change_reason"],
        }
        for row in rows
    ]


def get_stats() -> dict:
    """Get overall betting statistics."""
    conn = get_connection()
    cursor = conn.cursor()

    # Total bets
    cursor.execute("SELECT COUNT(*) as total FROM bets")
    total = cursor.fetchone()["total"]

    # Wins/losses
    cursor.execute("SELECT status, COUNT(*) as count FROM bets GROUP BY status")
    status_counts = {row["status"]: row["count"] for row in cursor.fetchall()}

    # Total P&L
    cursor.execute("SELECT SUM(pnl) as total_pnl FROM bets WHERE pnl IS NOT NULL")
    total_pnl = cursor.fetchone()["total_pnl"] or 0

    # Total staked
    cursor.execute("SELECT SUM(stake) as total_staked FROM bets")
    total_staked = cursor.fetchone()["total_staked"] or 0

    conn.close()

    starting = float(get_setting("starting_bankroll", "10.0"))
    current = get_current_bankroll()

    wins = status_counts.get("won", 0)
    losses = status_counts.get("lost", 0)
    settled = wins + losses + status_counts.get("push", 0)

    return {
        "total_bets": total,
        "pending": status_counts.get("pending", 0),
        "wins": wins,
        "losses": losses,
        "pushes": status_counts.get("push", 0),
        "win_rate": wins / settled if settled > 0 else 0,
        "total_pnl": total_pnl,
        "total_staked": total_staked,
        "roi": (total_pnl / total_staked * 100) if total_staked > 0 else 0,
        "starting_bankroll": starting,
        "current_bankroll": current,
        "bankroll_change": current - starting,
        "bankroll_change_pct": ((current - starting) / starting * 100) if starting > 0 else 0,
    }


def get_stats_by_type(is_auto: bool | None = None) -> dict:
    """Get betting statistics filtered by bet type (auto vs manual)."""
    conn = get_connection()
    cursor = conn.cursor()

    where_clause = "WHERE 1=1"
    params = []
    if is_auto is not None:
        where_clause += " AND is_auto = ?"
        params.append(1 if is_auto else 0)

    # Total bets
    cursor.execute(f"SELECT COUNT(*) as total FROM bets {where_clause}", params)
    total = cursor.fetchone()["total"]

    # Wins/losses
    cursor.execute(f"SELECT status, COUNT(*) as count FROM bets {where_clause} GROUP BY status", params)
    status_counts = {row["status"]: row["count"] for row in cursor.fetchall()}

    # Total P&L
    cursor.execute(f"SELECT SUM(pnl) as total_pnl FROM bets {where_clause} AND pnl IS NOT NULL", params)
    total_pnl = cursor.fetchone()["total_pnl"] or 0

    # Total staked
    cursor.execute(f"SELECT SUM(stake) as total_staked FROM bets {where_clause}", params)
    total_staked = cursor.fetchone()["total_staked"] or 0

    conn.close()

    wins = status_counts.get("won", 0)
    losses = status_counts.get("lost", 0)
    settled = wins + losses + status_counts.get("push", 0)

    return {
        "total_bets": total,
        "pending": status_counts.get("pending", 0),
        "wins": wins,
        "losses": losses,
        "pushes": status_counts.get("push", 0),
        "win_rate": wins / settled if settled > 0 else 0,
        "total_pnl": total_pnl,
        "total_staked": total_staked,
        "roi": (total_pnl / total_staked * 100) if total_staked > 0 else 0,
    }


def get_cumulative_pnl(is_auto: bool | None = None) -> list[dict]:
    """Get cumulative P&L over time for charting."""
    conn = get_connection()
    cursor = conn.cursor()

    where_clause = "WHERE status IN ('won', 'lost', 'push')"
    params = []
    if is_auto is not None:
        where_clause += " AND is_auto = ?"
        params.append(1 if is_auto else 0)

    cursor.execute(f"""
        SELECT game_date, pnl, pitcher_name, prop_type, side, line, status
        FROM bets
        {where_clause}
        ORDER BY game_date, created_at
    """, params)
    rows = cursor.fetchall()
    conn.close()

    cumulative = 0
    results = []
    for row in rows:
        pnl = row["pnl"] or 0
        cumulative += pnl
        results.append({
            "game_date": row["game_date"],
            "pnl": pnl,
            "cumulative_pnl": cumulative,
            "pitcher_name": row["pitcher_name"],
            "prop_type": row["prop_type"],
            "side": row["side"],
            "line": row["line"],
            "status": row["status"],
        })

    return results


def auto_bets_exist_for_date(game_date: str) -> bool:
    """Check if auto bets have already been placed for a date."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) as count FROM bets WHERE game_date = ? AND is_auto = 1",
        (game_date,)
    )
    count = cursor.fetchone()["count"]
    conn.close()
    return count > 0


# Initialize DB on import
init_db()
