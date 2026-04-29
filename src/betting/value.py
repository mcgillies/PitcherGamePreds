"""
Value detection - compare model predictions to betting odds.
"""

from dataclasses import dataclass
from typing import Any

from .espn_odds import american_to_implied_prob, american_to_decimal


@dataclass
class ValueBet:
    """A potential value betting opportunity."""
    pitcher_name: str
    prop_type: str
    line: float
    model_prediction: float
    edge: float  # Difference between model and line
    side: str  # "over" or "under"
    odds: int
    implied_prob: float
    model_prob: float  # Estimated probability of winning
    expected_value: float  # EV of the bet
    bookmaker: str
    home_team: str
    away_team: str

    def to_dict(self) -> dict:
        return {
            "pitcher_name": self.pitcher_name,
            "prop_type": self.prop_type,
            "line": self.line,
            "model_prediction": self.model_prediction,
            "edge": self.edge,
            "side": self.side,
            "odds": self.odds,
            "implied_prob": self.implied_prob,
            "model_prob": self.model_prob,
            "expected_value": self.expected_value,
            "bookmaker": self.bookmaker,
            "home_team": self.home_team,
            "away_team": self.away_team,
        }


def estimate_over_probability(prediction: float, line: float, std_dev: float = 1.5) -> float:
    """
    Estimate probability of going over a line given our prediction.

    Uses normal distribution approximation.
    For K: std_dev ~1.5-2.0
    For H: std_dev ~1.5
    For ER: std_dev ~1.5

    Args:
        prediction: Model's prediction (e.g., 6.5 Ks)
        line: Betting line (e.g., 5.5)
        std_dev: Estimated standard deviation

    Returns:
        Probability of going over the line
    """
    from scipy import stats

    # Z-score: how many std devs is the line from our prediction?
    # If prediction > line, z is negative, meaning more likely to go over
    z = (line - prediction) / std_dev

    # P(X > line) = 1 - P(X <= line) = 1 - CDF(z)
    prob_over = 1 - stats.norm.cdf(z)

    return prob_over


def calculate_expected_value(win_prob: float, american_odds: int) -> float:
    """
    Calculate expected value of a bet.

    EV = (win_prob * profit) - (lose_prob * stake)

    Assuming $1 stake:
    - If we win: profit = decimal_odds - 1
    - If we lose: lose stake = 1
    """
    decimal_odds = american_to_decimal(american_odds)
    profit_if_win = decimal_odds - 1
    lose_prob = 1 - win_prob

    ev = (win_prob * profit_if_win) - (lose_prob * 1)
    return ev


def find_value_bets(
    predictions: list[dict],
    props: list[dict],
    min_edge: float = 0.5,
    min_ev: float = 0.0,
    std_devs: dict[str, float] | None = None,
) -> list[ValueBet]:
    """
    Find value betting opportunities.

    Args:
        predictions: List of model predictions from GamePredictorBinary
        props: List of prop odds from odds.py
        min_edge: Minimum edge (prediction - line) to consider
        min_ev: Minimum expected value to consider
        std_devs: Standard deviations by prop type

    Returns:
        List of ValueBet opportunities sorted by EV
    """
    if std_devs is None:
        std_devs = {
            "strikeouts": 1.8,
            "hits_allowed": 1.5,
            "earned_runs": 1.5,
        }

    # Map predictions to pitcher names
    pred_by_pitcher = {}
    for game_pred in predictions:
        for key in ["away_prediction", "home_prediction"]:
            pred = game_pred.get(key)
            if pred:
                name = pred["pitcher_name"]
                stats = pred["expected_stats"]
                pred_by_pitcher[name.lower()] = {
                    "K": stats.get("K", 0),
                    "H": stats.get("H", 0),
                    "ER": stats.get("ER", 0),
                    "home_team": game_pred.get("home_team"),
                    "away_team": game_pred.get("away_team"),
                }

    # Prop type mapping
    prop_to_stat = {
        "strikeouts": "K",
        "hits_allowed": "H",
        "earned_runs": "ER",
    }

    value_bets = []

    for prop in props:
        pitcher_name = prop["pitcher_name"].lower()
        prop_type = prop["prop_type"]
        line = prop["line"]

        # Find matching prediction
        pred_data = pred_by_pitcher.get(pitcher_name)
        if not pred_data:
            # Try partial match
            for name, data in pred_by_pitcher.items():
                if pitcher_name in name or name in pitcher_name:
                    pred_data = data
                    break

        if not pred_data:
            continue

        stat_key = prop_to_stat.get(prop_type)
        if not stat_key:
            continue

        model_pred = pred_data.get(stat_key, 0)
        std_dev = std_devs.get(prop_type, 1.5)

        # Calculate edge and probabilities
        edge = model_pred - line

        # Check both over and under
        for side in ["over", "under"]:
            odds = prop.get(f"{side}_odds")
            if odds is None:
                continue

            implied_prob = american_to_implied_prob(odds)

            if side == "over":
                model_prob = estimate_over_probability(model_pred, line, std_dev)
                side_edge = edge  # Positive edge means over is good
            else:
                model_prob = 1 - estimate_over_probability(model_pred, line, std_dev)
                side_edge = -edge  # Negative edge means under is good

            ev = calculate_expected_value(model_prob, odds)

            # Filter by criteria
            if abs(side_edge) < min_edge:
                continue
            if ev < min_ev:
                continue

            # Only recommend if our probability exceeds implied
            if model_prob <= implied_prob:
                continue

            value_bets.append(ValueBet(
                pitcher_name=prop["pitcher_name"],
                prop_type=prop_type,
                line=line,
                model_prediction=model_pred,
                edge=side_edge,
                side=side,
                odds=odds,
                implied_prob=implied_prob,
                model_prob=model_prob,
                expected_value=ev,
                bookmaker=prop["bookmaker"],
                home_team=prop.get("home_team") or pred_data.get("home_team", ""),
                away_team=prop.get("away_team") or pred_data.get("away_team", ""),
            ))

    # Sort by expected value descending
    value_bets.sort(key=lambda x: x.expected_value, reverse=True)

    return value_bets


def format_value_bet(vb: ValueBet) -> str:
    """Format a value bet for display."""
    odds_str = f"+{vb.odds}" if vb.odds > 0 else str(vb.odds)

    return (
        f"{vb.pitcher_name} {vb.prop_type.upper()} {vb.side.upper()} {vb.line} ({odds_str})\n"
        f"  Model: {vb.model_prediction:.1f} | Edge: {vb.edge:+.1f}\n"
        f"  Model prob: {vb.model_prob:.1%} vs Implied: {vb.implied_prob:.1%}\n"
        f"  EV: {vb.expected_value:+.3f} | Book: {vb.bookmaker}"
    )
