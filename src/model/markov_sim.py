"""
Markov chain simulation for baseball innings.

Simulates innings using 24 base-out states and PA outcome probabilities.
"""

import numpy as np
from dataclasses import dataclass
from typing import Any


# Base state encoding (3 bits: 1st, 2nd, 3rd)
# 0 = empty, 1 = runner on that base
BASES_EMPTY = 0b000  # 0
RUNNER_1ST = 0b001   # 1
RUNNER_2ND = 0b010   # 2
RUNNER_3RD = 0b100   # 4
RUNNERS_12 = 0b011   # 3
RUNNERS_13 = 0b101   # 5
RUNNERS_23 = 0b110   # 6
BASES_LOADED = 0b111 # 7

# Outcome indices (alphabetical, matching OUTCOME_CLASSES)
OUTCOME_IDX = {
    "1B": 0,
    "2B": 1,
    "3B": 2,
    "BB": 3,
    "HR": 4,
    "K": 5,
    "OUT": 6,
}

OUTCOMES = ["1B", "2B", "3B", "BB", "HR", "K", "OUT"]


@dataclass
class InningState:
    """Current state of an inning."""
    outs: int = 0
    bases: int = 0  # 3-bit encoding
    runs: int = 0

    def __post_init__(self):
        self.outs = min(self.outs, 3)

    @property
    def runner_on_1st(self) -> bool:
        return bool(self.bases & 0b001)

    @property
    def runner_on_2nd(self) -> bool:
        return bool(self.bases & 0b010)

    @property
    def runner_on_3rd(self) -> bool:
        return bool(self.bases & 0b100)

    @property
    def is_over(self) -> bool:
        return self.outs >= 3

    def copy(self) -> "InningState":
        return InningState(outs=self.outs, bases=self.bases, runs=self.runs)


@dataclass
class GameState:
    """Accumulated game statistics."""
    innings_completed: int = 0
    partial_outs: int = 0  # Outs in current inning
    total_runs: int = 0
    total_hits: int = 0
    total_walks: int = 0
    total_strikeouts: int = 0
    total_home_runs: int = 0
    total_batters_faced: int = 0
    singles: int = 0
    doubles: int = 0
    triples: int = 0

    @property
    def ip(self) -> float:
        """Innings pitched as decimal (e.g., 5.2 = 5 2/3 innings)."""
        return self.innings_completed + self.partial_outs / 3

    @property
    def ip_display(self) -> str:
        """Innings pitched display format (e.g., '5.2')."""
        return f"{self.innings_completed}.{self.partial_outs}"


def apply_outcome(
    state: InningState,
    outcome: str,
    rng: np.random.Generator | None = None,
) -> InningState:
    """
    Apply a PA outcome to the current inning state.

    Returns new state with updated outs, bases, and runs.

    Uses probabilistic runner advancement based on MLB averages:
    - Runner on 2nd scores on single: ~60%
    - Runner on 1st scores on double: ~45%
    - Runner on 1st to 3rd on single: ~28%
    """
    new_state = state.copy()

    if outcome == "K" or outcome == "OUT":
        new_state.outs += 1
        return new_state

    if outcome == "HR":
        # Count runners + batter
        runners = bin(state.bases).count('1')
        new_state.runs += runners + 1
        new_state.bases = BASES_EMPTY
        return new_state

    if outcome == "3B":
        # All runners score, batter to 3rd
        runners = bin(state.bases).count('1')
        new_state.runs += runners
        new_state.bases = RUNNER_3RD
        return new_state

    if outcome == "2B":
        # Batter to 2nd
        # Runners on 2nd and 3rd score
        # Runner on 1st: scores ~45%, else to 3rd
        runs = 0
        if state.runner_on_3rd:
            runs += 1
        if state.runner_on_2nd:
            runs += 1

        new_bases = RUNNER_2ND  # Batter on 2nd
        if state.runner_on_1st:
            # 45% chance to score, else to 3rd
            if rng is not None and rng.random() < 0.45:
                runs += 1
            else:
                new_bases |= RUNNER_3RD

        new_state.runs += runs
        new_state.bases = new_bases
        return new_state

    if outcome == "1B":
        # Batter to 1st
        # Runner on 3rd scores
        # Runner on 2nd: scores ~60%, else to 3rd
        # Runner on 1st: to 3rd ~28%, else to 2nd
        runs = 0
        new_bases = RUNNER_1ST  # Batter on 1st

        if state.runner_on_3rd:
            runs += 1

        if state.runner_on_2nd:
            # 60% chance to score, else to 3rd
            if rng is not None and rng.random() < 0.60:
                runs += 1
            else:
                new_bases |= RUNNER_3RD

        if state.runner_on_1st:
            # 28% chance to reach 3rd, else to 2nd
            if rng is not None and rng.random() < 0.28:
                new_bases |= RUNNER_3RD
            else:
                new_bases |= RUNNER_2ND

        new_state.runs += runs
        new_state.bases = new_bases
        return new_state

    if outcome == "BB":
        # Forced advancement only
        runs = 0

        if state.bases == BASES_LOADED:
            runs = 1
            new_bases = BASES_LOADED
        elif state.bases == RUNNERS_23:
            # 2nd and 3rd occupied, batter to 1st, everyone stays
            new_bases = BASES_LOADED
        elif state.bases == RUNNERS_12:
            # 1st and 2nd, force to 2nd and 3rd
            new_bases = BASES_LOADED
        elif state.bases == RUNNERS_13:
            # 1st and 3rd, force runner to 2nd
            new_bases = BASES_LOADED
        elif state.bases == RUNNER_1ST:
            new_bases = RUNNERS_12
        elif state.bases == RUNNER_2ND:
            new_bases = RUNNERS_12
        elif state.bases == RUNNER_3RD:
            new_bases = RUNNERS_13
        else:  # Empty
            new_bases = RUNNER_1ST

        new_state.runs += runs
        new_state.bases = new_bases
        return new_state

    return new_state


def simulate_pa(proba: np.ndarray, rng: np.random.Generator) -> str:
    """
    Simulate a single PA outcome given probabilities.

    Args:
        proba: Array of probabilities for each outcome (length 7)
        rng: Random number generator

    Returns:
        Outcome string ("K", "BB", "1B", etc.)
    """
    # Ensure probabilities sum to 1
    proba = proba / proba.sum()
    outcome_idx = rng.choice(len(OUTCOMES), p=proba)
    return OUTCOMES[outcome_idx]


def simulate_inning(
    lineup_proba: list[np.ndarray],
    lineup_idx: int,
    rng: np.random.Generator,
) -> tuple[InningState, int, dict]:
    """
    Simulate a single inning.

    Args:
        lineup_proba: List of probability arrays for each batter in lineup
        lineup_idx: Current position in lineup (0-8)
        rng: Random number generator

    Returns:
        Tuple of (final inning state, new lineup index, outcome counts)
    """
    state = InningState()
    outcomes = {o: 0 for o in OUTCOMES}
    batters_faced = 0

    while not state.is_over:
        # Get current batter's probabilities
        batter_proba = lineup_proba[lineup_idx % 9]

        # Simulate outcome
        outcome = simulate_pa(batter_proba, rng)
        outcomes[outcome] += 1
        batters_faced += 1

        # Apply to state (pass rng for probabilistic advancement)
        state = apply_outcome(state, outcome, rng)

        # Next batter
        lineup_idx += 1

        # Safety check (prevent infinite loop)
        if batters_faced > 50:
            state.outs = 3
            break

    return state, lineup_idx, outcomes


def simulate_game(
    lineup_proba: list[np.ndarray],
    target_innings: float = 6.0,
    seed: int | None = None,
) -> GameState:
    """
    Simulate a full game (starter's portion).

    Args:
        lineup_proba: List of 9 probability arrays, one per lineup spot
        target_innings: Target innings to simulate (default 6.0)
        seed: Random seed for reproducibility

    Returns:
        GameState with accumulated statistics
    """
    rng = np.random.default_rng(seed)
    game = GameState()
    lineup_idx = 0

    full_innings = int(target_innings)
    partial_target = target_innings - full_innings

    # Simulate full innings
    for _ in range(full_innings):
        inning_state, lineup_idx, outcomes = simulate_inning(
            lineup_proba, lineup_idx, rng
        )

        game.innings_completed += 1
        game.total_runs += inning_state.runs
        game.total_strikeouts += outcomes["K"]
        game.total_walks += outcomes["BB"]
        game.total_home_runs += outcomes["HR"]
        game.singles += outcomes["1B"]
        game.doubles += outcomes["2B"]
        game.triples += outcomes["3B"]
        game.total_hits += outcomes["1B"] + outcomes["2B"] + outcomes["3B"] + outcomes["HR"]
        game.total_batters_faced += sum(outcomes.values())

    # Simulate partial inning if needed
    if partial_target > 0:
        target_outs = round(partial_target * 3)
        state = InningState()
        prev_runs = 0  # Track runs to get delta each PA

        while state.outs < target_outs and not state.is_over:
            batter_proba = lineup_proba[lineup_idx % 9]
            outcome = simulate_pa(batter_proba, rng)

            game.total_batters_faced += 1
            if outcome == "K":
                game.total_strikeouts += 1
            elif outcome == "BB":
                game.total_walks += 1
            elif outcome == "HR":
                game.total_home_runs += 1
                game.total_hits += 1
            elif outcome == "1B":
                game.singles += 1
                game.total_hits += 1
            elif outcome == "2B":
                game.doubles += 1
                game.total_hits += 1
            elif outcome == "3B":
                game.triples += 1
                game.total_hits += 1

            state = apply_outcome(state, outcome, rng)
            lineup_idx += 1

        game.partial_outs = state.outs
        game.total_runs += state.runs  # Add runs from partial inning once

    return game


def expected_game_stats(
    lineup_proba: list[np.ndarray],
    target_innings: float = 6.0,
    n_simulations: int = 1,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Get expected game statistics via simulation.

    Args:
        lineup_proba: List of 9 probability arrays, one per lineup spot
        target_innings: Target innings to simulate
        n_simulations: Number of simulations to average (1 = single sim)
        seed: Random seed

    Returns:
        Dictionary with expected statistics
    """
    rng = np.random.default_rng(seed)

    totals = {
        "runs": 0,
        "hits": 0,
        "walks": 0,
        "strikeouts": 0,
        "home_runs": 0,
        "batters_faced": 0,
        "singles": 0,
        "doubles": 0,
        "triples": 0,
        "ip": 0.0,
    }

    for i in range(n_simulations):
        sim_seed = rng.integers(0, 2**31) if seed is not None else None
        game = simulate_game(lineup_proba, target_innings, sim_seed)

        totals["runs"] += game.total_runs
        totals["hits"] += game.total_hits
        totals["walks"] += game.total_walks
        totals["strikeouts"] += game.total_strikeouts
        totals["home_runs"] += game.total_home_runs
        totals["batters_faced"] += game.total_batters_faced
        totals["singles"] += game.singles
        totals["doubles"] += game.doubles
        totals["triples"] += game.triples
        totals["ip"] += game.ip

    # Average over simulations
    result = {k: v / n_simulations for k, v in totals.items()}

    # Calculate ERA
    if result["ip"] > 0:
        result["era"] = result["runs"] / result["ip"] * 9
    else:
        result["era"] = None

    return result
