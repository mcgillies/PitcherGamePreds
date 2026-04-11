"""
Game Simulator

Orchestrates Monte Carlo simulation of bullpen usage from SP exit through game end.

Combines:
- Transition model (which role enters)
- Pitcher selection (which specific pitcher)
- Reliever exit model (when they leave)
- At-bat simulation (runs/outs/etc via GamePredictorBinary)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import date

from .pitcher_roles import ROLE_STARTER, RELIEF_ROLES
from .transition_model import BullpenTransitionModel, GameState
from .reliever_exit import RelieverExitModel, RelieverState
from .pitcher_selection import PitcherSelectionModel


@dataclass
class SimulationState:
    """Tracks current state during simulation."""
    inning: int
    outs_in_inning: int  # 0, 1, 2
    score_diff: int  # From pitching team's perspective (+ = winning)
    runners_on: int  # 0, 1, 2, 3 (simplified)
    current_pitcher_id: Optional[int] = None
    current_pitcher_role: str = ROLE_STARTER
    outs_by_current_pitcher: int = 0
    lineup_position: int = 0  # 0-8, current batter in lineup
    total_outs: int = 0  # Total outs recorded in game

    @property
    def is_game_over(self) -> bool:
        """Check if game should end (simplified: 9 innings)."""
        # Game ends after 9 innings (27 outs) if pitching team is winning
        # or after 9+ innings if tied, then someone takes lead
        return self.total_outs >= 27 and self.score_diff > 0

    def to_game_state(self) -> GameState:
        """Convert to GameState for transition model."""
        return GameState(
            inning=self.inning,
            score_diff=self.score_diff,
            outs=self.outs_in_inning,
            prev_role=self.current_pitcher_role,
        )

    def to_reliever_state(self) -> RelieverState:
        """Convert to RelieverState for exit model."""
        return RelieverState(
            outs_recorded=self.outs_by_current_pitcher,
            role=self.current_pitcher_role,
            inning=self.inning,
            score_diff=self.score_diff,
            runners_on=self.runners_on,
        )


@dataclass
class RelieverAppearance:
    """Stats for a single reliever appearance in simulation."""
    pitcher_id: int
    role: str
    outs: int = 0
    runs: int = 0
    hits: int = 0
    walks: int = 0
    strikeouts: int = 0
    entry_inning: int = 0
    entry_score_diff: int = 0


@dataclass
class SimulationResult:
    """Result of a single game simulation."""
    final_score_diff: int  # Final score differential
    total_runs_allowed: int  # Runs allowed by bullpen
    reliever_appearances: List[RelieverAppearance] = field(default_factory=list)
    innings_simulated: float = 0.0


@dataclass
class AggregatedResults:
    """Aggregated results across many simulations."""
    n_simulations: int
    mean_runs_allowed: float
    std_runs_allowed: float
    run_distribution: Dict[int, float]  # P(runs = k)

    # Reliever usage summary
    # pitcher_id -> {appearances: int, mean_outs: float, mean_runs: float, ...}
    reliever_usage: Dict[int, Dict[str, float]] = field(default_factory=dict)

    # Win probability (pitching team ahead at end)
    win_prob: float = 0.0


class GameSimulator:
    """
    Monte Carlo game simulator for bullpen phase.

    Usage:
        simulator = GameSimulator(transition_model, exit_model, selection_model)
        results = simulator.simulate(
            entry_state=state_when_sp_exits,
            team="NYY",
            lineup_proba=[...],  # 9 batters' outcome probabilities
            n_simulations=1000,
        )
    """

    # Outcome indices (matching markov_sim.OUTCOMES)
    OUT_1B = 0
    OUT_2B = 1
    OUT_3B = 2
    OUT_BB = 3
    OUT_HR = 4
    OUT_K = 5
    OUT_OUT = 6

    def __init__(
        self,
        transition_model: BullpenTransitionModel,
        exit_model: RelieverExitModel,
        selection_model: PitcherSelectionModel,
        max_innings: int = 15,
    ):
        """
        Initialize simulator.

        Args:
            transition_model: Model for P(next_role | state)
            exit_model: Model for P(exit | reliever_state)
            selection_model: Model for P(pitcher | role, team, availability)
            max_innings: Maximum innings to simulate (prevents infinite loops)
        """
        self.transition_model = transition_model
        self.exit_model = exit_model
        self.selection_model = selection_model
        self.max_innings = max_innings

    def simulate(
        self,
        entry_state: SimulationState,
        team: str,
        lineup_proba: List[np.ndarray],
        n_simulations: int = 1000,
        seed: Optional[int] = None,
    ) -> AggregatedResults:
        """
        Run Monte Carlo simulation.

        Args:
            entry_state: Game state when SP exits
            team: Pitching team code
            lineup_proba: List of 9 probability arrays (one per lineup spot)
                         Each array is [P(1B), P(2B), P(3B), P(BB), P(HR), P(K), P(OUT)]
            n_simulations: Number of simulations to run
            seed: Random seed for reproducibility

        Returns:
            Aggregated simulation results
        """
        if seed is not None:
            np.random.seed(seed)

        results: List[SimulationResult] = []

        for sim_idx in range(n_simulations):
            result = self._simulate_single(entry_state, team, lineup_proba)
            results.append(result)

        return self._aggregate_results(results)

    def _simulate_single(
        self,
        entry_state: SimulationState,
        team: str,
        lineup_proba: List[np.ndarray],
    ) -> SimulationResult:
        """Run a single simulation."""
        state = SimulationState(
            inning=entry_state.inning,
            outs_in_inning=entry_state.outs_in_inning,
            score_diff=entry_state.score_diff,
            runners_on=entry_state.runners_on,
            lineup_position=entry_state.lineup_position,
            total_outs=entry_state.total_outs,
            current_pitcher_role=entry_state.current_pitcher_role,
        )

        appearances: List[RelieverAppearance] = []
        pitchers_used: Set[int] = set()
        current_appearance: Optional[RelieverAppearance] = None
        total_runs = 0

        # Main simulation loop
        while not self._should_end_game(state):
            # Do we need a new pitcher?
            if current_appearance is None or self._should_change_pitcher(state, current_appearance):
                # Save current appearance if exists
                if current_appearance is not None:
                    appearances.append(current_appearance)
                    pitchers_used.add(current_appearance.pitcher_id)

                # Select new pitcher
                new_role = self.transition_model.sample_next_role(state.to_game_state())
                new_pitcher = self.selection_model.select_pitcher(
                    role=new_role,
                    team=team,
                    already_pitched=pitchers_used,
                )

                if new_pitcher is None:
                    # No one available - end simulation
                    break

                current_appearance = RelieverAppearance(
                    pitcher_id=new_pitcher,
                    role=new_role,
                    entry_inning=state.inning,
                    entry_score_diff=state.score_diff,
                )
                state.current_pitcher_id = new_pitcher
                state.current_pitcher_role = new_role
                state.outs_by_current_pitcher = 0

            # Simulate at-bat
            batter_idx = state.lineup_position % 9
            outcome = self._sample_outcome(lineup_proba[batter_idx])

            # Update state based on outcome
            runs_scored = self._process_outcome(state, outcome)
            total_runs += runs_scored

            # Update appearance stats
            if current_appearance:
                if outcome == self.OUT_K:
                    current_appearance.strikeouts += 1
                    current_appearance.outs += 1
                elif outcome == self.OUT_OUT:
                    current_appearance.outs += 1
                elif outcome == self.OUT_BB:
                    current_appearance.walks += 1
                elif outcome in (self.OUT_1B, self.OUT_2B, self.OUT_3B, self.OUT_HR):
                    current_appearance.hits += 1
                    if outcome == self.OUT_HR:
                        current_appearance.runs += runs_scored

                current_appearance.runs += runs_scored

            # Next batter
            state.lineup_position = (state.lineup_position + 1) % 9

        # Save final appearance
        if current_appearance is not None:
            appearances.append(current_appearance)

        innings_simulated = (state.total_outs - entry_state.total_outs) / 3

        return SimulationResult(
            final_score_diff=state.score_diff,
            total_runs_allowed=total_runs,
            reliever_appearances=appearances,
            innings_simulated=innings_simulated,
        )

    def _should_end_game(self, state: SimulationState) -> bool:
        """Check if game should end."""
        # End after 9 innings (27 outs) if pitching team winning
        if state.total_outs >= 27 and state.score_diff > 0:
            return True

        # End if we've hit max innings
        if state.inning > self.max_innings:
            return True

        return False

    def _should_change_pitcher(
        self,
        state: SimulationState,
        current: RelieverAppearance,
    ) -> bool:
        """Check if current pitcher should be replaced."""
        reliever_state = state.to_reliever_state()
        return self.exit_model.sample_exit(reliever_state)

    def _sample_outcome(self, proba: np.ndarray) -> int:
        """Sample an at-bat outcome."""
        # Ensure probabilities sum to 1
        proba = proba / proba.sum()
        return np.random.choice(7, p=proba)

    def _process_outcome(self, state: SimulationState, outcome: int) -> int:
        """
        Process at-bat outcome and update state.

        Returns runs scored on this play.
        """
        runs = 0

        if outcome == self.OUT_K or outcome == self.OUT_OUT:
            # Out recorded
            state.outs_in_inning += 1
            state.outs_by_current_pitcher += 1
            state.total_outs += 1

            # Check for inning end
            if state.outs_in_inning >= 3:
                state.inning += 1
                state.outs_in_inning = 0
                state.runners_on = 0

        elif outcome == self.OUT_BB or outcome == self.OUT_1B:
            # Runner on, possible run if bases loaded
            if state.runners_on == 3:
                runs = 1
            else:
                state.runners_on = min(3, state.runners_on + 1)

        elif outcome == self.OUT_2B:
            # Double - score some runners
            if state.runners_on >= 2:
                runs = state.runners_on - 1
                state.runners_on = 1
            else:
                state.runners_on = min(3, state.runners_on + 1)

        elif outcome == self.OUT_3B:
            # Triple - score all runners
            runs = state.runners_on
            state.runners_on = 0  # Runner on 3rd (simplified as 0)

        elif outcome == self.OUT_HR:
            # Home run - everyone scores
            runs = state.runners_on + 1
            state.runners_on = 0

        # Update score differential (runs against pitching team)
        state.score_diff -= runs

        return runs

    def _aggregate_results(self, results: List[SimulationResult]) -> AggregatedResults:
        """Aggregate results across simulations."""
        n = len(results)

        runs_allowed = [r.total_runs_allowed for r in results]
        mean_runs = np.mean(runs_allowed)
        std_runs = np.std(runs_allowed)

        # Run distribution
        from collections import Counter
        run_counts = Counter(runs_allowed)
        run_dist = {k: v / n for k, v in run_counts.items()}

        # Win probability
        wins = sum(1 for r in results if r.final_score_diff > 0)
        win_prob = wins / n

        # Reliever usage
        reliever_stats: Dict[int, Dict[str, List]] = {}

        for result in results:
            for app in result.reliever_appearances:
                pid = app.pitcher_id
                if pid not in reliever_stats:
                    reliever_stats[pid] = {
                        'appearances': 0,
                        'outs': [],
                        'runs': [],
                        'strikeouts': [],
                        'role': app.role,
                    }

                reliever_stats[pid]['appearances'] += 1
                reliever_stats[pid]['outs'].append(app.outs)
                reliever_stats[pid]['runs'].append(app.runs)
                reliever_stats[pid]['strikeouts'].append(app.strikeouts)

        # Aggregate reliever stats
        reliever_usage = {}
        for pid, stats in reliever_stats.items():
            reliever_usage[pid] = {
                'role': stats['role'],
                'appearance_rate': stats['appearances'] / n,
                'mean_outs': np.mean(stats['outs']) if stats['outs'] else 0,
                'mean_runs': np.mean(stats['runs']) if stats['runs'] else 0,
                'mean_strikeouts': np.mean(stats['strikeouts']) if stats['strikeouts'] else 0,
            }

        return AggregatedResults(
            n_simulations=n,
            mean_runs_allowed=mean_runs,
            std_runs_allowed=std_runs,
            run_distribution=run_dist,
            reliever_usage=reliever_usage,
            win_prob=win_prob,
        )


def create_simulator_from_data(
    pitches_df: pd.DataFrame,
    game_date: date,
    lookback_days: int = 30,
    method: str = "empirical",
) -> GameSimulator:
    """
    Convenience function to create a fully fitted simulator.

    Args:
        pitches_df: Pitch data
        game_date: Current date
        lookback_days: Days to look back for fitting
        method: Transition model method ("empirical" or "classifier")

    Returns:
        Fitted GameSimulator
    """
    from .extract_transitions import extract_transitions, extract_pitcher_usage_stats
    from .pitcher_roles import PitcherRoleClassifier

    # Filter to lookback window
    cutoff = game_date - pd.Timedelta(days=lookback_days)
    recent = pitches_df[pd.to_datetime(pitches_df['game_date']) > cutoff]

    print(f"Building simulator from {len(recent):,} pitches (last {lookback_days} days)")

    # Classify roles
    pitcher_stats = extract_pitcher_usage_stats(recent)
    classifier = PitcherRoleClassifier()
    classified = classifier.fit_transform(pitcher_stats)
    pitcher_roles = dict(zip(classified['pitcher'], classified['role']))

    # Fit transition model
    transitions = extract_transitions(recent)
    transition_model = BullpenTransitionModel(method=method)
    transition_model.fit(transitions, pitcher_roles)

    # Fit exit model
    exit_model = RelieverExitModel()
    exit_model.fit(recent, pitcher_roles)

    # Fit selection model
    selection_model = PitcherSelectionModel()
    selection_model.build_from_pitches(recent, pitcher_roles, game_date)

    return GameSimulator(transition_model, exit_model, selection_model)


if __name__ == "__main__":
    # Quick test
    from datetime import date

    print("Loading pitch data...")
    pitches = pd.read_parquet("data/raw/pitches.parquet")

    game_date = date(2024, 9, 15)
    simulator = create_simulator_from_data(pitches, game_date)

    print("\nSimulator created successfully!")
    print(f"Teams: {list(simulator.selection_model.team_bullpens.keys())[:5]}...")
