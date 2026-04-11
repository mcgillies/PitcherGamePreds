"""
Reliever Exit Model

Models P(exit | outs_recorded, role, game_state) - the probability that a reliever
exits after the current batter given how many outs they've recorded and game context.

Key factors:
- Role: Closers ~3 outs, Setup ~3 outs, Long Relief ~6+ outs, Specialists ~1-2 outs
- Outs recorded: More outs = higher exit probability
- Game state: Close games = shorter stints, blowouts = longer stints
- Inning: Later innings = more careful management
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import pickle

from .pitcher_roles import (
    ROLE_STARTER, ROLE_CLOSER, ROLE_SETUP, ROLE_MIDDLE, ROLE_LONG, ROLE_SPECIALIST,
    RELIEF_ROLES
)


@dataclass
class RelieverState:
    """State for predicting reliever exit."""
    outs_recorded: int  # Outs this reliever has gotten
    role: str
    inning: int
    score_diff: int  # From pitching team's perspective
    runners_on: int = 0  # 0, 1, 2, or 3

    @property
    def is_save_situation(self) -> bool:
        return self.inning >= 9 and 1 <= self.score_diff <= 3

    @property
    def is_blowout(self) -> bool:
        return abs(self.score_diff) >= 5

    @property
    def is_close_game(self) -> bool:
        return abs(self.score_diff) <= 2


class RelieverExitModel:
    """
    Models when relievers exit based on historical patterns.

    Uses empirical distributions of outs per appearance by role and game context.
    """

    def __init__(self):
        # Exit probability by (role, outs_recorded)
        # These are baseline probabilities, adjusted by game context
        self.base_exit_probs: Dict[Tuple[str, int], float] = {}

        # Distribution of outs per appearance by role
        self.outs_distribution: Dict[str, Dict[int, float]] = {}

        self.fitted = False

    def fit(self, pitches_df: pd.DataFrame, pitcher_roles: Dict[int, str]) -> "RelieverExitModel":
        """
        Fit exit model from historical pitch data.

        Args:
            pitches_df: Pitch-level data
            pitcher_roles: Mapping of pitcher_id -> role

        Returns:
            Self for chaining
        """
        print("Fitting reliever exit model...")

        df = pitches_df.copy()

        # Get reliever appearances (exclude starters)
        # Group by game and pitcher to get appearance-level stats
        appearances = df.groupby(['game_pk', 'pitcher']).agg(
            first_inning=('inning', 'min'),
            last_inning=('inning', 'max'),
            total_outs=('events', lambda x: x.isin([
                'strikeout', 'field_out', 'grounded_into_double_play',
                'force_out', 'sac_fly', 'sac_bunt', 'double_play',
                'fielders_choice_out', 'strikeout_double_play',
                'triple_play', 'fielders_choice'
            ]).sum()),
        ).reset_index()

        # Add role
        appearances['role'] = appearances['pitcher'].map(pitcher_roles)

        # Filter to relievers only
        appearances = appearances[appearances['role'].isin(RELIEF_ROLES)]

        print(f"  Analyzing {len(appearances):,} reliever appearances")

        if len(appearances) == 0:
            print("  WARNING: No reliever appearances found")
            self._set_defaults()
            self.fitted = True
            return self

        # Compute outs distribution by role
        for role in RELIEF_ROLES:
            role_apps = appearances[appearances['role'] == role]
            if len(role_apps) == 0:
                continue

            outs_counts = role_apps['total_outs'].value_counts(normalize=True)
            self.outs_distribution[role] = outs_counts.to_dict()

        # Compute exit probabilities by (role, outs_recorded)
        # P(exit after out k) = P(total_outs = k) / P(total_outs >= k)
        for role in RELIEF_ROLES:
            if role not in self.outs_distribution:
                continue

            dist = self.outs_distribution[role]

            # Compute survival function and hazard rate
            for outs in range(0, 15):  # Up to 5 innings
                # P(total_outs >= outs)
                prob_at_least = sum(p for k, p in dist.items() if k >= outs)

                # P(total_outs = outs)
                prob_exactly = dist.get(outs, 0)

                if prob_at_least > 0:
                    # Hazard rate: P(exit at outs | survived to outs)
                    exit_prob = prob_exactly / prob_at_least
                else:
                    exit_prob = 1.0  # Exit if we've exceeded typical usage

                self.base_exit_probs[(role, outs)] = exit_prob

        # Print summary
        print("\n  Average outs per appearance by role:")
        for role in RELIEF_ROLES:
            role_apps = appearances[appearances['role'] == role]
            if len(role_apps) > 0:
                avg_outs = role_apps['total_outs'].mean()
                print(f"    {role}: {avg_outs:.1f} outs ({len(role_apps)} appearances)")

        self.fitted = True
        return self

    def _set_defaults(self):
        """Set default exit probabilities if no data."""
        # Default outs distributions by role
        defaults = {
            ROLE_CLOSER: {3: 0.7, 4: 0.15, 2: 0.1, 5: 0.05},
            ROLE_SETUP: {3: 0.6, 4: 0.2, 2: 0.15, 5: 0.05},
            ROLE_MIDDLE: {3: 0.5, 4: 0.2, 5: 0.15, 6: 0.1, 2: 0.05},
            ROLE_LONG: {6: 0.3, 5: 0.25, 4: 0.2, 7: 0.15, 8: 0.1},
            ROLE_SPECIALIST: {1: 0.3, 2: 0.3, 3: 0.3, 4: 0.1},
        }

        self.outs_distribution = defaults

        # Compute exit probs from defaults
        for role, dist in defaults.items():
            for outs in range(0, 12):
                prob_at_least = sum(p for k, p in dist.items() if k >= outs)
                prob_exactly = dist.get(outs, 0)
                if prob_at_least > 0:
                    self.base_exit_probs[(role, outs)] = prob_exactly / prob_at_least
                else:
                    self.base_exit_probs[(role, outs)] = 1.0

    def predict_exit_prob(self, state: RelieverState) -> float:
        """
        Predict probability that reliever exits after current state.

        Args:
            state: Current reliever state

        Returns:
            Probability of exit (0 to 1)
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get base probability for this role and outs
        base_prob = self.base_exit_probs.get(
            (state.role, state.outs_recorded),
            0.5  # Default if not seen
        )

        # Adjust for game context
        adjustment = 1.0

        # Close games: managers pull relievers sooner
        if state.is_close_game and state.inning >= 7:
            adjustment *= 1.2

        # Blowouts: let them pitch longer
        if state.is_blowout:
            adjustment *= 0.7

        # Save situation with closer: let them finish
        if state.is_save_situation and state.role == ROLE_CLOSER:
            adjustment *= 0.8

        # Runners on: less likely to pull mid-situation
        if state.runners_on > 0:
            adjustment *= 0.7

        # Late innings (9th+): more careful management
        if state.inning >= 9 and state.role != ROLE_CLOSER:
            adjustment *= 1.3

        # Apply adjustment and clamp
        adjusted_prob = base_prob * adjustment
        return max(0.0, min(1.0, adjusted_prob))

    def sample_exit(self, state: RelieverState) -> bool:
        """
        Sample whether reliever exits.

        Returns:
            True if reliever exits, False if continues
        """
        prob = self.predict_exit_prob(state)
        return np.random.random() < prob

    def expected_outs(self, role: str) -> float:
        """Get expected outs for a role."""
        if role not in self.outs_distribution:
            return 3.0  # Default to 1 inning

        dist = self.outs_distribution[role]
        return sum(outs * prob for outs, prob in dist.items())

    def save(self, path: str | Path):
        """Save model to disk."""
        path = Path(path)
        with open(path, 'wb') as f:
            pickle.dump({
                'base_exit_probs': self.base_exit_probs,
                'outs_distribution': self.outs_distribution,
                'fitted': self.fitted,
            }, f)
        print(f"Saved reliever exit model to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "RelieverExitModel":
        """Load model from disk."""
        path = Path(path)
        with open(path, 'rb') as f:
            data = pickle.load(f)

        model = cls()
        model.base_exit_probs = data['base_exit_probs']
        model.outs_distribution = data['outs_distribution']
        model.fitted = data['fitted']

        return model


if __name__ == "__main__":
    # Test with sample data
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from game_simulation.extract_transitions import extract_pitcher_usage_stats
    from game_simulation.pitcher_roles import PitcherRoleClassifier

    print("Loading pitch data...")
    pitches = pd.read_parquet("data/raw/pitches.parquet")

    # Use recent data
    pitches_recent = pitches[pitches['game_date'] >= '2024-01-01']
    print(f"Using {len(pitches_recent):,} pitches from 2024+")

    # Classify pitcher roles
    pitcher_stats = extract_pitcher_usage_stats(pitches_recent)
    classifier = PitcherRoleClassifier()
    classified = classifier.fit_transform(pitcher_stats)
    pitcher_roles = dict(zip(classified['pitcher'], classified['role']))

    # Fit exit model
    exit_model = RelieverExitModel()
    exit_model.fit(pitches_recent, pitcher_roles)

    # Test predictions
    print("\n" + "="*60)
    print("Test exit probabilities:")
    print("="*60)

    test_states = [
        RelieverState(outs_recorded=3, role=ROLE_CLOSER, inning=9, score_diff=2),
        RelieverState(outs_recorded=3, role=ROLE_SETUP, inning=8, score_diff=1),
        RelieverState(outs_recorded=6, role=ROLE_LONG, inning=5, score_diff=-6),
        RelieverState(outs_recorded=2, role=ROLE_MIDDLE, inning=7, score_diff=0, runners_on=2),
    ]

    for state in test_states:
        prob = exit_model.predict_exit_prob(state)
        print(f"\n{state.role}, {state.outs_recorded} outs, inning {state.inning}, "
              f"score_diff={state.score_diff}, runners={state.runners_on}")
        print(f"  Exit probability: {prob:.2%}")
