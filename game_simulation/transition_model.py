"""
Bullpen Transition Model

Models the probability of which pitcher role enters next given the game state.

P(next_role | inning, score_diff, outs, prev_role)

This enables game simulation by predicting bullpen usage patterns.

The model captures managerial tendencies:
- Closers in save situations (9th, up 1-3)
- Setup men in 8th with lead
- Matchup-based specialist usage
- Long relievers in blowouts
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .pitcher_roles import (
    ROLE_STARTER, ROLE_CLOSER, ROLE_SETUP, ROLE_MIDDLE, ROLE_LONG, ROLE_SPECIALIST,
    RELIEF_ROLES, ALL_ROLES
)


@dataclass
class GameState:
    """Represents the current game state for transition prediction."""
    inning: int
    score_diff: int  # From pitching team's perspective (+ = winning)
    outs: int
    prev_role: str  # Role of pitcher being replaced
    is_home_pitching: bool = True

    def to_features(self) -> Dict:
        """Convert to feature dict for model."""
        return {
            'inning': self.inning,
            'score_diff_bucket': self._bucket_score_diff(),
            'outs': self.outs,
            'prev_is_starter': self.prev_role == ROLE_STARTER,
            'prev_role': self.prev_role,
            'is_home_pitching': self.is_home_pitching,
            'is_save_situation': self._is_save_situation(),
            'is_blowout': abs(self.score_diff) >= 5,
            'is_late_innings': self.inning >= 7,
            'is_ninth': self.inning >= 9,
        }

    def _bucket_score_diff(self) -> str:
        """Bucket score differential for modeling."""
        if self.score_diff <= -5:
            return "losing_big"
        elif self.score_diff <= -2:
            return "losing"
        elif self.score_diff == -1:
            return "losing_close"
        elif self.score_diff == 0:
            return "tied"
        elif self.score_diff == 1:
            return "winning_close"
        elif self.score_diff <= 3:
            return "winning"
        else:
            return "winning_big"

    def _is_save_situation(self) -> bool:
        """Check if this is a save situation (9th inning, lead of 1-3)."""
        return self.inning >= 9 and 1 <= self.score_diff <= 3


class BullpenTransitionModel:
    """
    Models bullpen transition probabilities.

    Uses historical pitcher change data to learn:
    P(next_role | game_state)

    Can use either:
    1. Empirical frequencies (counting)
    2. A trained classifier (e.g., LightGBM)
    """

    def __init__(self, method: str = "empirical"):
        """
        Initialize transition model.

        Args:
            method: "empirical" for frequency-based, "classifier" for ML model
        """
        self.method = method
        self.transition_counts: Dict[Tuple, Dict[str, int]] = {}
        self.classifier = None
        self.feature_columns: List[str] = []
        self.fitted = False

    def fit(
        self,
        transitions_df: pd.DataFrame,
        pitcher_roles: Dict[int, str]
    ) -> "BullpenTransitionModel":
        """
        Fit the transition model on historical data.

        Args:
            transitions_df: DataFrame of transition events from extract_transitions()
            pitcher_roles: Dict mapping pitcher_id -> role

        Returns:
            Self for chaining
        """
        print("Fitting bullpen transition model...")

        df = transitions_df.copy()

        # Add roles for prev and next pitcher
        df['prev_role'] = df['prev_pitcher'].map(pitcher_roles).fillna('UNKNOWN')
        df['next_role'] = df['next_pitcher'].map(pitcher_roles).fillna('UNKNOWN')

        # Filter to valid transitions (known roles, relief pitchers entering)
        df = df[
            (df['next_role'].isin(RELIEF_ROLES)) &
            (df['prev_role'].isin(ALL_ROLES))
        ]

        print(f"  Training on {len(df):,} valid transitions")

        if len(df) == 0:
            print("  WARNING: No valid transitions to train on!")
            self.fitted = True
            return self

        # Bucket score diff
        df['score_diff_bucket'] = df['score_diff'].apply(
            lambda x: self._bucket_score_diff(x) if pd.notna(x) else 'unknown'
        )

        # Create state features
        df['is_save_situation'] = (df['inning'] >= 9) & (df['score_diff'].between(1, 3))
        df['is_blowout'] = df['score_diff'].abs() >= 5
        df['is_late_innings'] = df['inning'] >= 7
        df['inning_bucket'] = pd.cut(
            df['inning'],
            bins=[0, 5, 6, 7, 8, 20],
            labels=['early', '6th', '7th', '8th', 'late']
        )

        if self.method == "empirical":
            self._fit_empirical(df)
        else:
            self._fit_classifier(df)

        self.fitted = True
        return self

    def _fit_empirical(self, df: pd.DataFrame):
        """Fit using empirical frequencies."""
        # Count transitions by state
        # State = (inning_bucket, score_diff_bucket, prev_is_starter)

        for _, row in df.iterrows():
            state = (
                row['inning_bucket'],
                row['score_diff_bucket'],
                row['prev_is_starter'],
            )

            if state not in self.transition_counts:
                self.transition_counts[state] = {role: 0 for role in RELIEF_ROLES}

            next_role = row['next_role']
            if next_role in RELIEF_ROLES:
                self.transition_counts[state][next_role] += 1

        # Print summary
        print("\n  Transition patterns by situation:")
        for state, counts in sorted(self.transition_counts.items())[:10]:
            total = sum(counts.values())
            if total > 10:  # Only show common situations
                probs = {r: c/total for r, c in counts.items() if c > 0}
                print(f"    {state}: {probs}")

    def _fit_classifier(self, df: pd.DataFrame):
        """Fit using ML classifier."""
        from lightgbm import LGBMClassifier
        from sklearn.preprocessing import LabelEncoder

        # Prepare features
        feature_cols = [
            'inning', 'score_diff', 'outs', 'prev_is_starter',
            'is_save_situation', 'is_blowout', 'is_late_innings'
        ]

        X = df[feature_cols].copy()
        X['prev_is_starter'] = X['prev_is_starter'].astype(int)
        X['is_save_situation'] = X['is_save_situation'].astype(int)
        X['is_blowout'] = X['is_blowout'].astype(int)
        X['is_late_innings'] = X['is_late_innings'].astype(int)
        X = X.fillna(0)

        # Encode target
        le = LabelEncoder()
        y = le.fit_transform(df['next_role'])

        # Train model
        self.classifier = LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=-1,
        )
        self.classifier.fit(X, y)
        self.feature_columns = feature_cols
        self._label_encoder = le

        print(f"  Trained classifier on {len(X)} samples")
        print(f"  Classes: {list(le.classes_)}")

    def predict_proba(self, state: GameState) -> Dict[str, float]:
        """
        Predict probability distribution over next pitcher role.

        Args:
            state: Current game state

        Returns:
            Dict mapping role -> probability
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.method == "empirical":
            return self._predict_empirical(state)
        else:
            return self._predict_classifier(state)

    def _predict_empirical(self, state: GameState) -> Dict[str, float]:
        """Predict using empirical frequencies."""
        features = state.to_features()

        # Map inning to bucket
        inning = features['inning']
        if inning <= 5:
            inning_bucket = 'early'
        elif inning == 6:
            inning_bucket = '6th'
        elif inning == 7:
            inning_bucket = '7th'
        elif inning == 8:
            inning_bucket = '8th'
        else:
            inning_bucket = 'late'

        state_key = (
            inning_bucket,
            features['score_diff_bucket'],
            features['prev_is_starter'],
        )

        if state_key in self.transition_counts:
            counts = self.transition_counts[state_key]
            total = sum(counts.values())
            if total > 0:
                return {role: count / total for role, count in counts.items()}

        # Fallback to overall distribution if state not seen
        return self._get_default_probs(state)

    def _predict_classifier(self, state: GameState) -> Dict[str, float]:
        """Predict using trained classifier."""
        features = state.to_features()

        X = pd.DataFrame([{
            'inning': features['inning'],
            'score_diff': state.score_diff,
            'outs': features['outs'],
            'prev_is_starter': int(features['prev_is_starter']),
            'is_save_situation': int(features['is_save_situation']),
            'is_blowout': int(features['is_blowout']),
            'is_late_innings': int(features['is_late_innings']),
        }])

        proba = self.classifier.predict_proba(X)[0]
        classes = self._label_encoder.classes_

        return dict(zip(classes, proba))

    def _get_default_probs(self, state: GameState) -> Dict[str, float]:
        """Default probabilities when state not seen."""
        features = state.to_features()

        # Use heuristics based on game situation
        if features['is_save_situation']:
            return {ROLE_CLOSER: 0.7, ROLE_SETUP: 0.2, ROLE_MIDDLE: 0.1}
        elif features['is_ninth']:
            return {ROLE_CLOSER: 0.5, ROLE_SETUP: 0.3, ROLE_MIDDLE: 0.2}
        elif features['is_late_innings']:
            return {ROLE_SETUP: 0.4, ROLE_MIDDLE: 0.4, ROLE_CLOSER: 0.1, ROLE_LONG: 0.1}
        elif features['is_blowout']:
            return {ROLE_LONG: 0.5, ROLE_MIDDLE: 0.4, ROLE_SETUP: 0.1}
        else:
            return {ROLE_MIDDLE: 0.5, ROLE_LONG: 0.3, ROLE_SETUP: 0.2}

    def _bucket_score_diff(self, score_diff: float) -> str:
        """Bucket score differential."""
        if pd.isna(score_diff):
            return "unknown"
        if score_diff <= -5:
            return "losing_big"
        elif score_diff <= -2:
            return "losing"
        elif score_diff == -1:
            return "losing_close"
        elif score_diff == 0:
            return "tied"
        elif score_diff == 1:
            return "winning_close"
        elif score_diff <= 3:
            return "winning"
        else:
            return "winning_big"

    def sample_next_role(self, state: GameState) -> str:
        """
        Sample a role from the predicted distribution.

        Useful for Monte Carlo simulation.
        """
        probs = self.predict_proba(state)
        roles = list(probs.keys())
        probabilities = list(probs.values())

        # Normalize in case of floating point issues
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]

        return np.random.choice(roles, p=probabilities)

    def save(self, path: str | Path):
        """Save model to disk."""
        path = Path(path)
        with open(path, 'wb') as f:
            pickle.dump({
                'method': self.method,
                'transition_counts': self.transition_counts,
                'classifier': self.classifier,
                'feature_columns': self.feature_columns,
                'label_encoder': getattr(self, '_label_encoder', None),
                'fitted': self.fitted,
            }, f)
        print(f"Saved transition model to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "BullpenTransitionModel":
        """Load model from disk."""
        path = Path(path)
        with open(path, 'rb') as f:
            data = pickle.load(f)

        model = cls(method=data['method'])
        model.transition_counts = data['transition_counts']
        model.classifier = data['classifier']
        model.feature_columns = data['feature_columns']
        model._label_encoder = data['label_encoder']
        model.fitted = data['fitted']

        return model


if __name__ == "__main__":
    # Test the transition model
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from game_simulation.extract_transitions import extract_transitions, extract_pitcher_usage_stats
    from game_simulation.pitcher_roles import PitcherRoleClassifier

    print("Loading pitch data...")
    pitches = pd.read_parquet("data/raw/pitches.parquet")

    # Use recent data for current patterns
    pitches_recent = pitches[pitches['game_date'] >= '2024-01-01']
    print(f"Using {len(pitches_recent):,} pitches from 2024+")

    # Extract transitions
    transitions = extract_transitions(pitches_recent)

    # Classify pitcher roles
    pitcher_stats = extract_pitcher_usage_stats(pitches_recent)
    classifier = PitcherRoleClassifier()
    classified = classifier.fit_transform(pitcher_stats)
    pitcher_roles = dict(zip(classified['pitcher'], classified['role']))

    # Fit transition model
    model = BullpenTransitionModel(method="empirical")
    model.fit(transitions, pitcher_roles)

    # Test predictions
    print("\n" + "="*60)
    print("Test predictions:")
    print("="*60)

    test_states = [
        GameState(inning=9, score_diff=2, outs=0, prev_role=ROLE_SETUP),
        GameState(inning=7, score_diff=0, outs=1, prev_role=ROLE_STARTER),
        GameState(inning=6, score_diff=-5, outs=2, prev_role=ROLE_STARTER),
        GameState(inning=8, score_diff=1, outs=0, prev_role=ROLE_MIDDLE),
    ]

    for state in test_states:
        probs = model.predict_proba(state)
        print(f"\nState: inning={state.inning}, score_diff={state.score_diff}, prev={state.prev_role}")
        print(f"  Probs: {probs}")
