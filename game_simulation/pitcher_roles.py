"""
Pitcher Role Classification

Classifies pitchers into roles based on their historical usage patterns:
- STARTER (SP): Starts games, pitches 5+ innings typically
- CLOSER (CL): 9th inning, save situations, ~1 IP
- SETUP (SU): 7th-8th inning, high leverage, holds
- MIDDLE (MR): 5th-7th inning, medium leverage
- LONG (LR): Multi-inning relief, blowouts, spot starts
- SPECIALIST (SPEC): Short appearances, platoon matchups

Role classification uses:
- Starter vs relief appearance ratio
- Average inning of appearance
- Innings per appearance
- Total appearances
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path


# Role constants
ROLE_STARTER = "SP"
ROLE_CLOSER = "CL"
ROLE_SETUP = "SU"
ROLE_MIDDLE = "MR"
ROLE_LONG = "LR"
ROLE_SPECIALIST = "SPEC"

ALL_ROLES = [ROLE_STARTER, ROLE_CLOSER, ROLE_SETUP, ROLE_MIDDLE, ROLE_LONG, ROLE_SPECIALIST]
RELIEF_ROLES = [ROLE_CLOSER, ROLE_SETUP, ROLE_MIDDLE, ROLE_LONG, ROLE_SPECIALIST]


@dataclass
class RoleThresholds:
    """Thresholds for role classification."""
    # Starter thresholds
    starter_relief_pct_max: float = 0.3  # Max 30% relief apps to be starter
    starter_min_starts: int = 3           # Minimum starts to qualify

    # Closer thresholds
    closer_min_avg_inning: float = 8.5    # Must pitch late
    closer_max_ip_per_app: float = 1.3    # Short outings
    closer_min_apps: int = 10             # Minimum appearances

    # Setup thresholds
    setup_min_avg_inning: float = 7.0     # 7th-8th innings
    setup_max_avg_inning: float = 8.5     # Before closer
    setup_max_ip_per_app: float = 1.5

    # Long relief thresholds
    long_min_ip_per_app: float = 1.5      # Multi-inning

    # Specialist thresholds (very short appearances)
    spec_max_ip_per_app: float = 0.5
    spec_max_outs_per_app: float = 3      # 1 inning or less


class PitcherRoleClassifier:
    """
    Classifies pitchers into roles based on usage patterns.

    Usage:
        classifier = PitcherRoleClassifier()
        roles = classifier.fit_transform(pitcher_stats)
    """

    def __init__(self, thresholds: Optional[RoleThresholds] = None):
        """
        Initialize classifier.

        Args:
            thresholds: Custom thresholds for role classification.
                       If None, uses defaults.
        """
        self.thresholds = thresholds or RoleThresholds()
        self.role_assignments: Dict[int, str] = {}

    def fit_transform(self, pitcher_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Classify pitchers into roles based on their usage stats.

        Args:
            pitcher_stats: DataFrame with columns:
                - pitcher: pitcher ID
                - games: total games
                - starts: games started
                - relief_apps: relief appearances
                - avg_first_inning: average first inning pitched
                - avg_ip_per_app: average innings per appearance
                - relief_pct: percentage of appearances in relief

        Returns:
            DataFrame with additional 'role' column
        """
        df = pitcher_stats.copy()
        t = self.thresholds

        # Initialize role as unknown
        df['role'] = 'UNKNOWN'

        # 1. Identify STARTERS
        # High percentage of starts, multiple starts
        is_starter = (
            (df['relief_pct'] <= t.starter_relief_pct_max) &
            (df['starts'] >= t.starter_min_starts)
        )
        df.loc[is_starter, 'role'] = ROLE_STARTER

        # For remaining pitchers (relievers), classify by usage
        relievers = df['role'] == 'UNKNOWN'

        # 2. Identify CLOSERS
        # Pitch late (avg inning >= 8.5), short outings, enough appearances
        is_closer = (
            relievers &
            (df['avg_first_inning'] >= t.closer_min_avg_inning) &
            (df['avg_ip_per_app'] <= t.closer_max_ip_per_app) &
            (df['games'] >= t.closer_min_apps)
        )
        df.loc[is_closer, 'role'] = ROLE_CLOSER

        # Update relievers mask
        relievers = df['role'] == 'UNKNOWN'

        # 3. Identify SETUP
        # Pitch 7th-8th, short outings
        is_setup = (
            relievers &
            (df['avg_first_inning'] >= t.setup_min_avg_inning) &
            (df['avg_first_inning'] < t.setup_max_avg_inning) &
            (df['avg_ip_per_app'] <= t.setup_max_ip_per_app)
        )
        df.loc[is_setup, 'role'] = ROLE_SETUP

        # Update relievers mask
        relievers = df['role'] == 'UNKNOWN'

        # 4. Identify LONG RELIEF
        # Multi-inning appearances
        is_long = (
            relievers &
            (df['avg_ip_per_app'] >= t.long_min_ip_per_app)
        )
        df.loc[is_long, 'role'] = ROLE_LONG

        # Update relievers mask
        relievers = df['role'] == 'UNKNOWN'

        # 5. Identify SPECIALISTS
        # Very short appearances (< 1 IP on average)
        is_specialist = (
            relievers &
            (df['avg_ip_per_app'] <= t.spec_max_ip_per_app)
        )
        df.loc[is_specialist, 'role'] = ROLE_SPECIALIST

        # 6. Remaining are MIDDLE RELIEF
        df.loc[df['role'] == 'UNKNOWN', 'role'] = ROLE_MIDDLE

        # Store assignments
        self.role_assignments = dict(zip(df['pitcher'], df['role']))

        # Print summary
        print("Role classification summary:")
        print(df['role'].value_counts().to_string())

        return df

    def get_role(self, pitcher_id: int) -> str:
        """Get role for a specific pitcher."""
        return self.role_assignments.get(pitcher_id, 'UNKNOWN')

    def get_pitchers_by_role(self, role: str) -> List[int]:
        """Get all pitchers with a specific role."""
        return [p for p, r in self.role_assignments.items() if r == role]

    def get_team_bullpen_roles(
        self,
        pitcher_stats: pd.DataFrame,
        team_rosters: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get bullpen roles for a specific team.

        Args:
            pitcher_stats: DataFrame with pitcher stats and roles
            team_rosters: DataFrame with pitcher-team assignments

        Returns:
            DataFrame with team, pitcher, role
        """
        # Merge pitcher stats with team info
        bullpen = pitcher_stats.merge(team_rosters, on='pitcher', how='inner')

        # Filter to relievers only
        bullpen = bullpen[bullpen['role'] != ROLE_STARTER]

        return bullpen[['pitcher', 'team', 'role', 'games', 'avg_first_inning', 'avg_ip_per_app']]


def classify_pitchers_from_pitches(pitches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to classify pitchers directly from pitch data.

    Args:
        pitches_df: Raw pitch data

    Returns:
        DataFrame with pitcher IDs and their roles
    """
    from .extract_transitions import extract_pitcher_usage_stats

    # Extract usage stats
    pitcher_stats = extract_pitcher_usage_stats(pitches_df)

    # Classify roles
    classifier = PitcherRoleClassifier()
    classified = classifier.fit_transform(pitcher_stats)

    return classified


if __name__ == "__main__":
    # Test with sample data
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from game_simulation.extract_transitions import extract_pitcher_usage_stats

    print("Loading pitch data...")
    pitches = pd.read_parquet("data/raw/pitches.parquet")

    # Get 2025 data only for current roles
    pitches_2025 = pitches[pitches['game_date'] >= '2025-01-01']
    print(f"Using {len(pitches_2025):,} pitches from 2025+")

    # Extract stats and classify
    pitcher_stats = extract_pitcher_usage_stats(pitches_2025)
    classifier = PitcherRoleClassifier()
    classified = classifier.fit_transform(pitcher_stats)

    print("\nSample classifications:")
    print(classified.nlargest(20, 'games')[
        ['pitcher', 'games', 'starts', 'relief_pct', 'avg_first_inning', 'avg_ip_per_app', 'role']
    ].to_string())
