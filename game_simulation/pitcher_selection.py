"""
Pitcher Selection Model

Models P(pitcher | role, team, availability) - which specific pitcher enters
from a given role based on team roster and recent usage.

Availability factors:
- Days since last pitched (back-to-back = lower probability)
- Recent workload (3 consecutive days = very low probability)
- Already appeared in this game = 0 probability
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import date, timedelta

from .pitcher_roles import RELIEF_ROLES


@dataclass
class PitcherAvailability:
    """Tracks a pitcher's availability status."""
    pitcher_id: int
    role: str
    team: str
    days_since_pitched: int = 999  # Large number if hasn't pitched recently
    consecutive_days: int = 0  # Days pitched in a row
    innings_last_3_days: float = 0.0
    appeared_today: bool = False

    @property
    def availability_score(self) -> float:
        """
        Compute availability score (0 to 1).

        Higher = more available.
        """
        if self.appeared_today:
            return 0.0

        # Base availability by days rest
        if self.days_since_pitched == 0:
            # Pitched today already
            return 0.0
        elif self.days_since_pitched == 1:
            # Back-to-back
            base = 0.3
        elif self.days_since_pitched == 2:
            base = 0.7
        else:
            # 3+ days rest
            base = 1.0

        # Penalty for consecutive days
        if self.consecutive_days >= 3:
            # Pitched 3 days in a row - very unlikely to be used
            base *= 0.05
        elif self.consecutive_days == 2:
            # Pitched 2 days in a row
            base *= 0.5

        # Penalty for high recent workload
        if self.innings_last_3_days >= 3.0:
            base *= 0.5

        return base


@dataclass
class TeamBullpen:
    """Represents a team's bullpen roster with availability."""
    team: str
    pitchers: Dict[int, PitcherAvailability] = field(default_factory=dict)

    def get_available_by_role(self, role: str, exclude: Set[int] = None) -> List[PitcherAvailability]:
        """Get available pitchers for a role."""
        exclude = exclude or set()
        return [
            p for p in self.pitchers.values()
            if p.role == role and p.pitcher_id not in exclude and p.availability_score > 0
        ]

    def get_all_available(self, exclude: Set[int] = None) -> List[PitcherAvailability]:
        """Get all available relievers."""
        exclude = exclude or set()
        return [
            p for p in self.pitchers.values()
            if p.pitcher_id not in exclude and p.availability_score > 0
        ]


class PitcherSelectionModel:
    """
    Selects which specific pitcher enters from a role.

    Uses availability-weighted sampling within role.
    Falls back to adjacent roles if no one available in target role.
    """

    # Role fallback order (if primary role has no one available)
    ROLE_FALLBACKS = {
        'CL': ['SU', 'MR'],
        'SU': ['MR', 'CL'],
        'MR': ['SU', 'LR'],
        'LR': ['MR', 'SU'],
        'SPEC': ['MR', 'SU'],
    }

    def __init__(self):
        self.team_bullpens: Dict[str, TeamBullpen] = {}
        self.pitcher_roles: Dict[int, str] = {}
        self.fitted = False

    def fit(
        self,
        pitcher_roles: Dict[int, str],
        team_rosters: pd.DataFrame,
        recent_appearances: pd.DataFrame,
        game_date: date,
    ) -> "PitcherSelectionModel":
        """
        Fit selection model with current roster and availability info.

        Args:
            pitcher_roles: Mapping of pitcher_id -> role
            team_rosters: DataFrame with pitcher_id, team columns
            recent_appearances: DataFrame with pitcher_id, game_date, innings_pitched
            game_date: Current date (to compute days since pitched)

        Returns:
            Self for chaining
        """
        print("Fitting pitcher selection model...")

        self.pitcher_roles = pitcher_roles
        self.team_bullpens = {}

        # Group pitchers by team
        for _, row in team_rosters.iterrows():
            pitcher_id = row['pitcher']
            team = row['team']
            role = pitcher_roles.get(pitcher_id, 'MR')

            # Skip starters
            if role not in RELIEF_ROLES:
                continue

            if team not in self.team_bullpens:
                self.team_bullpens[team] = TeamBullpen(team=team)

            # Compute availability from recent appearances
            pitcher_apps = recent_appearances[
                recent_appearances['pitcher'] == pitcher_id
            ].sort_values('game_date', ascending=False)

            days_since = 999
            consecutive = 0
            ip_last_3 = 0.0

            if len(pitcher_apps) > 0:
                last_pitched = pd.to_datetime(pitcher_apps.iloc[0]['game_date']).date()
                days_since = (game_date - last_pitched).days

                # Count consecutive days
                for i, app in pitcher_apps.iterrows():
                    app_date = pd.to_datetime(app['game_date']).date()
                    expected_date = game_date - timedelta(days=consecutive + 1)
                    if app_date == expected_date:
                        consecutive += 1
                    else:
                        break

                # Sum IP in last 3 days
                cutoff = game_date - timedelta(days=3)
                recent = pitcher_apps[pd.to_datetime(pitcher_apps['game_date']).dt.date > cutoff]
                if 'innings_pitched' in recent.columns:
                    ip_last_3 = recent['innings_pitched'].sum()
                elif 'total_outs' in recent.columns:
                    ip_last_3 = recent['total_outs'].sum() / 3

            avail = PitcherAvailability(
                pitcher_id=pitcher_id,
                role=role,
                team=team,
                days_since_pitched=days_since,
                consecutive_days=consecutive,
                innings_last_3_days=ip_last_3,
            )

            self.team_bullpens[team].pitchers[pitcher_id] = avail

        print(f"  Loaded {len(self.team_bullpens)} team bullpens")
        for team, bullpen in list(self.team_bullpens.items())[:3]:
            print(f"    {team}: {len(bullpen.pitchers)} relievers")

        self.fitted = True
        return self

    def build_from_pitches(
        self,
        pitches_df: pd.DataFrame,
        pitcher_roles: Dict[int, str],
        game_date: date,
        lookback_days: int = 7,
    ) -> "PitcherSelectionModel":
        """
        Build selection model directly from pitch data.

        Args:
            pitches_df: Pitch-level data with game_date, pitcher, home_team, away_team
            pitcher_roles: Mapping of pitcher_id -> role
            game_date: Current game date
            lookback_days: Days to look back for recent appearances

        Returns:
            Self for chaining
        """
        print("Building pitcher selection model from pitch data...")

        self.pitcher_roles = pitcher_roles

        # Get recent appearances
        cutoff = game_date - timedelta(days=lookback_days)
        recent_pitches = pitches_df[
            pd.to_datetime(pitches_df['game_date']).dt.date > cutoff
        ]

        # Build appearance summary
        appearances = recent_pitches.groupby(['game_pk', 'pitcher', 'game_date']).agg(
            total_outs=('events', lambda x: x.isin([
                'strikeout', 'field_out', 'grounded_into_double_play',
                'force_out', 'sac_fly', 'sac_bunt', 'double_play',
                'fielders_choice_out', 'strikeout_double_play'
            ]).sum()),
        ).reset_index()
        appearances['innings_pitched'] = appearances['total_outs'] / 3

        # Get team assignments from most recent appearance
        # Determine team by checking if pitcher was home or away
        team_assignments = []
        for pitcher_id in recent_pitches['pitcher'].unique():
            p_pitches = recent_pitches[recent_pitches['pitcher'] == pitcher_id]
            latest = p_pitches.sort_values('game_date').iloc[-1]

            # Check if this pitcher was on home or away team
            # Pitchers pitch when their team is fielding (opponent batting)
            # If inning_topbot == 'Top', away team is batting, so home team is pitching
            if 'inning_topbot' in latest.index:
                if latest['inning_topbot'] == 'Top':
                    team = latest['home_team']
                else:
                    team = latest['away_team']
            else:
                team = latest.get('home_team', 'UNK')

            team_assignments.append({'pitcher': pitcher_id, 'team': team})

        team_rosters = pd.DataFrame(team_assignments)

        return self.fit(pitcher_roles, team_rosters, appearances, game_date)

    def select_pitcher(
        self,
        role: str,
        team: str,
        already_pitched: Set[int] = None,
    ) -> Optional[int]:
        """
        Select a pitcher from a role using availability-weighted sampling.

        Args:
            role: Target role (CL, SU, MR, LR, SPEC)
            team: Team code
            already_pitched: Set of pitcher IDs already used in this game

        Returns:
            Selected pitcher ID, or None if no one available
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        already_pitched = already_pitched or set()

        if team not in self.team_bullpens:
            return None

        bullpen = self.team_bullpens[team]

        # Try primary role first
        candidates = bullpen.get_available_by_role(role, exclude=already_pitched)

        # Fallback to adjacent roles if needed
        if not candidates and role in self.ROLE_FALLBACKS:
            for fallback_role in self.ROLE_FALLBACKS[role]:
                candidates = bullpen.get_available_by_role(fallback_role, exclude=already_pitched)
                if candidates:
                    break

        # Last resort: any available reliever
        if not candidates:
            candidates = bullpen.get_all_available(exclude=already_pitched)

        if not candidates:
            return None

        # Availability-weighted sampling
        scores = [c.availability_score for c in candidates]
        total = sum(scores)

        if total == 0:
            return None

        probs = [s / total for s in scores]
        selected = np.random.choice(
            [c.pitcher_id for c in candidates],
            p=probs
        )

        return selected

    def get_selection_probs(
        self,
        role: str,
        team: str,
        already_pitched: Set[int] = None,
    ) -> Dict[int, float]:
        """
        Get probability distribution over pitchers for a role.

        Returns:
            Dict mapping pitcher_id -> selection probability
        """
        already_pitched = already_pitched or set()

        if team not in self.team_bullpens:
            return {}

        bullpen = self.team_bullpens[team]
        candidates = bullpen.get_available_by_role(role, exclude=already_pitched)

        # Include fallbacks
        if not candidates and role in self.ROLE_FALLBACKS:
            for fallback_role in self.ROLE_FALLBACKS[role]:
                candidates.extend(
                    bullpen.get_available_by_role(fallback_role, exclude=already_pitched)
                )

        if not candidates:
            return {}

        scores = {c.pitcher_id: c.availability_score for c in candidates}
        total = sum(scores.values())

        if total == 0:
            return {}

        return {pid: score / total for pid, score in scores.items()}

    def mark_pitched(self, team: str, pitcher_id: int):
        """Mark a pitcher as having appeared in today's game."""
        if team in self.team_bullpens:
            if pitcher_id in self.team_bullpens[team].pitchers:
                self.team_bullpens[team].pitchers[pitcher_id].appeared_today = True


if __name__ == "__main__":
    # Test with sample data
    import sys
    from datetime import date
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from game_simulation.extract_transitions import extract_pitcher_usage_stats
    from game_simulation.pitcher_roles import PitcherRoleClassifier

    print("Loading pitch data...")
    pitches = pd.read_parquet("data/raw/pitches.parquet")

    # Use recent data
    pitches_recent = pitches[pitches['game_date'] >= '2024-01-01']
    print(f"Using {len(pitches_recent):,} pitches")

    # Classify pitcher roles
    pitcher_stats = extract_pitcher_usage_stats(pitches_recent)
    classifier = PitcherRoleClassifier()
    classified = classifier.fit_transform(pitcher_stats)
    pitcher_roles = dict(zip(classified['pitcher'], classified['role']))

    # Build selection model
    game_date = date(2024, 9, 15)  # Example date
    selector = PitcherSelectionModel()
    selector.build_from_pitches(pitches_recent, pitcher_roles, game_date)

    # Test selection
    print("\n" + "="*60)
    print("Test pitcher selection:")
    print("="*60)

    # Pick a team
    teams = list(selector.team_bullpens.keys())
    if teams:
        test_team = teams[0]
        print(f"\nTeam: {test_team}")
        print(f"Bullpen: {len(selector.team_bullpens[test_team].pitchers)} relievers")

        for role in ['CL', 'SU', 'MR']:
            probs = selector.get_selection_probs(role, test_team)
            print(f"\n{role} selection probabilities:")
            for pid, prob in sorted(probs.items(), key=lambda x: -x[1])[:3]:
                avail = selector.team_bullpens[test_team].pitchers.get(pid)
                if avail:
                    print(f"  {pid}: {prob:.1%} (days_rest={avail.days_since_pitched}, "
                          f"consec={avail.consecutive_days})")
