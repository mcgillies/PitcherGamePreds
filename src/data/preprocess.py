"""
Preprocessing for pitcher-batter matchup model.

Builds training data from Statcast pitch-level data with:
- At-bat pitch characteristics (actual pitches in each PA)
- Pitcher historical profiles (lagged, rolling by starts)
- Batter historical profiles (lagged, rolling by games)
- Handedness encoding
- Numeric scaling
"""

from datetime import datetime
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Pitch type groupings for at-bat aggregation
FASTBALLS = ['FF', 'SI', 'FC']
BREAKING = ['SL', 'CU', 'KC', 'SV', 'ST']
OFFSPEED = ['CH', 'FS', 'SC', 'FO']

# Target outcome mapping
OUTCOME_MAP = {
    'strikeout': 'K',
    'strikeout_double_play': 'K',
    'walk': 'BB',
    'hit_by_pitch': 'HBP',
    'single': '1B',
    'double': '2B',
    'triple': '3B',
    'home_run': 'HR',
    'field_out': 'OUT',
    'grounded_into_double_play': 'OUT',
    'force_out': 'OUT',
    'sac_fly': 'OUT',
    'sac_bunt': 'OUT',
    'fielders_choice': 'OUT',
    'fielders_choice_out': 'OUT',
    'double_play': 'OUT',
    'triple_play': 'OUT',
    'field_error': 'OUT',
}

# Outcome classes for model
OUTCOME_CLASSES = ['K', 'BB', '1B', '2B', '3B', 'HR', 'OUT']


class MatchupPreprocessor:
    """
    Preprocessor for pitcher-batter matchup data.

    Handles:
    - Extracting plate appearances with outcomes
    - Computing rolling stats by starts/games
    - Scaling numeric features
    - Encoding target outcomes
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.numeric_columns = []
        self.binary_columns = []
        self.fitted = False

    def build_matchup_data(
        self,
        pitches_df: pd.DataFrame,
        pitcher_profiles: pd.DataFrame,
        batter_profiles: pd.DataFrame,
        pitcher_rolling_windows: list[int] = [5, 10, 20],
        batter_rolling_windows: list[int] = [25, 50, 100],
    ) -> pd.DataFrame:
        """
        Build matchup training data from raw components.

        Args:
            pitches_df: Raw Statcast pitch-level data
            pitcher_profiles: Pitcher arsenal profiles (includes usage rates per pitch type)
            batter_profiles: Batter pitch performance profiles
            pitcher_rolling_windows: Windows for pitcher rolling stats by starts (default: 5, 10, 20)
            batter_rolling_windows: Windows for batter rolling stats by games (default: 25, 50, 100)

        Returns:
            DataFrame with one row per plate appearance
        """
        print("Building matchup training data...")

        # Step 1: Extract plate appearances (no at-bat aggregation - use pitcher profile instead)
        print("  Extracting plate appearances...")
        pa_features = self._extract_plate_appearances(pitches_df)

        # Step 2: Compute rolling stats for pitchers (by starts)
        print("  Computing pitcher rolling stats...")
        pitcher_rolling = self._compute_pitcher_rolling(
            pitches_df, pitcher_rolling_windows
        )

        # Step 3: Compute rolling stats for batters (by games)
        print("  Computing batter rolling stats...")
        batter_rolling = self._compute_batter_rolling(
            pitches_df, batter_rolling_windows
        )

        # Step 4: Merge everything together
        print("  Merging features...")
        matchups = pa_features.copy()

        # Add pitcher profile (season-level)
        matchups = matchups.merge(
            pitcher_profiles.add_prefix('p_'),
            left_on='pitcher_id',
            right_on='p_pitcher_id',
            how='left'
        )

        # Add batter profile (season-level)
        matchups = matchups.merge(
            batter_profiles.add_prefix('b_'),
            left_on='batter_id',
            right_on='b_batter_id',
            how='left'
        )

        # Add pitcher rolling stats
        if pitcher_rolling is not None:
            matchups = matchups.merge(
                pitcher_rolling,
                on=['pitcher_id', 'game_date'],
                how='left'
            )

        # Add batter rolling stats
        if batter_rolling is not None:
            matchups = matchups.merge(
                batter_rolling,
                on=['batter_id', 'game_date'],
                how='left'
            )

        # Step 5: Add handedness features
        print("  Adding handedness features...")
        matchups = self._add_handedness_features(matchups)

        # Step 6: Filter to valid outcomes
        matchups = matchups[matchups['outcome'].isin(OUTCOME_CLASSES)]

        print(f"  Built {len(matchups):,} matchup rows")
        return matchups

    def _extract_plate_appearances(self, pitches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract plate appearances with outcomes (no pitch aggregation).

        Pitch characteristics come from pitcher profiles instead of per-AB aggregation.
        """
        df = pitches_df.copy()

        # Filter to valid data
        df = df[df['pitcher'].notna() & df['batter'].notna()]

        # Group by plate appearance - just get the outcome and handedness
        pa_groups = df.groupby(['game_pk', 'game_date', 'at_bat_number', 'pitcher', 'batter'])

        pa_features = pa_groups.agg(
            # Handedness
            p_throws=('p_throws', 'first'),
            stand=('stand', 'first'),

            # Outcome (from last pitch of PA where event occurred)
            events=('events', 'last'),
        ).reset_index()

        # Rename columns
        pa_features = pa_features.rename(columns={
            'pitcher': 'pitcher_id',
            'batter': 'batter_id',
        })

        # Map outcome
        pa_features['outcome'] = pa_features['events'].map(OUTCOME_MAP)

        # Filter to completed PAs with valid outcomes
        pa_features = pa_features[pa_features['outcome'].notna()]

        return pa_features

    def _compute_pitcher_rolling(
        self,
        pitches_df: pd.DataFrame,
        windows: list[int],
    ) -> pd.DataFrame | None:
        """
        Compute rolling stats for pitchers by number of starts.

        Includes: whiff%, csw%, K%, BB%, zone%, chase%, avg_velo,
                  xwOBA against, avg_exit_velo against, barrel% against
        """
        df = pitches_df.copy()

        # Get unique pitcher-game combinations (starts)
        starts = df.groupby(['pitcher', 'game_date']).agg(
            # Pitch counts
            pitches=('release_speed', 'count'),

            # Swing/whiff data
            whiffs=('description', lambda x: x.isin(['swinging_strike', 'swinging_strike_blocked']).sum()),
            swings=('description', lambda x: x.isin([
                'swinging_strike', 'swinging_strike_blocked',
                'foul', 'foul_tip', 'hit_into_play'
            ]).sum()),
            called_strikes=('description', lambda x: (x == 'called_strike').sum()),

            # Zone data
            in_zone=('zone', lambda x: x.between(1, 9).sum()),
            out_zone=('zone', lambda x: (~x.between(1, 9)).sum()),
            out_zone_swings=('zone', 'count'),  # Placeholder, computed below

            # PA outcomes (for K%, BB%)
            strikeouts=('events', lambda x: x.isin(['strikeout', 'strikeout_double_play']).sum()),
            walks=('events', lambda x: x.isin(['walk']).sum()),
            pa_count=('events', lambda x: x.notna().sum()),

            # Velocity
            avg_velo=('release_speed', 'mean'),

            # Batted ball data (for xwOBA, EV, barrel%)
            avg_exit_velo=('launch_speed', 'mean'),
            xwoba=('estimated_woba_using_speedangle', 'mean'),
            batted_balls=('launch_speed', 'count'),
            hard_hit=('launch_speed', lambda x: (x >= 95).sum()),
            barrels=('launch_speed', lambda x: ((x >= 98) & (df.loc[x.index, 'launch_angle'].between(26, 30))).sum() if len(x) > 0 else 0),
        ).reset_index()

        # Compute chase swings separately (swings on pitches outside zone)
        chase_data = df[~df['zone'].between(1, 9)].groupby(['pitcher', 'game_date']).agg(
            out_zone_swings=('description', lambda x: x.isin([
                'swinging_strike', 'swinging_strike_blocked',
                'foul', 'foul_tip', 'hit_into_play'
            ]).sum()),
        ).reset_index()

        starts = starts.merge(chase_data, on=['pitcher', 'game_date'], how='left', suffixes=('_drop', ''))
        starts = starts.drop(columns=[c for c in starts.columns if c.endswith('_drop')])

        starts = starts.sort_values(['pitcher', 'game_date'])

        # Compute rates
        starts['whiff_rate'] = starts['whiffs'] / starts['swings'].replace(0, np.nan)
        starts['csw_rate'] = (starts['whiffs'] + starts['called_strikes']) / starts['pitches']
        starts['k_rate'] = starts['strikeouts'] / starts['pa_count'].replace(0, np.nan)
        starts['bb_rate'] = starts['walks'] / starts['pa_count'].replace(0, np.nan)
        starts['zone_rate'] = starts['in_zone'] / starts['pitches']
        starts['chase_rate'] = starts['out_zone_swings'] / starts['out_zone'].replace(0, np.nan)
        starts['barrel_rate'] = starts['barrels'] / starts['batted_balls'].replace(0, np.nan)
        starts['hard_hit_rate'] = starts['hard_hit'] / starts['batted_balls'].replace(0, np.nan)

        # Stats to compute rolling for
        rolling_stats = [
            'whiff_rate', 'csw_rate', 'k_rate', 'bb_rate',
            'zone_rate', 'chase_rate', 'avg_velo',
            'xwoba', 'avg_exit_velo', 'barrel_rate', 'hard_hit_rate'
        ]

        # Compute rolling stats for each window
        rolling_cols = []
        for window in windows:
            for stat in rolling_stats:
                col_name = f'p_roll{window}_{stat}'
                starts[col_name] = (
                    starts.groupby('pitcher')[stat]
                    .shift(1)  # Lag to avoid leakage
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
                rolling_cols.append(col_name)

        # Select output columns
        output_cols = ['pitcher', 'game_date'] + rolling_cols
        result = starts[output_cols].rename(columns={'pitcher': 'pitcher_id'})

        return result

    def _compute_batter_rolling(
        self,
        pitches_df: pd.DataFrame,
        windows: list[int],
    ) -> pd.DataFrame | None:
        """
        Compute rolling stats for batters by number of games.

        Includes: whiff%, contact%, K%, BB%, chase%, zone_swing%,
                  xwOBA, avg_exit_velo, barrel%, hard_hit%
        """
        df = pitches_df.copy()

        # Get unique batter-game combinations
        games = df.groupby(['batter', 'game_date']).agg(
            # Pitch counts
            pitches=('release_speed', 'count'),

            # Swing/whiff data
            whiffs=('description', lambda x: x.isin(['swinging_strike', 'swinging_strike_blocked']).sum()),
            swings=('description', lambda x: x.isin([
                'swinging_strike', 'swinging_strike_blocked',
                'foul', 'foul_tip', 'hit_into_play'
            ]).sum()),

            # Zone discipline
            in_zone=('zone', lambda x: x.between(1, 9).sum()),
            out_zone=('zone', lambda x: (~x.between(1, 9)).sum()),

            # PA outcomes (for K%, BB%)
            strikeouts=('events', lambda x: x.isin(['strikeout', 'strikeout_double_play']).sum()),
            walks=('events', lambda x: x.isin(['walk']).sum()),
            pa_count=('events', lambda x: x.notna().sum()),

            # Batted ball data
            avg_exit_velo=('launch_speed', 'mean'),
            xwoba=('estimated_woba_using_speedangle', 'mean'),
            batted_balls=('launch_speed', 'count'),
            hard_hit=('launch_speed', lambda x: (x >= 95).sum()),
            barrels=('launch_speed', lambda x: ((x >= 98) & (df.loc[x.index, 'launch_angle'].between(26, 30))).sum() if len(x) > 0 else 0),
        ).reset_index()

        # Compute zone swings separately
        zone_swing_data = df[df['zone'].between(1, 9)].groupby(['batter', 'game_date']).agg(
            zone_swings=('description', lambda x: x.isin([
                'swinging_strike', 'swinging_strike_blocked',
                'foul', 'foul_tip', 'hit_into_play'
            ]).sum()),
        ).reset_index()

        chase_data = df[~df['zone'].between(1, 9)].groupby(['batter', 'game_date']).agg(
            chase_swings=('description', lambda x: x.isin([
                'swinging_strike', 'swinging_strike_blocked',
                'foul', 'foul_tip', 'hit_into_play'
            ]).sum()),
        ).reset_index()

        games = games.merge(zone_swing_data, on=['batter', 'game_date'], how='left')
        games = games.merge(chase_data, on=['batter', 'game_date'], how='left')

        games = games.sort_values(['batter', 'game_date'])

        # Compute rates
        games['whiff_rate'] = games['whiffs'] / games['swings'].replace(0, np.nan)
        games['contact_rate'] = 1 - games['whiff_rate']
        games['k_rate'] = games['strikeouts'] / games['pa_count'].replace(0, np.nan)
        games['bb_rate'] = games['walks'] / games['pa_count'].replace(0, np.nan)
        games['zone_swing_rate'] = games['zone_swings'] / games['in_zone'].replace(0, np.nan)
        games['chase_rate'] = games['chase_swings'] / games['out_zone'].replace(0, np.nan)
        games['barrel_rate'] = games['barrels'] / games['batted_balls'].replace(0, np.nan)
        games['hard_hit_rate'] = games['hard_hit'] / games['batted_balls'].replace(0, np.nan)

        # Stats to compute rolling for
        rolling_stats = [
            'whiff_rate', 'contact_rate', 'k_rate', 'bb_rate',
            'zone_swing_rate', 'chase_rate',
            'xwoba', 'avg_exit_velo', 'barrel_rate', 'hard_hit_rate'
        ]

        # Compute rolling stats for each window
        rolling_cols = []
        for window in windows:
            for stat in rolling_stats:
                col_name = f'b_roll{window}_{stat}'
                games[col_name] = (
                    games.groupby('batter')[stat]
                    .shift(1)  # Lag to avoid leakage
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
                rolling_cols.append(col_name)

        # Select output columns
        output_cols = ['batter', 'game_date'] + rolling_cols
        result = games[output_cols].rename(columns={'batter': 'batter_id'})

        return result

    def _add_handedness_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add handedness encoding features.
        """
        result = df.copy()

        # Get handedness from columns (may be from different sources)
        if 'p_throws' not in result.columns and 'p_p_throws' in result.columns:
            result['p_throws'] = result['p_p_throws']
        if 'stand' not in result.columns and 'b_stand' in result.columns:
            result['stand'] = result['b_stand']

        # Encode pitcher hand as binary
        result['p_throws_L'] = (result['p_throws'] == 'L').astype(int)
        result['p_throws_R'] = (result['p_throws'] == 'R').astype(int)

        # Encode batter stance as binary
        result['stand_L'] = (result['stand'] == 'L').astype(int)
        result['stand_R'] = (result['stand'] == 'R').astype(int)

        # Platoon matchup encoding (all 4 combinations)
        result['matchup_LvL'] = ((result['p_throws'] == 'L') & (result['stand'] == 'L')).astype(int)
        result['matchup_LvR'] = ((result['p_throws'] == 'L') & (result['stand'] == 'R')).astype(int)
        result['matchup_RvL'] = ((result['p_throws'] == 'R') & (result['stand'] == 'L')).astype(int)
        result['matchup_RvR'] = ((result['p_throws'] == 'R') & (result['stand'] == 'R')).astype(int)

        # Same hand indicator
        result['same_hand'] = (result['p_throws'] == result['stand']).astype(int)

        return result

    def get_feature_columns(self, df: pd.DataFrame) -> tuple[list, list]:
        """
        Identify numeric and binary feature columns.

        Returns:
            Tuple of (numeric_columns, binary_columns)
        """
        # Columns to exclude
        exclude_cols = {
            # Identifiers
            'pitcher_id', 'batter_id', 'game_pk', 'game_date', 'at_bat_number',
            'p_pitcher_id', 'b_batter_id',
            # Names
            'pitcher_name', 'batter_name', 'p_pitcher_name', 'b_batter_name',
            # Raw handedness (encoded separately)
            'p_throws', 'stand', 'p_p_throws', 'b_stand',
            # Outcome columns
            'events', 'outcome',
            # Primary pitch (dropped per user request)
            'p_primary_pitch', 'primary_pitch',
        }

        # Binary columns (handedness encoding)
        binary_cols = [
            'p_throws_L', 'p_throws_R', 'stand_L', 'stand_R',
            'matchup_LvL', 'matchup_LvR', 'matchup_RvL', 'matchup_RvR',
            'same_hand',
        ]
        binary_cols = [c for c in binary_cols if c in df.columns]

        # Numeric columns (everything else that's numeric)
        numeric_cols = [
            c for c in df.columns
            if c not in exclude_cols
            and c not in binary_cols
            and df[c].dtype in ['float64', 'float32', 'int64', 'int32']
        ]

        return numeric_cols, binary_cols

    def fit(self, df: pd.DataFrame) -> 'MatchupPreprocessor':
        """
        Fit the preprocessor on training data.
        """
        self.numeric_columns, self.binary_columns = self.get_feature_columns(df)
        self.feature_columns = self.numeric_columns + self.binary_columns

        # Fit scaler on numeric columns (handle NaN by filling with 0 for fitting)
        numeric_data = df[self.numeric_columns].fillna(0).values
        self.scaler.fit(numeric_data)

        # Fit label encoder on outcomes
        self.label_encoder.fit(OUTCOME_CLASSES)

        self.fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        include_target: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Transform data for model input.

        Args:
            df: DataFrame with matchup features
            include_target: Whether to return target labels

        Returns:
            Tuple of (X features, y labels) or just X if include_target=False
        """
        if not self.fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        # Scale numeric features (preserve NaN for tree models)
        numeric_data = df[self.numeric_columns].values
        # For scaling, temporarily fill NaN, scale, then restore NaN
        nan_mask = np.isnan(numeric_data)
        numeric_filled = np.where(nan_mask, 0, numeric_data)
        X_numeric = self.scaler.transform(numeric_filled)
        X_numeric = np.where(nan_mask, np.nan, X_numeric)

        # Get binary features (no scaling needed)
        X_binary = df[self.binary_columns].values

        # Combine
        X = np.hstack([X_numeric, X_binary])

        if include_target:
            y = self.label_encoder.transform(df['outcome'])
            return X, y
        return X, None

    def fit_transform(
        self,
        df: pd.DataFrame,
        include_target: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Fit and transform in one step.
        """
        self.fit(df)
        return self.transform(df, include_target)

    def get_feature_names(self) -> list[str]:
        """Get ordered list of feature names."""
        return self.numeric_columns + self.binary_columns

    def get_outcome_classes(self) -> list[str]:
        """Get ordered list of outcome classes."""
        return list(self.label_encoder.classes_)

    def save(self, path: str) -> None:
        """Save preprocessor to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_columns': self.feature_columns,
                'numeric_columns': self.numeric_columns,
                'binary_columns': self.binary_columns,
                'fitted': self.fitted,
            }, f)
        print(f"Saved preprocessor to {path}")

    @classmethod
    def load(cls, path: str) -> 'MatchupPreprocessor':
        """Load preprocessor from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        preprocessor = cls()
        preprocessor.scaler = data['scaler']
        preprocessor.label_encoder = data['label_encoder']
        preprocessor.feature_columns = data['feature_columns']
        preprocessor.numeric_columns = data['numeric_columns']
        preprocessor.binary_columns = data['binary_columns']
        preprocessor.fitted = data['fitted']

        return preprocessor


def prepare_temporal_split(
    df: pd.DataFrame,
    test_date: str,
    val_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """
    Split data temporally (no random shuffle - avoid leakage).

    Args:
        df: Full matchup DataFrame
        test_date: Date string, data >= this is test set
        val_date: Optional date string for validation set

    Returns:
        Tuple of (train_df, test_df, val_df or None)
    """
    df = df.copy()
    df['game_date'] = pd.to_datetime(df['game_date'])
    test_date = pd.to_datetime(test_date)

    test_df = df[df['game_date'] >= test_date]

    if val_date is not None:
        val_date = pd.to_datetime(val_date)
        val_df = df[(df['game_date'] >= val_date) & (df['game_date'] < test_date)]
        train_df = df[df['game_date'] < val_date]
        return train_df, test_df, val_df
    else:
        train_df = df[df['game_date'] < test_date]
        return train_df, test_df, None


# Legacy class for backwards compatibility with old model code
class DataPreprocessor(MatchupPreprocessor):
    """Alias for MatchupPreprocessor for backwards compatibility."""
    pass
