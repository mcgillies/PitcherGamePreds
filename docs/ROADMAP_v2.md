# Roadmap: Advanced Pitcher-Batter Matchup Model

## Vision

Build a **pitch-profile aware** matchup model that predicts plate appearance outcomes based on how batters perform against **similar types of pitchers** - not just handedness, but velocity, movement, pitch mix, and command.

Core insight: A batter who struggles against high-velocity fastballs will struggle against any high-velo pitcher, regardless of team or name. We model the **pitch characteristics**, not just the pitcher identity.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    PITCH PROFILE FEATURES                        │
│  Velocity, Movement, Spin, Location, Pitch Mix, Whiff%, CSW%    │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                 BATTER PERFORMANCE vs PROFILES                   │
│  "How does this batter perform against pitchers with these      │
│   characteristics?" (computed from historical PAs)               │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MATCHUP PREDICTION MODEL                      │
│  Input: Pitcher profile + Batter profile + Context              │
│  Output: P(K), P(BB), P(1B), P(2B), P(3B), P(HR), P(Out)        │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GAME AGGREGATION                              │
│  For each batter in lineup: Matchup prediction × Expected PAs   │
│  Sum across lineup → Full game prediction                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Statcast Data Deep Dive

### 1.1 Available Pitch-Level Features

Statcast provides incredibly rich pitch-by-pitch data. These are the key columns:

**Pitch Characteristics:**
| Column | Description | Use |
|--------|-------------|-----|
| `release_speed` | Pitch velocity (mph) | Primary velo metric |
| `effective_speed` | Perceived velocity | Accounts for extension |
| `pfx_x` | Horizontal movement (inches) | Break direction |
| `pfx_z` | Vertical movement (inches) | Drop/rise |
| `release_spin_rate` | Spin (rpm) | Spin rate |
| `spin_axis` | Spin direction (degrees) | Spin type |
| `release_extension` | Extension toward plate (ft) | Deception |
| `release_pos_x` | Horizontal release point | Arm slot |
| `release_pos_z` | Vertical release point | Arm slot |
| `plate_x` | Horizontal location | Command |
| `plate_z` | Vertical location | Command |
| `zone` | Strike zone region (1-14) | Location bucket |

**Pitch Type:**
| Column | Description |
|--------|-------------|
| `pitch_type` | FF, SI, SL, CU, CH, FC, etc. |
| `pitch_name` | Full name (4-Seam Fastball, etc.) |

**Outcome Data:**
| Column | Description |
|--------|-------------|
| `description` | Pitch result (called_strike, swinging_strike, ball, hit_into_play, etc.) |
| `events` | PA result if PA ended (strikeout, single, home_run, etc.) |
| `type` | S (strike), B (ball), X (in play) |
| `launch_speed` | Exit velocity if batted |
| `launch_angle` | Launch angle if batted |
| `estimated_ba_using_speedangle` | xBA |
| `estimated_woba_using_speedangle` | xwOBA |

**Context:**
| Column | Description |
|--------|-------------|
| `balls`, `strikes` | Count |
| `outs_when_up` | Outs |
| `inning` | Inning |
| `stand` | Batter stance (L/R) |
| `p_throws` | Pitcher hand (L/R) |

### 1.2 Derived Pitcher Profile Metrics

From pitch-level data, we compute **pitcher profile features** that describe their arsenal:

```python
PITCHER_PROFILE_FEATURES = {
    # Fastball characteristics
    'fb_velo_avg': 'Average fastball velocity',
    'fb_velo_max': 'Max fastball velocity',
    'fb_spin_avg': 'Average fastball spin rate',
    'fb_vmov_avg': 'Average fastball vertical movement',
    'fb_hmov_avg': 'Average fastball horizontal movement',
    'fb_usage_pct': 'Fastball usage percentage',

    # Breaking ball characteristics (SL, CU, KC)
    'brk_velo_avg': 'Average breaking ball velocity',
    'brk_spin_avg': 'Average breaking ball spin',
    'brk_vmov_avg': 'Breaking ball vertical drop',
    'brk_hmov_avg': 'Breaking ball horizontal sweep',
    'brk_usage_pct': 'Breaking ball usage percentage',

    # Offspeed characteristics (CH, FS)
    'off_velo_avg': 'Average offspeed velocity',
    'off_velo_diff': 'Velo differential from fastball',
    'off_vmov_avg': 'Offspeed vertical movement',
    'off_usage_pct': 'Offspeed usage percentage',

    # Command/Control metrics
    'zone_pct': 'Percentage of pitches in zone',
    'edge_pct': 'Percentage on edges (shadow zone)',
    'chase_pct_induced': 'Chase rate induced',
    'first_pitch_strike_pct': 'First pitch strike rate',

    # Swing/Miss metrics
    'whiff_pct': 'Swinging strike rate (swings that miss)',
    'csw_pct': 'Called strike + whiff percentage',
    'swstr_pct': 'Swinging strike per pitch',
    'contact_pct': 'Contact rate when swung at',

    # Platoon
    'p_throws': 'Pitcher handedness (L/R)',

    # Extension/Deception
    'extension_avg': 'Average release extension',
    'vaa_avg': 'Vertical approach angle',
}
```

### 1.3 Computing Pitcher Profiles

```python
def compute_pitcher_profile(
    pitches: pd.DataFrame,
    pitcher_id: int,
    as_of_date: str = None,
) -> dict:
    """
    Compute pitcher profile from pitch-level Statcast data.

    Args:
        pitches: Statcast pitch data
        pitcher_id: MLB player ID
        as_of_date: Only use pitches before this date (for leakage prevention)

    Returns:
        Dictionary of profile features
    """
    # Filter to this pitcher
    p = pitches[pitches['pitcher'] == pitcher_id].copy()

    if as_of_date:
        p = p[p['game_date'] < as_of_date]

    if len(p) < 100:  # Minimum sample
        return None

    profile = {}

    # Classify pitch types
    fastballs = ['FF', 'SI', 'FC']  # 4-seam, sinker, cutter
    breaking = ['SL', 'CU', 'KC', 'SV']  # slider, curve, knuckle-curve, sweeper
    offspeed = ['CH', 'FS', 'SC']  # changeup, splitter, screwball

    fb = p[p['pitch_type'].isin(fastballs)]
    brk = p[p['pitch_type'].isin(breaking)]
    off = p[p['pitch_type'].isin(offspeed)]

    # Fastball metrics
    if len(fb) > 20:
        profile['fb_velo_avg'] = fb['release_speed'].mean()
        profile['fb_velo_max'] = fb['release_speed'].max()
        profile['fb_spin_avg'] = fb['release_spin_rate'].mean()
        profile['fb_vmov_avg'] = fb['pfx_z'].mean()
        profile['fb_hmov_avg'] = fb['pfx_x'].mean()
    profile['fb_usage_pct'] = len(fb) / len(p) if len(p) > 0 else 0

    # Breaking ball metrics
    if len(brk) > 20:
        profile['brk_velo_avg'] = brk['release_speed'].mean()
        profile['brk_spin_avg'] = brk['release_spin_rate'].mean()
        profile['brk_vmov_avg'] = brk['pfx_z'].mean()
        profile['brk_hmov_avg'] = brk['pfx_x'].mean()
    profile['brk_usage_pct'] = len(brk) / len(p) if len(p) > 0 else 0

    # Offspeed metrics
    if len(off) > 20:
        profile['off_velo_avg'] = off['release_speed'].mean()
        profile['off_vmov_avg'] = off['pfx_z'].mean()
        if 'fb_velo_avg' in profile:
            profile['off_velo_diff'] = profile['fb_velo_avg'] - profile['off_velo_avg']
    profile['off_usage_pct'] = len(off) / len(p) if len(p) > 0 else 0

    # Command metrics
    profile['zone_pct'] = (p['zone'] <= 9).mean()  # Zones 1-9 are in strike zone

    # Whiff/CSW metrics
    swings = p[p['description'].isin([
        'swinging_strike', 'swinging_strike_blocked',
        'foul', 'foul_tip', 'hit_into_play', 'foul_bunt'
    ])]
    whiffs = p[p['description'].isin(['swinging_strike', 'swinging_strike_blocked'])]
    called_strikes = p[p['description'] == 'called_strike']

    profile['whiff_pct'] = len(whiffs) / len(swings) if len(swings) > 0 else 0
    profile['csw_pct'] = (len(whiffs) + len(called_strikes)) / len(p) if len(p) > 0 else 0
    profile['swstr_pct'] = len(whiffs) / len(p) if len(p) > 0 else 0

    # Chase rate (swings outside zone)
    outside_zone = p[p['zone'] > 9]
    outside_swings = outside_zone[outside_zone['description'].isin([
        'swinging_strike', 'swinging_strike_blocked',
        'foul', 'foul_tip', 'hit_into_play'
    ])]
    profile['chase_pct_induced'] = len(outside_swings) / len(outside_zone) if len(outside_zone) > 0 else 0

    # Extension
    profile['extension_avg'] = p['release_extension'].mean()

    # Handedness
    profile['p_throws'] = p['p_throws'].iloc[0]

    return profile
```

### 1.4 Pitcher Archetypes (Optional Enhancement)

Instead of using raw metrics, we can cluster pitchers into archetypes:

```python
PITCHER_ARCHETYPES = {
    'power_fb': 'High velocity (96+), fastball dominant',
    'sinker_ground': 'Heavy sinker, induces ground balls',
    'spin_master': 'Elite spin rates, vertical FB + sharp curve',
    'sweeper_slider': 'Horizontal breaking ball specialist',
    'changeup_artist': 'Big velo differential, offspeed heavy',
    'command_pitcher': 'Below avg velo but elite command/location',
    'unicorn': 'Does everything well',
}

def cluster_pitchers(profiles: pd.DataFrame, n_clusters: int = 8) -> pd.DataFrame:
    """Cluster pitchers into archetypes using k-means or similar."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    features = ['fb_velo_avg', 'fb_spin_avg', 'brk_hmov_avg', 'whiff_pct', ...]
    X = StandardScaler().fit_transform(profiles[features])

    kmeans = KMeans(n_clusters=n_clusters)
    profiles['archetype'] = kmeans.fit_predict(X)

    return profiles
```

---

## Phase 2: Batter Performance Profiles

### 2.1 The Key Insight

Instead of asking "how does this batter do vs LHP?", we ask:

**"How does this batter perform against pitchers with these characteristics?"**

Examples:
- Batter A crushes high fastballs but chases sliders low
- Batter B can't catch up to 97+ but handles offspeed well
- Batter C struggles vs horizontal breaking balls

### 2.2 Batter vs Pitch-Type Performance

```python
def compute_batter_pitch_type_stats(
    pitches: pd.DataFrame,
    batter_id: int,
    as_of_date: str = None,
) -> dict:
    """
    Compute batter's performance vs each pitch type.
    """
    b = pitches[pitches['batter'] == batter_id].copy()

    if as_of_date:
        b = b[b['game_date'] < as_of_date]

    stats = {}

    for pitch_group, types in [
        ('fastball', ['FF', 'SI', 'FC']),
        ('breaking', ['SL', 'CU', 'KC', 'SV']),
        ('offspeed', ['CH', 'FS']),
    ]:
        group = b[b['pitch_type'].isin(types)]

        if len(group) < 30:
            continue

        # Whiff rate on this pitch type
        swings = group[group['description'].str.contains('swing|foul|hit_into_play', na=False)]
        whiffs = group[group['description'].str.contains('swinging_strike', na=False)]
        stats[f'{pitch_group}_whiff_rate'] = len(whiffs) / len(swings) if len(swings) > 0 else None

        # Called strike rate (taking strikes)
        takes = group[group['description'].isin(['called_strike', 'ball'])]
        called_strikes = group[group['description'] == 'called_strike']
        stats[f'{pitch_group}_cs_rate'] = len(called_strikes) / len(takes) if len(takes) > 0 else None

        # In-play results
        in_play = group[group['type'] == 'X']
        if len(in_play) > 10:
            stats[f'{pitch_group}_xwoba'] = in_play['estimated_woba_using_speedangle'].mean()
            stats[f'{pitch_group}_exit_velo'] = in_play['launch_speed'].mean()

    return stats
```

### 2.3 Batter vs Velocity Buckets

```python
def compute_batter_velo_stats(
    pitches: pd.DataFrame,
    batter_id: int,
) -> dict:
    """
    Compute batter's performance vs velocity ranges.
    """
    b = pitches[pitches['batter'] == batter_id].copy()

    stats = {}

    velo_buckets = [
        ('velo_90_93', 90, 93),
        ('velo_94_96', 94, 96),
        ('velo_97_plus', 97, 105),
    ]

    for name, low, high in velo_buckets:
        bucket = b[(b['release_speed'] >= low) & (b['release_speed'] < high)]

        if len(bucket) < 30:
            continue

        # Swing decisions
        swings = bucket[bucket['description'].str.contains('swing|foul|hit_into_play', na=False)]
        whiffs = bucket[bucket['description'].str.contains('swinging_strike', na=False)]

        stats[f'{name}_whiff_rate'] = len(whiffs) / len(swings) if len(swings) > 0 else None
        stats[f'{name}_swing_rate'] = len(swings) / len(bucket)

        # Contact quality
        in_play = bucket[bucket['type'] == 'X']
        if len(in_play) > 5:
            stats[f'{name}_exit_velo'] = in_play['launch_speed'].mean()

    return stats
```

### 2.4 Batter vs Movement Profiles

```python
def compute_batter_movement_stats(
    pitches: pd.DataFrame,
    batter_id: int,
) -> dict:
    """
    Compute batter's performance vs different movement profiles.
    """
    b = pitches[pitches['batter'] == batter_id].copy()

    stats = {}

    # High vertical movement (rising fastballs)
    high_vmov = b[b['pfx_z'] > 14]  # inches of rise
    if len(high_vmov) > 30:
        swings = high_vmov[high_vmov['description'].str.contains('swing|foul|hit', na=False)]
        whiffs = high_vmov[high_vmov['description'].str.contains('swinging_strike', na=False)]
        stats['high_rise_whiff_rate'] = len(whiffs) / len(swings) if len(swings) > 0 else None

    # Heavy horizontal break (sweepers)
    heavy_sweep = b[abs(b['pfx_x']) > 12]
    if len(heavy_sweep) > 30:
        swings = heavy_sweep[heavy_sweep['description'].str.contains('swing|foul|hit', na=False)]
        whiffs = heavy_sweep[heavy_sweep['description'].str.contains('swinging_strike', na=False)]
        stats['heavy_sweep_whiff_rate'] = len(whiffs) / len(swings) if len(swings) > 0 else None

    # Heavy vertical drop (curves, splitters)
    heavy_drop = b[b['pfx_z'] < -6]
    if len(heavy_drop) > 30:
        swings = heavy_drop[heavy_drop['description'].str.contains('swing|foul|hit', na=False)]
        whiffs = heavy_drop[heavy_drop['description'].str.contains('swinging_strike', na=False)]
        stats['heavy_drop_whiff_rate'] = len(whiffs) / len(swings) if len(swings) > 0 else None

    return stats
```

### 2.5 Full Batter Profile

```python
def compute_batter_profile(
    pitches: pd.DataFrame,
    batter_id: int,
    as_of_date: str = None,
) -> dict:
    """
    Compute comprehensive batter profile for matchup prediction.
    """
    profile = {}

    # Basic info
    b = pitches[pitches['batter'] == batter_id]
    if as_of_date:
        b = b[b['game_date'] < as_of_date]

    profile['stand'] = b['stand'].iloc[0] if len(b) > 0 else None

    # Overall plate discipline
    all_pitches = len(b)
    swings = b[b['description'].str.contains('swing|foul|hit_into_play', na=False)]
    whiffs = b[b['description'].str.contains('swinging_strike', na=False)]
    takes = b[b['description'].isin(['called_strike', 'ball'])]
    called_strikes = b[b['description'] == 'called_strike']

    profile['swing_pct'] = len(swings) / all_pitches if all_pitches > 0 else None
    profile['whiff_pct'] = len(whiffs) / len(swings) if len(swings) > 0 else None
    profile['contact_pct'] = 1 - profile['whiff_pct'] if profile['whiff_pct'] else None
    profile['csw_pct_against'] = (len(whiffs) + len(called_strikes)) / all_pitches if all_pitches > 0 else None

    # Zone discipline
    in_zone = b[b['zone'] <= 9]
    out_zone = b[b['zone'] > 9]
    zone_swings = in_zone[in_zone['description'].str.contains('swing|foul|hit', na=False)]
    chase_swings = out_zone[out_zone['description'].str.contains('swing|foul|hit', na=False)]

    profile['zone_swing_pct'] = len(zone_swings) / len(in_zone) if len(in_zone) > 0 else None
    profile['chase_pct'] = len(chase_swings) / len(out_zone) if len(out_zone) > 0 else None

    # Batted ball quality
    in_play = b[b['type'] == 'X']
    if len(in_play) > 20:
        profile['avg_exit_velo'] = in_play['launch_speed'].mean()
        profile['avg_launch_angle'] = in_play['launch_angle'].mean()
        profile['xwoba'] = in_play['estimated_woba_using_speedangle'].mean()
        profile['xba'] = in_play['estimated_ba_using_speedangle'].mean()
        profile['barrel_pct'] = (
            (in_play['launch_speed'] >= 98) &
            (in_play['launch_angle'].between(26, 30))
        ).mean()

    # Add pitch-type specific stats
    profile.update(compute_batter_pitch_type_stats(pitches, batter_id, as_of_date))

    # Add velocity-specific stats
    profile.update(compute_batter_velo_stats(pitches, batter_id))

    # Add movement-specific stats
    profile.update(compute_batter_movement_stats(pitches, batter_id))

    return profile
```

---

## Phase 3: Matchup Feature Engineering

### 3.1 Combining Pitcher and Batter Profiles

The magic happens when we combine profiles to create **matchup-specific features**:

```python
def compute_matchup_features(
    pitcher_profile: dict,
    batter_profile: dict,
) -> dict:
    """
    Compute features that capture the specific matchup dynamics.
    """
    features = {}

    # === Direct profile features ===
    # Include raw pitcher stats
    for k, v in pitcher_profile.items():
        features[f'p_{k}'] = v

    # Include raw batter stats
    for k, v in batter_profile.items():
        features[f'b_{k}'] = v

    # === Interaction features ===

    # Velocity matchup: pitcher velo vs batter's performance against that velo
    if pitcher_profile.get('fb_velo_avg'):
        velo = pitcher_profile['fb_velo_avg']
        if velo >= 97:
            features['batter_whiff_at_this_velo'] = batter_profile.get('velo_97_plus_whiff_rate')
        elif velo >= 94:
            features['batter_whiff_at_this_velo'] = batter_profile.get('velo_94_96_whiff_rate')
        else:
            features['batter_whiff_at_this_velo'] = batter_profile.get('velo_90_93_whiff_rate')

    # Breaking ball matchup
    if pitcher_profile.get('brk_usage_pct', 0) > 0.20:
        features['batter_breaking_whiff'] = batter_profile.get('breaking_whiff_rate')
        features['batter_vs_heavy_sweep'] = batter_profile.get('heavy_sweep_whiff_rate')

    # Offspeed matchup
    if pitcher_profile.get('off_usage_pct', 0) > 0.15:
        features['batter_offspeed_whiff'] = batter_profile.get('offspeed_whiff_rate')

    # Whiff potential: pitcher whiff rate × batter whiff rate
    if pitcher_profile.get('whiff_pct') and batter_profile.get('whiff_pct'):
        features['whiff_potential'] = (
            pitcher_profile['whiff_pct'] * batter_profile['whiff_pct']
        )

    # CSW matchup
    if pitcher_profile.get('csw_pct') and batter_profile.get('csw_pct_against'):
        features['csw_matchup'] = (
            pitcher_profile['csw_pct'] + batter_profile['csw_pct_against']
        ) / 2

    # Platoon advantage
    features['same_hand'] = int(
        pitcher_profile.get('p_throws') == batter_profile.get('stand')
    )

    # Chase matchup: pitcher chase induced × batter chase rate
    if pitcher_profile.get('chase_pct_induced') and batter_profile.get('chase_pct'):
        features['chase_matchup'] = (
            pitcher_profile['chase_pct_induced'] * batter_profile['chase_pct']
        )

    return features
```

### 3.2 Full Feature List

```python
MATCHUP_FEATURES = {
    # Pitcher arsenal
    'p_fb_velo_avg': 'Pitcher fastball velocity',
    'p_fb_spin_avg': 'Pitcher fastball spin',
    'p_fb_vmov_avg': 'Pitcher FB vertical movement',
    'p_brk_velo_avg': 'Pitcher breaking ball velocity',
    'p_brk_hmov_avg': 'Pitcher breaking ball sweep',
    'p_brk_vmov_avg': 'Pitcher breaking ball drop',
    'p_off_velo_diff': 'Pitcher velo differential',
    'p_fb_usage_pct': 'Fastball usage',
    'p_brk_usage_pct': 'Breaking ball usage',
    'p_off_usage_pct': 'Offspeed usage',
    'p_whiff_pct': 'Pitcher whiff rate',
    'p_csw_pct': 'Pitcher CSW%',
    'p_zone_pct': 'Pitcher zone rate',
    'p_chase_pct_induced': 'Pitcher chase induced',
    'p_extension_avg': 'Pitcher extension',

    # Batter tendencies
    'b_whiff_pct': 'Batter whiff rate',
    'b_contact_pct': 'Batter contact rate',
    'b_chase_pct': 'Batter chase rate',
    'b_zone_swing_pct': 'Batter zone swing rate',
    'b_avg_exit_velo': 'Batter avg exit velocity',
    'b_xwoba': 'Batter expected wOBA',
    'b_barrel_pct': 'Batter barrel rate',

    # Batter vs pitch types
    'b_fastball_whiff_rate': 'Batter whiff on fastballs',
    'b_breaking_whiff_rate': 'Batter whiff on breaking',
    'b_offspeed_whiff_rate': 'Batter whiff on offspeed',
    'b_fastball_xwoba': 'Batter xwOBA on fastballs',
    'b_breaking_xwoba': 'Batter xwOBA on breaking',

    # Batter vs velocity
    'b_velo_97_plus_whiff_rate': 'Batter whiff vs 97+',
    'b_velo_94_96_whiff_rate': 'Batter whiff vs 94-96',

    # Batter vs movement
    'b_high_rise_whiff_rate': 'Batter whiff vs riding FB',
    'b_heavy_sweep_whiff_rate': 'Batter whiff vs sweepers',
    'b_heavy_drop_whiff_rate': 'Batter whiff vs droppers',

    # Matchup interactions
    'whiff_potential': 'Pitcher whiff × Batter whiff',
    'csw_matchup': 'Combined CSW metric',
    'chase_matchup': 'Pitcher chase × Batter chase',
    'batter_whiff_at_this_velo': 'Batter whiff at pitcher velo',
    'same_hand': 'Same handedness (platoon)',

    # Context
    'lineup_position': 'Batting order position (1-9)',
    'is_home': 'Batter is home team',
}
```

---

## Phase 4: Training Data Construction

### 4.1 Ground Truth: PA Outcomes

From Statcast, we get actual outcomes for every plate appearance:

```python
PA_OUTCOMES = {
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
    'catcher_interf': 'OTHER',
    # ... etc
}

# Simplified outcome categories for prediction
OUTCOME_CATEGORIES = ['K', 'BB', '1B', '2B', '3B', 'HR', 'OUT']
```

### 4.2 Building Training Dataset

```python
def build_training_data(
    start_date: str,
    end_date: str,
    min_pitcher_pitches: int = 500,
    min_batter_pitches: int = 200,
) -> pd.DataFrame:
    """
    Build PA-level training dataset with matchup features.

    Each row = one plate appearance with:
    - Pitcher profile features (as of game date)
    - Batter profile features (as of game date)
    - Matchup interaction features
    - Actual outcome
    """
    from pybaseball import statcast

    print(f"Loading Statcast data from {start_date} to {end_date}...")
    pitches = statcast(start_dt=start_date, end_dt=end_date)

    # Get completed plate appearances
    pas = pitches[pitches['events'].notna()].copy()
    print(f"Total plate appearances: {len(pas)}")

    # Map outcomes to categories
    pas['outcome'] = pas['events'].map(PA_OUTCOMES)
    pas = pas[pas['outcome'].isin(OUTCOME_CATEGORIES)]

    # Get unique pitchers and batters
    pitchers = pas['pitcher'].unique()
    batters = pas['batter'].unique()

    print(f"Unique pitchers: {len(pitchers)}")
    print(f"Unique batters: {len(batters)}")

    # Compute profiles (this is the slow part - cache results)
    print("Computing pitcher profiles...")
    pitcher_profiles = {}
    for pid in tqdm(pitchers):
        pitcher_profiles[pid] = compute_pitcher_profile(pitches, pid)

    print("Computing batter profiles...")
    batter_profiles = {}
    for bid in tqdm(batters):
        batter_profiles[bid] = compute_batter_profile(pitches, bid)

    # Build feature rows
    print("Building feature matrix...")
    rows = []

    for _, pa in tqdm(pas.iterrows(), total=len(pas)):
        pitcher_id = pa['pitcher']
        batter_id = pa['batter']
        game_date = pa['game_date']

        # Get profiles (ideally lagged to avoid leakage - implement as_of_date)
        p_profile = pitcher_profiles.get(pitcher_id)
        b_profile = batter_profiles.get(batter_id)

        if p_profile is None or b_profile is None:
            continue

        # Compute matchup features
        features = compute_matchup_features(p_profile, b_profile)

        # Add identifiers and outcome
        features['game_pk'] = pa['game_pk']
        features['game_date'] = game_date
        features['pitcher_id'] = pitcher_id
        features['batter_id'] = batter_id
        features['outcome'] = pa['outcome']

        rows.append(features)

    df = pd.DataFrame(rows)
    print(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")

    return df
```

### 4.3 Handling Data Leakage

**Critical**: We must ensure features only use data BEFORE the plate appearance:

```python
def compute_lagged_profiles(
    all_pitches: pd.DataFrame,
    game_date: str,
    lookback_days: int = 365,
) -> tuple[dict, dict]:
    """
    Compute profiles using only historical data.

    Args:
        all_pitches: Full Statcast dataset
        game_date: Date of the PA we're predicting
        lookback_days: How far back to look for profile data

    Returns:
        Tuple of (pitcher_profiles, batter_profiles) dicts
    """
    cutoff_date = pd.to_datetime(game_date)
    start_date = cutoff_date - pd.Timedelta(days=lookback_days)

    # Filter to historical pitches only
    historical = all_pitches[
        (all_pitches['game_date'] < cutoff_date) &
        (all_pitches['game_date'] >= start_date)
    ]

    # Compute profiles from historical data only
    # ... (same logic as before)
```

**Practical approach**: Pre-compute profiles at regular intervals (weekly) and use the most recent profile before each game date.

---

## Phase 5: Model Architecture

### 5.1 Multi-Class Classification

Predict probability distribution over outcomes:

```python
import tensorflow as tf
from tensorflow import keras

def build_matchup_model(
    n_features: int,
    n_outcomes: int = 7,  # K, BB, 1B, 2B, 3B, HR, OUT
) -> keras.Model:
    """
    Build neural network for PA outcome prediction.
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(n_features,)),

        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),

        # Output: probability for each outcome
        keras.layers.Dense(n_outcomes, activation='softmax'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model
```

### 5.2 Alternative: Gradient Boosting

```python
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

def build_xgb_model() -> xgb.XGBClassifier:
    """XGBoost multi-class classifier for PA outcomes."""
    return xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=7,
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
    )
```

### 5.3 Embeddings for Sparse Features (Advanced)

For handling pitcher/batter identity as features (collaborative filtering style):

```python
def build_embedding_model(
    n_pitchers: int,
    n_batters: int,
    n_features: int,
    embedding_dim: int = 32,
) -> keras.Model:
    """
    Model with learned pitcher/batter embeddings + profile features.
    """
    # Inputs
    pitcher_input = keras.layers.Input(shape=(1,), name='pitcher_id')
    batter_input = keras.layers.Input(shape=(1,), name='batter_id')
    features_input = keras.layers.Input(shape=(n_features,), name='features')

    # Embeddings
    pitcher_emb = keras.layers.Embedding(n_pitchers, embedding_dim)(pitcher_input)
    pitcher_emb = keras.layers.Flatten()(pitcher_emb)

    batter_emb = keras.layers.Embedding(n_batters, embedding_dim)(batter_input)
    batter_emb = keras.layers.Flatten()(batter_emb)

    # Combine
    combined = keras.layers.Concatenate()([pitcher_emb, batter_emb, features_input])

    # Dense layers
    x = keras.layers.Dense(256, activation='relu')(combined)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(64, activation='relu')(x)

    output = keras.layers.Dense(7, activation='softmax')(x)

    return keras.Model(
        inputs=[pitcher_input, batter_input, features_input],
        outputs=output,
    )
```

---

## Phase 6: Game-Level Aggregation

### 6.1 From PA Predictions to Game Predictions

```python
def predict_game(
    model,
    pitcher_id: int,
    lineup: list[int],  # List of 9 batter IDs in order
    pitcher_profile: dict,
    batter_profiles: dict[int, dict],
    expected_ip: float = 6.0,
) -> dict:
    """
    Predict full game outcomes by aggregating matchup predictions.

    Args:
        model: Trained matchup model
        pitcher_id: Starting pitcher
        lineup: Opponent lineup (9 batter IDs)
        pitcher_profile: Pitcher's profile features
        batter_profiles: Dict of batter_id -> profile
        expected_ip: Projected innings pitched

    Returns:
        Dict with expected K, BB, H, HR, etc.
    """
    # Expected PAs by lineup position (9-inning baseline)
    PA_BY_POSITION = {1: 4.8, 2: 4.7, 3: 4.6, 4: 4.4, 5: 4.3, 6: 4.1, 7: 4.0, 8: 3.8, 9: 3.7}

    # Scale by expected IP
    ip_scale = expected_ip / 9.0

    game_totals = {
        'K': 0, 'BB': 0, '1B': 0, '2B': 0, '3B': 0, 'HR': 0, 'OUT': 0
    }

    for position, batter_id in enumerate(lineup, start=1):
        # Get batter profile
        b_profile = batter_profiles.get(batter_id)
        if b_profile is None:
            continue

        # Compute matchup features
        features = compute_matchup_features(pitcher_profile, b_profile)
        features['lineup_position'] = position

        # Predict outcome probabilities
        X = prepare_features(features)  # Convert to model input format
        probs = model.predict(X)[0]  # [P(K), P(BB), P(1B), P(2B), P(3B), P(HR), P(OUT)]

        # Expected PAs for this batter
        expected_pa = PA_BY_POSITION[position] * ip_scale

        # Accumulate expected outcomes
        for i, outcome in enumerate(['K', 'BB', '1B', '2B', '3B', 'HR', 'OUT']):
            game_totals[outcome] += probs[i] * expected_pa

    # Derived stats
    game_totals['H'] = game_totals['1B'] + game_totals['2B'] + game_totals['3B'] + game_totals['HR']
    game_totals['IP'] = expected_ip

    return game_totals
```

### 6.2 Innings Pitched Model

```python
def build_ip_model(features: pd.DataFrame, target: pd.Series) -> keras.Model:
    """
    Predict innings pitched for a start.

    Features:
    - Recent IP averages (roll3, roll5)
    - Season IP average
    - Pitch count tendencies
    - Team bullpen usage patterns
    - Game context (home/away, opponent strength)
    - Rest days
    """
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1),  # Regression output
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
```

---

## Phase 7: Implementation Timeline

### Week 1-2: Data Infrastructure

- [ ] **Statcast data pipeline**
  - [ ] Function to download and store pitch-level data
  - [ ] Incremental updates (append new games)
  - [ ] Storage: Parquet files partitioned by month

- [ ] **Profile computation**
  - [ ] `compute_pitcher_profile()` function
  - [ ] `compute_batter_profile()` function
  - [ ] Caching layer (don't recompute every time)

- [ ] **Notebook: `02_statcast_exploration.ipynb`**
  - [ ] Explore pitch-level columns
  - [ ] Visualize pitch profiles
  - [ ] Test profile computations

### Week 3-4: Feature Engineering

- [ ] **Matchup features**
  - [ ] Implement `compute_matchup_features()`
  - [ ] Handle missing values (sparse profiles)
  - [ ] Feature normalization

- [ ] **Lagging logic**
  - [ ] Ensure no data leakage
  - [ ] Pre-compute weekly profile snapshots

- [ ] **Notebook: `03_matchup_features.ipynb`**
  - [ ] Feature distributions
  - [ ] Correlation analysis
  - [ ] Feature importance preview

### Week 5-6: Model Training

- [ ] **Training dataset**
  - [ ] Build PA-level dataset with features
  - [ ] Train/val/test split (temporal)

- [ ] **Model experiments**
  - [ ] Neural network approach
  - [ ] XGBoost comparison
  - [ ] Hyperparameter tuning

- [ ] **Evaluation**
  - [ ] Multi-class metrics (log loss, accuracy)
  - [ ] Per-outcome calibration
  - [ ] Comparison to baselines

- [ ] **Notebook: `04_matchup_model.ipynb`**

### Week 7-8: Game Aggregation

- [ ] **IP prediction model**
- [ ] **Aggregation logic**
- [ ] **Full game predictions**
- [ ] **Comparison to v1 model**

### Week 9-10: Production Pipeline

- [ ] **Lineup integration**
  - [ ] Projected lineup fetching
  - [ ] Handle lineup uncertainty

- [ ] **Daily prediction workflow**
  - [ ] Script to generate predictions
  - [ ] Output format for betting analysis

---

## Appendix A: Statcast Pitch Types

```python
PITCH_TYPES = {
    # Fastballs
    'FF': '4-Seam Fastball',
    'SI': 'Sinker',
    'FC': 'Cutter',

    # Breaking balls
    'SL': 'Slider',
    'CU': 'Curveball',
    'KC': 'Knuckle Curve',
    'SV': 'Sweeper',
    'ST': 'Sweeping Curve',

    # Offspeed
    'CH': 'Changeup',
    'FS': 'Splitter',
    'SC': 'Screwball',
    'FO': 'Forkball',

    # Other
    'KN': 'Knuckleball',
    'EP': 'Eephus',
    'CS': 'Slow Curve',
    'PO': 'Pitch Out',
    'FA': 'Other Fastball',
}
```

## Appendix B: Key Statcast Queries

```python
from pybaseball import statcast, statcast_pitcher, statcast_batter

# All pitches in date range
all_pitches = statcast(start_dt='2024-04-01', end_dt='2024-09-30')

# Specific pitcher's pitches
cole_pitches = statcast_pitcher(start_dt='2024-04-01', end_dt='2024-09-30', player_id=543037)

# Specific batter's PAs
judge_pas = statcast_batter(start_dt='2024-04-01', end_dt='2024-09-30', player_id=592450)

# Get just completed PAs (where event occurred)
pas_only = all_pitches[all_pitches['events'].notna()]

# Get just swings
swings = all_pitches[all_pitches['description'].str.contains('swing|foul|hit_into_play', na=False)]
```

## Appendix C: Example Pitcher Profiles

```python
# Example: High-velo power pitcher (like Gerrit Cole)
cole_profile = {
    'fb_velo_avg': 96.8,
    'fb_spin_avg': 2550,
    'fb_vmov_avg': 15.2,
    'fb_usage_pct': 0.48,
    'brk_velo_avg': 84.5,
    'brk_hmov_avg': -4.2,
    'brk_usage_pct': 0.32,
    'whiff_pct': 0.32,
    'csw_pct': 0.31,
    'zone_pct': 0.44,
    'p_throws': 'R',
}

# Example: Soft-tossing command pitcher (like Kyle Hendricks)
hendricks_profile = {
    'fb_velo_avg': 87.5,
    'fb_spin_avg': 1980,
    'fb_vmov_avg': 8.5,
    'fb_usage_pct': 0.30,
    'off_velo_avg': 78.5,
    'off_velo_diff': 9.0,
    'off_usage_pct': 0.35,
    'whiff_pct': 0.18,
    'csw_pct': 0.29,
    'zone_pct': 0.52,
    'p_throws': 'R',
}
```

---

## Summary

This approach moves from simple aggregates to **pitch-characteristic-aware matchup modeling**:

1. **Pitcher profiles** capture arsenal (velocity, movement, spin, pitch mix)
2. **Batter profiles** capture tendencies (whiff rates by pitch type, velocity, movement)
3. **Matchup features** combine profiles to predict specific interactions
4. **PA-level model** predicts outcome probabilities for each plate appearance
5. **Game aggregation** sums across lineup weighted by expected PAs

This is significantly more sophisticated than L/R splits and should capture the true matchup dynamics that drive outcomes.
