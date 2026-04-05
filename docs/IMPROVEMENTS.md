# Future Improvements

Tracked ideas for model and data pipeline enhancements.

---

## Arsenal & Profile Enhancements

### Rolling Arsenal Stats
**Status:** Not started  
**Priority:** Medium  
**Description:** Compute pitch characteristics (velocity, spin, movement, usage%) using a rolling window of recent starts (e.g., last 5-10 starts) instead of season-level aggregation.

**Benefits:**
- More responsive to in-season changes (mechanical adjustments, fatigue, injury effects)
- Captures mid-season pitch additions/drops faster than yearly aggregation
- Better for prediction at season boundaries (early season uses recent data, not stale prior-year)

**Tradeoffs:**
- Noisier with small sample sizes (especially for secondary pitches)
- More complex to compute and maintain
- Need to handle cold-start (beginning of season / new pitchers)

**Implementation notes:**
- Add to `_compute_pitcher_rolling()` in `preprocess.py`
- Consider separate windows: short (3-5 starts) for volatile stats, longer (10-15) for stable characteristics
- Could weight by recency within window

---

### Recency-Weighted Arsenal Aggregation
**Status:** Not started  
**Priority:** Low-Medium  
**Description:** When aggregating arsenal stats, weight recent pitches more heavily than older ones using exponential decay or similar weighting scheme.

**Benefits:**
- Smooth transition between seasons (no hard cutoff)
- Balances stability (larger sample) with freshness (recent form)
- Single profile per pitcher (simpler than rolling)

**Tradeoffs:**
- Less interpretable than pure season or rolling stats
- Need to tune decay rate (half-life parameter)
- Still requires sufficient history to be meaningful

**Implementation notes:**
- Apply weights based on days since pitch: `weight = exp(-lambda * days_ago)`
- Half-life of ~60-90 days might be reasonable starting point
- Could apply at aggregation time in notebook 02 or via custom function

---

## Model Enhancements

### Retrain with Higher num_leaves
**Status:** Not started  
**Priority:** HIGH  
**Description:** Current FLAML model selected `num_leaves=4` which severely limits prediction range, causing regression to mean.

**Symptoms:**
- Elite K pitchers (Skubal 41.7% K rate) predicted at ~23% K rate
- Model can't differentiate between elite and average pitchers
- All predictions clustered around league average

**Fix:** Retrain with constraints:
```python
automl_settings = {
    "custom_hp": {
        "lgbm": {
            "num_leaves": {"domain": tune.randint(16, 128)},  # Force higher
        }
    }
}
```

Or increase time budget to allow FLAML to explore more complex models.

---

### Pitcher Fatigue / Workload Features
**Status:** Not started  
**Priority:** Medium  
**Description:** Add features capturing recent workload that may affect performance.

**Ideas:**
- Pitches thrown in last 7/14/30 days
- Days since last start
- Innings pitched in current season
- Pitch count in current game (for within-game predictions)

---

### Ballpark Factors
**Status:** Not started  
**Priority:** Low  
**Description:** Incorporate park effects on outcomes (HR-friendly parks, pitcher parks, etc.)

**Ideas:**
- Park factors by outcome type (HR, 2B, K)
- Altitude/humidity effects on pitch movement
- Could use historical park factors from FanGraphs

---

### Umpire Tendencies
**Status:** Not started  
**Priority:** Low  
**Description:** Umpire strike zone size/shape affects K and BB rates.

**Ideas:**
- Umpire K% and BB% historical
- Zone size metrics (larger zone = more Ks)
- May not be known far in advance for predictions

---

## Game-Level Predictions

### Extend Model to Full Starting Pitcher Games
**Status:** Not started  
**Priority:** High  
**Description:** Use PA-level model to predict full game statlines for starting pitchers by aggregating predictions across expected batters faced.

**Components needed:**
1. Predict starter duration (batters faced / innings)
2. Pull opposing lineup
3. Aggregate PA predictions into game totals (K, BB, H, HR, etc.)

---

### Base-Out Run Expectancy Matrix
**Status:** Not started  
**Priority:** Medium  
**Description:** Replace simple linear run weights with a full base-out state simulation using the RE24 matrix.

**Current approach:** Multiply outcome probabilities by fixed run values (1B=0.45, HR=1.40, etc.)

**Better approach:** Simulate innings with 24 base-out states:
1. Start at state (0 outs, bases empty)
2. For each batter, get P(outcome) from model
3. Apply state transition probabilities (e.g., single with runner on 2nd → runner scores, batter on 1st)
4. Sum run expectancy changes across all state transitions weighted by probability
5. Continue until 3 outs, repeat for expected innings

**Benefits:**
- Context-dependent run values (bases loaded single worth more than bases empty single)
- More accurate run expectancy
- Could output full distributions, not just expected values

**Complexity:**
- Need to track state transition matrix for each outcome type
- Markov chain simulation across ~24 batters
- May need Monte Carlo for variance estimates

**Resources:**
- RE24 matrix: https://library.fangraphs.com/misc/re24/
- State transitions can be derived from historical Statcast data

---

### Incorporate Relievers / Full Team Games
**Status:** Not started  
**Priority:** Low (future)  
**Description:** Extend beyond starters to predict full bullpen usage and team pitching statlines.

**Challenges:**
- Reliever usage is highly variable/matchup-dependent
- Need to model manager decisions
- Handedness matchups matter more for short stints

---

## Data Pipeline

### Incremental Data Updates
**Status:** Not started  
**Priority:** Medium  
**Description:** Currently notebooks recompute everything from scratch. Add incremental updates for daily predictions.

---

### Feature Store
**Status:** Not started  
**Priority:** Low  
**Description:** Pre-compute and cache common features (rolling stats, profiles) for faster inference.

---

## Completed

### Season-Level Arsenal Separation
**Status:** Completed (2026-03-31)  
**Description:** Changed arsenal/profile aggregation from all-years combined to per-season. Profiles now joined on `(player_id, season)` to capture year-over-year arsenal changes.
