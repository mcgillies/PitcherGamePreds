# Game Simulation Module - Design Document

## Overview

Simulate full games from the point a starting pitcher exits through the end of the game. Uses Monte Carlo simulation with stochastic bullpen transitions to generate probabilistic game outcomes.

**Integration point**: SP innings/exit already modeled via `markov_sim`. This module handles everything after the starter leaves.

## Architecture

```
SP Exit (existing model)
         |
         v
+------------------+     +----------------------+
| Transition Model | --> | Pitcher Selection    |
| P(role | state)  |     | P(pitcher | role,    |
+------------------+     |   team, availability)|
         |               +----------------------+
         v                         |
+------------------+               v
| Reliever Exit    |     +----------------------+
| P(exit | outs,   | <-- | At-Bat Simulation    |
|   role, state)   |     | (existing predictor) |
+------------------+     +----------------------+
         |
         v
    [Loop until game ends]
```

## Components

### 1. Role Classification (exists: `pitcher_roles.py`)
Classifies pitchers into roles based on recent usage:
- SP (Starter), CL (Closer), SU (Setup), MR (Middle), LR (Long Relief), SPEC (Specialist)

**Key decision**: Use **rolling 30-day window** per team to capture role changes mid-season.

### 2. Transition Model (exists: `transition_model.py`)
Models `P(next_role | inning, score_diff, outs, prev_role)`.

**Current state**: Empirical counting or LightGBM classifier.
**Enhancement needed**: Team-specific transition matrices to capture managerial tendencies.

### 3. Pitcher Selection Model (NEW)
Models `P(pitcher | role, team, availability)`.

**Inputs**:
- `role`: Which role is entering (from transition model)
- `team`: Team's current bullpen roster
- `availability`: Each pitcher's availability score

**Availability scoring**:
| Days since last pitched | Weight |
|------------------------|--------|
| 0 (same game)          | 0.0    |
| 1 (back-to-back)       | 0.3    |
| 2                      | 0.7    |
| 3+                     | 1.0    |
| 3 consecutive days     | 0.05   |

**Output**: Probability distribution over pitchers in that role for that team.

### 4. Reliever Exit Model (NEW)
Models `P(exit_after_this_batter | outs_recorded, role, game_state)`.

**Features**:
- `outs_recorded`: How many outs this reliever has gotten (0, 1, 2, 3, ...)
- `role`: Affects expected duration (CL ~3 outs, LR ~6+ outs, SPEC ~1-2 outs)
- `inning`: Late innings = shorter stints
- `score_diff`: Close games = more careful management
- `times_through_order`: TTO penalty kicks in for multi-inning guys

**Output**: Binary probability - does this pitcher exit after current batter?

### 5. Game Simulator (NEW)
Orchestrates the full simulation loop.

```python
class GameSimulator:
    def simulate_game(
        self,
        sp_exit_state: GameState,  # From existing SP model
        batting_lineup: List[dict],
        pitching_team: str,
        n_simulations: int = 1000,
    ) -> SimulationResults:
        """
        Run Monte Carlo simulation from SP exit to game end.
        
        Returns distribution over:
        - Final score differential
        - Total runs allowed by bullpen
        - Which relievers appeared (aggregated across sims)
        - Individual reliever stat lines
        """
```

## Data Requirements

All derived from `pitches.parquet`:

| Data | Source | Recency |
|------|--------|---------|
| Role classification | `extract_pitcher_usage_stats()` | Rolling 30 days |
| Transition probabilities | `extract_transitions()` | Rolling 30 days |
| Pitcher availability | Game logs (days since pitched) | Real-time |
| Reliever exit patterns | Pitch data (innings per appearance by role) | Rolling 30 days |
| Matchup predictions | `pitcher_profiles.csv`, `batter_profiles.csv` | Existing |

## Handling Bullpen Volatility

**Problem**: Roles change constantly (trades, injuries, hot hands, demotions).

**Solution**: Rolling window approach
1. **In-season (30+ days)**: Use last 30 days of data per team
2. **Early season (<30 days)**: 
   - Fallback to prior year for returning pitchers
   - League-average transition probabilities for unknowns
   - Do NOT use spring training data (not representative of regular season usage)
3. **Re-fit daily**: Role classifications and transition matrices update each day

**Alternative considered**: Pitcher-level "role embedding" that adapts continuously. More complex, potentially better - could add later.

## Monte Carlo Flow (Pseudocode)

```python
def simulate_bullpen_phase(sp_exit_state, lineup, team, n_sims):
    results = []
    
    for sim in range(n_sims):
        state = sp_exit_state.copy()
        appeared = {}  # pitcher_id -> stats
        prev_role = "SP"
        
        while not game_over(state):
            # 1. Sample which role enters
            role = transition_model.sample_next_role(state, prev_role)
            
            # 2. Sample which specific pitcher
            available = get_available_pitchers(team, role, appeared)
            pitcher = sample_pitcher(available, role)
            appeared[pitcher.id] = PitcherStats()
            
            # 3. Simulate this reliever's appearance
            while not reliever_exits(state, pitcher, role):
                batter = get_current_batter(lineup, state)
                outcome = simulate_at_bat(pitcher, batter)
                update_state(state, outcome)
                appeared[pitcher.id].update(outcome)
                
                if game_over(state):
                    break
            
            prev_role = role
        
        results.append(SimResult(state, appeared))
    
    return aggregate_results(results)
```

## Output Format

```python
@dataclass
class SimulationResults:
    # Aggregate stats
    mean_runs_allowed: float
    std_runs_allowed: float
    run_distribution: Dict[int, float]  # P(runs = k)
    
    # Reliever usage (aggregated across sims)
    reliever_appearances: Dict[int, RelieverUsage]
    # Each has: P(appears), expected_outs, expected_runs, expected_K
    
    # Win probability impact
    bullpen_win_prob_delta: float  # How much bullpen helps/hurts
```

## Implementation Plan

1. **Reliever Exit Model** (`reliever_exit.py`)
   - Extract historical reliever stint lengths from pitches.parquet
   - Fit P(exit | outs, role, state) model
   
2. **Pitcher Selection Model** (`pitcher_selection.py`)
   - Build team rosters with roles
   - Availability tracking from recent game logs
   - Selection probability based on role match + availability
   
3. **Game Simulator** (`simulator.py`)
   - Orchestration class
   - Integration with existing GamePredictorBinary for at-bat simulation
   
4. **Daily Data Pipeline** (`update_bullpen_data.py`)
   - Re-fit roles and transitions on rolling window
   - Update availability from yesterday's games

## Open Questions

1. **Pinch hitters**: Do we model lineup changes in late innings? (Probably not v1)
2. **Extra innings**: Ghost runner rule affects bullpen strategy
3. **Blowout detection**: Should we short-circuit simulation in 10+ run games?
4. **Leverage-based selection within role**: Currently, game state (score, inning, runners) determines which *role* enters. Should it also affect which *pitcher* within that role? E.g., prefer better setup man in higher leverage. Suggestion: skip for v1, role already encodes leverage.
