"""
Game Simulation Module

This module provides tools for simulating full baseball games,
including bullpen transition modeling and batter projections.

Components:
- extract_transitions: Extract pitcher change events from historical data
- pitcher_roles: Classify pitchers into roles (Closer, Setup, etc.)
- transition_model: Model P(next_pitcher_role | game_state)
- reliever_exit: Model P(exit | outs, role, game_state)
- pitcher_selection: Model P(pitcher | role, team, availability)
- simulator: Full game Monte Carlo simulation
"""

from .pitcher_roles import PitcherRoleClassifier
from .transition_model import BullpenTransitionModel, GameState
from .reliever_exit import RelieverExitModel, RelieverState
from .pitcher_selection import PitcherSelectionModel, PitcherAvailability, TeamBullpen
from .simulator import GameSimulator, SimulationState, AggregatedResults, create_simulator_from_data
