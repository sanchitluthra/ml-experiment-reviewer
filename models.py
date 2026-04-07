# ─── models.py ───────────────────────────────────────────────
# Defines all typed models for our ML Experiment Environment

from openenv.core.env_server import Action, Observation, State
from typing import List, Optional

# ─── Easy Task Action ────────────────────────────────────────

class EasyAction(Action):
    """Agent's response for Task 1 - Overfitting Detection"""
    diagnosis: str        # "overfitting" / "underfitting" / "good_fit"
    reason: str           # Why agent made this diagnosis

# ─── Medium Task Action ──────────────────────────────────────

class MediumAction(Action):
    """Agent's response for Medium Task - Hyperparameter Tuning"""
    diagnosis: str           # Overall diagnosis "overfitting" etc
    issues_found: List[str]  # List of problematic hyperparameters
    suggestions: List[str]   # List of fixes
    reason: str              # Why these changes help

# ─── Hard Task Action ────────────────────────────────────────

class HardAction(Action):
    """Agent's response for Hard Task - Full Experiment Diagnosis"""
    diagnosis: str              # Overall diagnosis
    issues_found: List[str]     # All issues found
    suggestions: List[str]      # All fixes
    reason: str                 # Why these fixes help
    data_quality: str           # Data leakage? Balanced?
    preprocessing: str          # Normalization? Split correct?
    model_check: str            # Right model for task?
    overall_score: float        # How bad is it? 0.0-1.0

# ─── Observation (What Agent Sees) ───────────────────────────

class MLObservation(Observation):
    """What we send to the agent to analyze"""
    experiment_data: dict    # All experiment numbers and config
    task_difficulty: str     # "easy" / "medium" / "hard"
    task_description: str    # Instructions for agent
    # done and reward → inherited from Observation automatically!

# ─── State (Background Info) ─────────────────────────────────

class MLState(State):
    """Inherits episode_id, current_task, step_count from State"""
    pass
# ─── Combined Action (for OpenEnv) ───────────────────────────
class MLAction(Action):
    """Combined action class for all 3 tasks"""
    # Easy fields
    diagnosis: str
    reason: str

    # Hard (Hyperparameter only/Numeric Tuning) fields
    issues_found:  Optional[List[str]] = None
    suggestions:   Optional[List[str]] = None
    reason:        Optional[str] = None

    # Medium (Full Diagnostic/Master Audit) fields
    data_quality:  Optional[str] = None
    preprocessing: Optional[str] = None
    model_check:   Optional[str] = None
    overall_score: Optional[float] = None
