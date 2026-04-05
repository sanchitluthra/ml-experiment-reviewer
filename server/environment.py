# ─── environment.py ──────────────────────────────────────────

import uuid
import random
import sys
import os
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server import Environment
from models import MLObservation, MLState
from server.tasks import (
    generate_easy_experiment,
    generate_medium_experiment,
    generate_hard_experiment
)
from server.grader import grade


class MLExperimentEnvironment(Environment):
    """ML Experiment Reviewer Environment"""

    SUPPORTS_CONCURRENT_SESSIONS = True

    @property
    def metadata(self):
        return {
            "name": "ml-experiment-env",
            "description": "ML Experiment Reviewer Environment"
        }

    def __init__(self):
        super().__init__()
        self._episode_id         = None
        self._current_task       = None
        self._step_count         = 0
        self._current_experiment = None
        self._done               = False

    def reset(self, seed=None, episode_id=None, difficulty=None, **kwargs):
        self._reset_rubric()
        self._episode_id = episode_id or str(uuid.uuid4())[:8]
        self._step_count = 0
        self._done       = False

        if difficulty is None:
            self._current_task = random.choice(["easy", "medium", "hard"])
        else:
            self._current_task = difficulty.lower()

        if self._current_task == "easy":
            self._current_experiment = generate_easy_experiment()
        elif self._current_task == "medium":
            self._current_experiment = generate_medium_experiment()
        else:
            self._current_experiment = generate_hard_experiment()

        return MLObservation(
            experiment_data  = self._current_experiment["experiment_data"],
            task_difficulty  = self._current_task,
            task_description = self._current_experiment["description"],
            done             = False,
            reward           = 0.0
        )

    def step(self, action, timeout_s=None, **kwargs):
        # Already done?
        if self._done:
            return MLObservation(
                experiment_data  = self._current_experiment["experiment_data"],
                task_difficulty  = self._current_task,
                task_description = "Episode done! Call reset() to start new.",
                done             = True,
                reward           = 0.0
            )

        self._step_count += 1

        # Convert to correct typed action
        if self._current_task == "easy":
            from models import EasyAction
            typed_action = EasyAction(
                diagnosis = action.diagnosis,
                reason    = action.reason
            )
        elif self._current_task == "medium":
            from models import HardAction
            typed_action = HardAction(
                diagnosis     = action.diagnosis,
                issues_found  = action.issues_found or [],
                suggestions   = action.suggestions or [],
                reason        = action.reason,
                data_quality  = action.data_quality or "not provided",
                preprocessing = action.preprocessing or "not provided",
                model_check   = action.model_check or "not provided",
                overall_score = action.overall_score or 0.0
            )
        elif self._current_task == "hard":
            from models import MediumAction
            typed_action = MediumAction(
                diagnosis    = action.diagnosis,
                issues_found = action.issues_found or [],
                suggestions  = action.suggestions or [],
                reason       = action.reason
            )
        else:
            typed_action = action  # fallback

        reward     = grade(
            action         = typed_action,
            correct_answer = self._current_experiment["correct_answer"],
            difficulty     = self._current_task
        )
        self._done = True

        return MLObservation(
            experiment_data  = self._current_experiment["experiment_data"],
            task_difficulty  = self._current_task,
            task_description = self._current_experiment["description"],
            done             = True,
            reward           = reward
        )

    @property
    def state(self):
        return MLState(
            episode_id   = self._episode_id or "not_started",
            current_task = self._current_task or "not_started",
            step_count   = self._step_count
        )
