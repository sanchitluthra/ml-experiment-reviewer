from typing import Any, Dict
from openenv.core.env_client import EnvClient, StepResult
from models import EasyAction, MediumAction, HardAction, MLObservation, MLState


class MLExperimentClient(EnvClient):

    def _step_payload(self, action) -> Dict[str, Any]:
        if isinstance(action, HardAction):
            return {
                "diagnosis":     action.diagnosis,
                "issues_found":  action.issues_found,
                "suggestions":   action.suggestions,
                "reason":        action.reason,
                "data_quality":  action.data_quality,
                "preprocessing": action.preprocessing,
                "model_check":   action.model_check,
                "overall_score": action.overall_score,
            }

        elif isinstance(action, MediumAction):
            return {
                "diagnosis":    action.diagnosis,
                "issues_found": action.issues_found,
                "suggestions":  action.suggestions,
                "reason":       action.reason,
            }

        elif isinstance(action, EasyAction):
            return {
                "diagnosis": action.diagnosis,
                "reason":    action.reason,
            }

        else:
            # fallback for unknown action types
            return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult:
        # OpenEnv wraps obs in "observation" key
        obs_data = payload.get("observation", payload)

        obs = MLObservation(
            experiment_data  = obs_data.get("experiment_data", {}),
            task_difficulty  = obs_data.get("task_difficulty", "easy"),
            task_description = obs_data.get("task_description", ""),
            done             = obs_data.get("done", payload.get("done", False)),
            reward           = obs_data.get("reward", payload.get("reward", 0.0)),
        )

        return StepResult(
            observation = obs,
            reward      = payload.get("reward", 0.0),
            done        = payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> MLState:
        return MLState(
            episode_id   = payload.get("episode_id", "unknown"),
            current_task = payload.get("current_task", "unknown"),
            step_count   = payload.get("step_count", 0),
        )