# ─── app.py ──────────────────────────────────────────────────

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from openenv.core.env_server import create_app
from server.environment import MLExperimentEnvironment
from models import MLObservation, MLAction

app = create_app(MLExperimentEnvironment, MLAction, MLObservation)

@app.get("/")
def root():
    return {
        "message": "Welcome to the ML Experiment Reviewer API!",
        "status": "running",
        "docs": "/docs"
    }

# ─── Tasks Endpoint ──────────────────────────────────────────

@app.get("/tasks")
def get_tasks():
    return {
        "tasks": [
            {
                "name": "easy",
                "difficulty": "easy",
                "description": "Determine if the model is overfitting, underfitting, good_fit, or stagnant. (Note: A gap of 5% or more indicates overfitting)",
                "action_schema": {
                    "diagnosis": "string (overfitting/underfitting/good_fit)",
                    "reason":    "string (why you made this diagnosis)"
                }
            },
            {
                "name": "medium",
                "difficulty": "medium",
                "description": "Full logic audit (architecture + data quality) plus hyperparameter bugs",
                "action_schema": {
                    "diagnosis":     "string (overall diagnosis)",
                    "issues_found":  "list of strings (all issues found)",
                    "suggestions":   "list of strings (all fixes)",
                    "reason":        "string (why these fixes help)",
                    "data_quality":  "string (data leakage? balanced?)",
                    "preprocessing": "string (normalization? augmentation?)",
                    "model_check":   "string (right model for task?)",
                    "overall_score": "float (0.0-1.0 experiment quality)"
                }
            },
            {
                "name": "hard",
                "difficulty": "hard",
                "description": "Find problematic hyperparameters and suggest fixes",
                "action_schema": {
                    "diagnosis":    "string (overall diagnosis)",
                    "issues_found": "list of strings (problematic hyperparameters)",
                    "suggestions":  "list of strings (how to fix each problem)",
                    "reason":       "string (why these changes help)"
                }
            }
        ]
    }

# ─── Baseline Endpoint ───────────────────────────────────────

@app.get("/baseline")
def get_baseline():
    try:
        import random
        from server.tasks import (
            generate_easy_experiment,
            generate_medium_experiment,
            generate_hard_experiment
        )
        from server.grader import grade
        from models import EasyAction, MediumAction, HardAction

        # Easy baseline
        easy_scores = []
        for _ in range(10):
            exp    = generate_easy_experiment()
            action = EasyAction(
                diagnosis = random.choice(["overfitting", "underfitting", "good_fit"]),
                reason    = "random baseline diagnosis"
            )
            score = grade(action, exp["correct_answer"], "easy")
            easy_scores.append(score)

        # Medium baseline (Audit)
        medium_scores = []
        for _ in range(10):
            exp    = generate_medium_experiment()
            action = HardAction(
                diagnosis     = "overfitting",
                issues_found  = ["learning rate too high"],
                suggestions   = ["reduce learning rate"],
                reason        = "random baseline diagnosis",
                data_quality  = "looks fine no issues",
                preprocessing = "looks correct no issues",
                model_check   = "correct model for task",
                overall_score = 0.5
            )
            score = grade(action, exp["correct_answer"], "medium")
            medium_scores.append(score)

        # Hard baseline (Numeric)
        hard_scores = []
        for _ in range(10):
            exp    = generate_hard_experiment()
            action = MediumAction(
                diagnosis    = "overfitting",
                issues_found = ["learning rate too high"],
                suggestions  = ["reduce learning rate"],
                reason       = "random baseline diagnosis"
            )
            score = grade(action, exp["correct_answer"], "hard")
            hard_scores.append(score)

        return {
            "baselines": {
                "easy":   max(0.01, min(0.99, round(sum(easy_scores)   / len(easy_scores),   2))),
                "medium": max(0.01, min(0.99, round(sum(medium_scores) / len(medium_scores), 2))),
                "hard":   max(0.01, min(0.99, round(sum(hard_scores)   / len(hard_scores),   2))),
            },
            "description": "Random baseline scores averaged over 10 episodes per task"
        }

    except Exception as e:
        return {"error": str(e)}

# ─── Grader Endpoint ─────────────────────────────────────────

@app.get("/grader")
def get_grader_info():
    return {
        "grading_criteria": {
            "easy": {
                "correct_diagnosis": "80% of score",
                "good_reason":       "20% of score",
                "total":             "1.0"
            },
            "medium": {
                "problems_found":   "dynamic proportional score based on actual problems",
                "false_alarm":      "-10% per hallucinated issue",
                "total":            "1.0"
            },
            "hard": {
                "problems_found":   "dynamic proportional score based on actual problems",
                "false_alarm":      "-10% per hallucinated issue",
                "total":            "1.0"
            }
        }
    }

# ─── Run Server ──────────────────────────────────────────────

def main():
    uvicorn.run(
        "server.app:app",
        host   = "0.0.0.0",
        port   = 8000,
        reload = True
    )

if __name__ == "__main__":
    main()
