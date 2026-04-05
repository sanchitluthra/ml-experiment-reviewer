# ─── inference.py ────────────────────────────────────────────
# Runs the LLM against our ML Experiment Reviewer environment
# and records baseline scores with mandatory [START]/[STEP]/[END] format.

import asyncio
import os
import json
import re
from typing import List, Optional

from openai import OpenAI
from client import MLExperimentClient
from models import EasyAction, MediumAction, HardAction, MLAction

# ─── Mandatory Environment Variables ─────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

BENCHMARK = "ml-experiment-env"
MAX_STEPS = 3  # one step per difficulty (easy, medium, hard)

client_ai = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ─── Stdout Logging (Mandatory Format) ──────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ─── Helper: Ask LLM ────────────────────────────────────────

def ask_llm(prompt: str) -> str:
    """Send prompt to LLM and get response."""
    try:
        response = client_ai.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior ML engineer. Analyze ML experiments "
                        "and provide diagnosis. Always respond in valid JSON "
                        "format only. No markdown, no code blocks, just raw JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content or "{}"
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        return "{}"


def parse_json_response(text: str) -> dict:
    """Robustly parse JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    # Strip markdown code fences if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"[DEBUG] JSON parse failed, raw text: {text[:200]}", flush=True)
        return {}


# ─── Task Runners ────────────────────────────────────────────

def run_easy(env) -> tuple:
    """Run easy task. Returns (reward, action_summary, error)."""
    result = env.reset(difficulty="easy")
    obs = result.observation

    prompt = f"""{obs.task_description}

Experiment Data:
{json.dumps(obs.experiment_data, indent=2)}

Respond in this exact JSON format:
{{
    "diagnosis": "overfitting" or "underfitting" or "good_fit",
    "reason": "your explanation here"
}}"""

    raw = ask_llm(prompt)
    response = parse_json_response(raw)

    action = EasyAction(
        diagnosis=response.get("diagnosis", "good_fit"),
        reason=response.get("reason", "no reason given"),
    )

    result = env.step(action)
    action_summary = f"diagnosis={action.diagnosis}"
    return result.reward, action_summary, None


def run_medium(env) -> tuple:
    """Run medium task (full experiment diagnosis). Returns (reward, action_summary, error)."""
    result = env.reset(difficulty="medium")
    obs = result.observation

    prompt = f"""{obs.task_description}

Experiment Data:
{json.dumps(obs.experiment_data, indent=2)}

Respond in this exact JSON format:
{{
    "diagnosis": "overall diagnosis",
    "issues_found": ["issue 1", "issue 2"],
    "suggestions": ["fix 1", "fix 2"],
    "reason": "why these fixes help",
    "data_quality": "assessment of data quality",
    "preprocessing": "assessment of preprocessing",
    "model_check": "correct model for this data type" or "wrong model - should use X instead",
    "overall_score": 0.5
}}"""

    raw = ask_llm(prompt)
    response = parse_json_response(raw)

    action = HardAction(
        diagnosis=response.get("diagnosis", "unknown"),
        issues_found=response.get("issues_found", []),
        suggestions=response.get("suggestions", []),
        reason=response.get("reason", "no reason given"),
        data_quality=response.get("data_quality", "unknown"),
        preprocessing=response.get("preprocessing", "unknown"),
        model_check=response.get("model_check", "unknown"),
        overall_score=float(response.get("overall_score", 0.0)),
    )

    result = env.step(action)
    action_summary = f"diagnosis={action.diagnosis},issues={len(action.issues_found)}"
    return result.reward, action_summary, None


def run_hard(env) -> tuple:
    """Run hard task (pure hyperparameter tuning). Returns (reward, action_summary, error)."""
    result = env.reset(difficulty="hard")
    obs = result.observation

    prompt = f"""{obs.task_description}

Experiment Data:
{json.dumps(obs.experiment_data, indent=2)}

Respond in this exact JSON format:
{{
    "diagnosis": "overall diagnosis",
    "issues_found": ["issue 1", "issue 2"],
    "suggestions": ["fix 1", "fix 2"],
    "reason": "why these fixes help"
}}"""

    raw = ask_llm(prompt)
    response = parse_json_response(raw)

    action = MediumAction(
        diagnosis=response.get("diagnosis", "unknown"),
        issues_found=response.get("issues_found", []),
        suggestions=response.get("suggestions", []),
        reason=response.get("reason", "no reason given"),
    )

    result = env.step(action)
    action_summary = f"diagnosis={action.diagnosis},issues={len(action.issues_found)}"
    return result.reward, action_summary, None


# ─── Main Runner ─────────────────────────────────────────────

TASK_RUNNERS = [
    ("easy", run_easy),
    ("medium", run_medium),
    ("hard", run_hard),
]


def run_all_tasks():
    """
    Run all 3 tasks with mandatory [START]/[STEP]/[END] logging.
    Each task is one episode: reset → step → reward.
    """
    for task_name, runner in TASK_RUNNERS:
        rewards: List[float] = []
        steps_taken = 0
        success = False
        score = 0.0

        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

        try:
            with MLExperimentClient(base_url=ENV_URL).sync() as env:
                try:
                    reward, action_summary, error = runner(env)
                    steps_taken = 1
                    rewards.append(reward)
                    done = True

                    log_step(
                        step=1,
                        action=action_summary,
                        reward=reward,
                        done=done,
                        error=error,
                    )

                    score = reward  # single-step, so score = reward
                    success = score > 0.0

                except Exception as exc:
                    steps_taken = 1
                    rewards.append(0.0)
                    log_step(
                        step=1,
                        action="error",
                        reward=0.0,
                        done=True,
                        error=str(exc),
                    )

        except Exception as exc:
            # Connection error
            print(f"[DEBUG] Connection failed for {task_name}: {exc}", flush=True)

        finally:
            log_end(
                success=success,
                steps=steps_taken,
                score=score,
                rewards=rewards,
            )


# ─── Entry Point ─────────────────────────────────────────────

if __name__ == "__main__":
    run_all_tasks()