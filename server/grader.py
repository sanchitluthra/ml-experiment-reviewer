# ─── grader.py ───────────────────────────────────────────────

import re
from models import EasyAction, MediumAction, HardAction

def _clamp_score(score: float) -> float:
    return max(0.01, min(0.99, round(score, 2)))

def _normalize_text(text: str) -> str:
    text = text.lower().replace("_", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = text.replace("lr", "learning rate").replace("bs", "batch size")
    return " ".join(text.split())

def grade_easy(action: EasyAction, correct_answer: str) -> float:
    reward = 0.8 if action.diagnosis.lower() == correct_answer.lower() else 0.0
    if action.reason and len(action.reason.strip()) > 10: reward += 0.2
    return _clamp_score(reward)

def _score_dynamic(guesses, actuals, sol_guesses, actual_sols, difficulty="medium"):
    stop_words = {"too", "very", "is", "the", "a", "an", "for", "to", "at", "it", "needs", "be"}
    config = {
        "medium": {"threshold": 0.60, "penalty": 0.05},
        "hard":   {"threshold": 0.85, "penalty": 0.10},
    }
    tier = config.get(difficulty, config["medium"])

    def calc(g_list, a_list):
        if not a_list: return 1.0 - (len(g_list) * tier["penalty"])
        matched = 0
        norm_g = [_normalize_text(g) for g in g_list]
        combined_g = " ".join(norm_g)
        g_tokens = set(combined_g.split())

        for target in a_list:
            t_tokens = [t for t in _normalize_text(target).split() if t not in stop_words]
            if not t_tokens: continue
            if sum(1 for t in t_tokens if t in g_tokens) / len(t_tokens) >= tier["threshold"]:
                matched += 1
        
        # Penalties: False Alarms (all tiers)
        false_alarm_penalty = 0.0
        for g in norm_g:
            g_t = set(g.split())
            if not any(sum(1 for t in [tk for tk in _normalize_text(ta).split() if tk not in stop_words] if t in g_t) / len([tk for tk in _normalize_text(ta).split() if tk not in stop_words]) >= tier["threshold"] for ta in a_list if [tk for tk in _normalize_text(ta).split() if tk not in stop_words]):
                false_alarm_penalty += tier["penalty"]
        
        return (matched / len(a_list)) - false_alarm_penalty

    i_score = calc(guesses, actuals)
    s_score = calc(sol_guesses, actual_sols)
    return _clamp_score(0.7 * i_score + 0.3 * s_score)

def grade_medium(action: HardAction, correct_answer: dict) -> float:
    # Medium is now the Master Audit (using HardAction schema)
    return _score_dynamic(
        action.issues_found, correct_answer.get("problems", []),
        action.suggestions, correct_answer.get("solutions", []),
        difficulty="medium"
    )

def grade_hard(action: MediumAction, correct_answer: dict) -> float:
    # Hard is now the Numeric Tuning (using MediumAction schema)
    return _score_dynamic(
        action.issues_found, correct_answer.get("problems", []),
        action.suggestions, correct_answer.get("solutions", []),
        difficulty="hard"
    )

def grade(action, correct_answer, difficulty: str) -> float:
    try:
        if difficulty == "easy": return grade_easy(action, correct_answer)
        if difficulty == "medium": return grade_medium(action, correct_answer)
        if difficulty == "hard": return grade_hard(action, correct_answer)
        return 0.01 
    except Exception: return 0.01
