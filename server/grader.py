# ─── grader.py ───────────────────────────────────────────────
# Strict, dynamic grader with false alarm penalties. Returns reward 0.0 - 1.0

from models import EasyAction, MediumAction, HardAction

# ─── Easy Task Grader ────────────────────────────────────────

def grade_easy(action: EasyAction, correct_answer: str) -> float:
    reward = 0.0
    if action.diagnosis.lower() == correct_answer.lower():
        reward += 0.8
    if action.reason and len(action.reason.strip()) > 10:
        reward += 0.2
    return max(0.0, round(reward, 2))


# ─── Dynamic Grading Helpers ─────────────────────────────────

def _score_dynamic(agent_guesses_list: list[str], actual_problems: list[str]) -> float:
    """
    Grades by calculating:
      1. Base Reward = (Correct Matches) / (Total Actual Problems)
      2. False Alarm Penalty = -0.1 for each guess in agent_guesses_list that matched NO actual problems
    """
    if not actual_problems:
        # 0 actual problems means this is a "good_fit" experiment.
        # Penalty for hallucinating problems when none exist:
        penalty = len(agent_guesses_list) * 0.1
        return max(0.0, 1.0 - penalty)

    found_count = 0
    matched_problems = set()
    
    # 1. Check how many actual problems the agent listed explicitly in its issues array
    agent_guesses_combined = " ".join(agent_guesses_list).lower()
    for problem in actual_problems:
        problem_phrase = problem.replace("_", " ").lower()
        if problem_phrase in agent_guesses_combined:
            found_count += 1
            matched_problems.add(problem)
            
    base_reward = found_count / len(actual_problems)
    
    # 2. False Alarm Penalty
    # For every explicit issue the agent listed, if it doesn't vaguely match an actual problem, penalize it.
    penalty = 0.0
    for guess in agent_guesses_list:
        guess_lower = guess.lower()
        
        is_real_problem = False
        for problem in actual_problems:
            if problem.replace("_", " ").lower() in guess_lower:
                is_real_problem = True
                break
        
        if not is_real_problem:
            penalty += 0.1  # 10% penalty for hallucinating an issue
            
    final_score = base_reward - penalty
    return max(0.0, round(final_score, 2))


# ─── Hard Task Grader (Pure Hyperparameter Tuning) ──────────

def grade_hard(action: MediumAction, correct_answer: list[str]) -> float:
    actual_problems = correct_answer
    
    return _score_dynamic(
        agent_guesses_list=action.issues_found,
        actual_problems=actual_problems
    )


# ─── Medium Task Grader (Full Experiment Diagnosis) ──────────

def grade_medium(action: HardAction, correct_answer: dict) -> float:
    actual_problems = []
    
    actual_problems.extend(correct_answer.get("hyperparameter_problems", []))
    actual_problems.extend(correct_answer.get("data_quality_issues", []))
    actual_problems.extend(correct_answer.get("preprocessing_issues", []))
    
    if not correct_answer.get("model_correct", True):
        actual_problems.append("wrong_model")
        
    return _score_dynamic(
        agent_guesses_list=action.issues_found,
        actual_problems=actual_problems
    )


# ─── Main Grader Function ────────────────────────────────────

def grade(action, correct_answer, difficulty: str) -> float:
    if difficulty == "easy":
        return grade_easy(action, correct_answer)
    elif difficulty == "medium":
        return grade_medium(action, correct_answer)
    elif difficulty == "hard":
        return grade_hard(action, correct_answer)
    else:
        return 0.0