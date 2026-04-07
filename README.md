---
title: ML Experiment Reviewer
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# ML Experiment Reviewer Environment

## Introduction

_Everyone says AI is going to replace software engineers—but can it actually debug the invisible logic of a broken ML pipeline?_

The ML Experiment Reviewer is an RL environment designed to evaluate how well AI agents can review, debug, and optimize Machine Learning experiments. Finding silent bugs in ML pipelines is a challenging engineering task. Instead of basic code generation, this environment procedurally generates flawed ML experiments and requires the agent to diagnose the root causes. By testing agents across varied architectures, data splits, and hyperparameter configurations, we evaluate their debugging capabilities under constraints.

## The Three Difficulty Tiers

The environment progresses from basic pattern matching to numerical deduction:

- **Easy (The Basics):** The agent receives fundamental metrics including Training Accuracy, Validation Accuracy, and the overall loss progression across epochs. By carefully analyzing how these learning curves diverge, the agent must identify whether the model is collapsing into Overfitting, stalling due to Underfitting, stagnating at 50% accuracy (Stagnant), or generalizing effectively as a Good Fit.
- **Medium (The Architect Test):** The agent performs a full logic audit including architecture, data quality (leakage/imbalance), and 1-3 hyperparameter bugs. This requires mapping model architectures correctly to data types.
- **Hard (The Precision Math Test):** The agent receiveshardware constraints (e.g., GPU Memory) alongside a full list of 10 low-level hyperparameters. We inject a high density of 4-6 numeric bugs simultaneously, requiring the agent to pinpoint exactly where the math doesn't interact safely with the hardware.

## Grading System

Agents are graded on precision to avoid rewarding keyword stuffing.

- **Strict Output Arrays:** The agent must return its diagnosis in specific `issues_found` and `suggestions` arrays.
- **Proportional Rewards (Calibrated):** Points are awarded based on the **Matched / Total** ratio of identified bugs, ensuring partial successes are fairly rewarded.
- **False Alarm Penalty:** For every bug the AI claims exists that does not actually exist, it loses a penalty (5% for Medium, 10% for Hard). This structurally penalizes random guessing.

## Experimental Observations

While benchmarking this environment using Qwen-72B, we made the following observations:

- **Reasoning vs. Math:** The model is a stronger architect than a mathematician. It can spot architectural flaws (Medium) with ~40% accuracy but struggles with high-density numeric tuning (Hard).
- **Threshold Sensitivity:** The system uses tiered thresholds (60% for Medium, 85% for Hard), meaning the AI must be nearly word-perfect to pass the Hardest tier.
- **Scoring impact:** Because the model frequently misses subtle numeric details in high-density tasks (4-6 bugs), its Hard score naturally anchors near 4%, establishing a definitive "Elite" barrier.

## Baseline Benchmark Scores (Reality Match)

We evaluated Qwen-72B across 30 games per tier to establish the final calibrated baseline:

- **Easy Score:** **~89%** (Testing overfit/stagnant detection)
- **Medium Score:** **~40%** (Strong logic/audit capabilities)
- **Hard Score:** **~4%** (The "Precision Wall" - Extreme math difficulty)

## Quick Start: Example Interaction

Here is an example of a **Medium** tier audit task (Master Audit).

**Agent Input (Observation Payload):**

```json
{
  "task_difficulty": "medium",
  "experiment_data": {
    "data_type": "image",
    "model": "GPT-2",
    "gpu_memory_gb": 8,
    "class_distribution": { "majority_class_pct": 0.90 },
    "train_accuracy": 0.99,
    "val_accuracy": 0.99
  }
}
```

**Expected Agent Output (Action Schema):**

```json
{
  "diagnosis": "Severe architectural mismatch and data leakage detected.",
  "issues_found": ["wrong_model", "class_imbalance", "data_leakage"],
  "suggestions": [
    "Switch from GPT-2 to a ResNet or ViT for image data",
    "Use oversampling to fix balance",
    "Fix data split to remove leakage"
  ],
  "reason": "GPT-2 is a text model. 99% validation accuracy suggests target leakage.",
  "data_quality": "Severe leakage and imbalance",
  "preprocessing": "Missing augmentation",
  "model_check": "Wrong model - should use CNN/ViT",
  "overall_score": 0.1
}
```
