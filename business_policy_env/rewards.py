from __future__ import annotations

from collections import Counter

from .models import RewardBreakdown
from .tasks import component_scores, grade_actions

VALID_ACTION_REWARD = 0.05
POLICY_VIOLATION_PENALTY = -0.2
REDUNDANCY_PENALTY = 0.05
EFFICIENCY_BONUS = 0.1
INVALID_ACTION_REWARD = -0.1


def invalid_action_breakdown(message: str) -> RewardBreakdown:
    return RewardBreakdown(
        reward=INVALID_ACTION_REWARD,
        components={"invalid_action": INVALID_ACTION_REWARD},
        explanation=message,
    )


def _redundancy_penalty(action_types: list[str]) -> float:
    counts = Counter(action_types)
    repeated_actions = sum(max(0, count - 1) for count in counts.values())
    return round(repeated_actions * REDUNDANCY_PENALTY, 4)


def shaped_reward(
    actions,
    ground_truth: dict,
    done: bool,
    max_steps: int,
    policy_violations: list[str],
) -> RewardBreakdown:
    partial_score = grade_actions(actions, ground_truth)
    penalty = POLICY_VIOLATION_PENALTY if policy_violations else 0.0
    components = {
        "valid_action": VALID_ACTION_REWARD,
        "policy_penalty": penalty,
    }

    if done:
        efficiency_bonus = EFFICIENCY_BONUS if len(actions) <= max_steps / 2 else 0.0
        redundancy_penalty = _redundancy_penalty([action.action_type for action in actions])
        final_reward = max(
            0.0,
            min(
                1.0,
                round(partial_score + VALID_ACTION_REWARD + efficiency_bonus - redundancy_penalty + penalty, 4),
            ),
        )
        components.update(
            {
                "final_score": partial_score,
                "efficiency_bonus": efficiency_bonus,
                "redundancy_penalty": -redundancy_penalty,
            }
        )
        explanation = "Final reward includes graded score, efficiency bonus, redundancy penalty, and policy penalties."
        return RewardBreakdown(reward=final_reward, components=components, explanation=explanation)

    intermediate_reward = round(VALID_ACTION_REWARD + partial_score + penalty, 4)
    components["partial_score"] = partial_score
    explanation = "Intermediate reward includes a valid-action bonus, current partial score, and any policy penalties."
    return RewardBreakdown(reward=intermediate_reward, components=components, explanation=explanation)


def current_progress(actions, ground_truth: dict) -> tuple[float, dict[str, float]]:
    return grade_actions(actions, ground_truth), component_scores(actions, ground_truth)
