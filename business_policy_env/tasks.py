from __future__ import annotations

from functools import lru_cache

from .data_generation import build_scenarios
from .models import Action, TaskScenario, TicketSnapshot
from .policies import policies_satisfied


def compute_issue_age_hours(snapshot: TicketSnapshot, now) -> float:
    first_timestamp = snapshot.thread[0].timestamp
    return round((now - first_timestamp).total_seconds() / 3600, 2)


@lru_cache(maxsize=1)
def scenario_registry() -> dict[str, TaskScenario]:
    return {scenario.scenario_id: scenario for scenario in build_scenarios()}


def scenarios_for_task(task_name: str | None = None) -> list[TaskScenario]:
    scenarios = list(scenario_registry().values())
    if task_name is None:
        return sorted(scenarios, key=lambda item: (item.difficulty, item.scenario_id))
    return sorted(
        [scenario for scenario in scenarios if scenario.difficulty == task_name],
        key=lambda item: item.scenario_id,
    )


def build_ground_truth_payload(scenario: TaskScenario, snapshot: TicketSnapshot) -> dict:
    return {
        "difficulty": scenario.difficulty,
        "expected_category": scenario.ground_truth.expected_category,
        "expected_priority": scenario.ground_truth.expected_priority,
        "expected_escalation": scenario.ground_truth.expected_escalation,
        "expected_escalation_reason": scenario.ground_truth.expected_escalation_reason,
        "requires_request_info": scenario.ground_truth.requires_request_info,
        "request_info_first_required": scenario.ground_truth.request_info_first_required,
        "clarification_keywords": scenario.ground_truth.clarification_keywords,
        "response_keywords": scenario.ground_truth.response_keywords,
        "history_keywords": scenario.ground_truth.history_keywords,
        "completion_action_types": scenario.ground_truth.completion_action_types,
        "ambiguous": scenario.ground_truth.ambiguous,
        "snapshot": snapshot.model_dump(mode="json"),
        "issue_age_hours": compute_issue_age_hours(snapshot, scenario.now),
    }


def latest_action(actions: list[Action], action_type: str) -> Action | None:
    for action in reversed(actions):
        if action.action_type == action_type:
            return action
    return None


def _request_info_quality(action: Action | None, keywords: list[str]) -> float:
    if action is None or not action.clarifying_question:
        return 0.0
    text = action.clarifying_question.lower()
    if not keywords:
        return 1.0
    hits = sum(1 for keyword in keywords if keyword.lower() in text)
    if hits:
        return min(1.0, hits / len(keywords))
    return 0.0


def _keyword_score(text: str | None, keywords: list[str]) -> float:
    if not text:
        return 0.0
    lowered = text.lower()
    if not keywords:
        return 1.0
    hits = sum(1 for keyword in keywords if keyword.lower() in lowered)
    return min(1.0, hits / len(keywords))


def _categorize_score(actions: list[Action], expected_category: str | None) -> float:
    if expected_category is None:
        return 1.0
    action = latest_action(actions, "categorize")
    if action is None:
        return 0.0
    return 1.0 if action.category == expected_category else 0.0


def _priority_score(actions: list[Action], expected_priority: str | None) -> float:
    if expected_priority is None:
        return 1.0
    action = latest_action(actions, "set_priority")
    if action is None:
        return 0.0
    return 1.0 if action.priority == expected_priority else 0.0


def _escalation_score(actions: list[Action], expected_escalation: bool) -> float:
    escalated = any(action.action_type == "escalate" for action in actions)
    return 1.0 if escalated == expected_escalation else 0.0


def _policy_score(actions: list[Action], ground_truth: dict) -> float:
    snapshot = TicketSnapshot.model_validate(ground_truth["snapshot"])
    return 1.0 if policies_satisfied(actions, snapshot, float(ground_truth["issue_age_hours"])) else 0.0


def easy_grader(actions: list[Action], ground_truth: dict) -> float:
    components = easy_components(actions, ground_truth)
    return round(
        0.4 * components["category_correct"]
        + 0.35 * components["priority_correct"]
        + 0.25 * components["policy_compliance"],
        4,
    )



def easy_components(actions: list[Action], ground_truth: dict) -> dict[str, float]:
    return {
        "category_correct": _categorize_score(actions, ground_truth["expected_category"]),
        "priority_correct": _priority_score(actions, ground_truth["expected_priority"]),
        "policy_compliance": _policy_score(actions, ground_truth),
    }


def medium_grader(actions: list[Action], ground_truth: dict) -> float:
    if ground_truth["request_info_first_required"]:
        if not actions or actions[0].action_type != "request_info":
            return 0.0
    components = medium_components(actions, ground_truth)
    return round(
        0.25 * components["ambiguity_recognition"]
        + 0.2 * components["clarifying_question_quality"]
        + 0.15 * components["policy_compliance"]
        + 0.1 * components["category_correct"]
        + 0.05 * components["priority_correct"]
        + 0.25 * components["response_appropriateness"],
        4,
    )


def medium_components(actions: list[Action], ground_truth: dict) -> dict[str, float]:
    request_info_action = actions[0] if actions and actions[0].action_type == "request_info" else None
    draft_action = latest_action(actions, "draft_response")
    return {
        "ambiguity_recognition": 1.0 if request_info_action else 0.0,
        "clarifying_question_quality": _request_info_quality(request_info_action, ground_truth["clarification_keywords"]),
        "policy_compliance": _policy_score(actions, ground_truth),
        "category_correct": _categorize_score(actions, ground_truth["expected_category"]),
        "priority_correct": _priority_score(actions, ground_truth["expected_priority"]),
        "response_appropriateness": _keyword_score(
            draft_action.response_text if draft_action else None,
            ground_truth["response_keywords"],
        ),
    }


def hard_grader(actions: list[Action], ground_truth: dict) -> float:
    if ground_truth["request_info_first_required"]:
        if not actions or actions[0].action_type != "request_info":
            return 0.0
    components = hard_components(actions, ground_truth)
    return round(
        0.1 * components["temporal_reasoning"]
        + 0.1 * components["policy_compliance"]
        + 0.1 * components["escalation_accuracy"]
        + 0.3 * components["history_acknowledgment"]
        + 0.4 * components["response_completeness"],
        4,
    )


def hard_components(actions: list[Action], ground_truth: dict) -> dict[str, float]:
    draft_action = latest_action(actions, "draft_response")
    response_text = draft_action.response_text if draft_action else None
    response_keywords = _keyword_score(response_text, ground_truth["response_keywords"])
    history_score = _keyword_score(response_text, ground_truth["history_keywords"])
    category_score = _categorize_score(actions, ground_truth["expected_category"])
    policy_score = 1.0 if _policy_score(actions, ground_truth) == 1.0 and category_score == 1.0 else 0.0
    return {
        "temporal_reasoning": _priority_score(actions, ground_truth["expected_priority"]),
        "policy_compliance": policy_score,
        "escalation_accuracy": _escalation_score(actions, ground_truth["expected_escalation"]),
        "history_acknowledgment": history_score,
        "response_completeness": response_keywords,
    }


def grade_actions(actions: list[Action], ground_truth: dict) -> float:
    difficulty = ground_truth["difficulty"]
    if difficulty == "easy":
        return easy_grader(actions, ground_truth)
    if difficulty == "medium":
        return medium_grader(actions, ground_truth)
    return hard_grader(actions, ground_truth)


def component_scores(actions: list[Action], ground_truth: dict) -> dict[str, float]:
    difficulty = ground_truth["difficulty"]
    if difficulty == "easy":
        return easy_components(actions, ground_truth)
    if difficulty == "medium":
        return medium_components(actions, ground_truth)
    return hard_components(actions, ground_truth)
