from __future__ import annotations

import re
from datetime import datetime
from typing import Any, cast

from .data_generation import build_scenarios
from .llm_grader import score_response_with_optional_llm
from .models import Action, ActionType, PolicyVersion, TaskScenario, TicketSnapshot
from .policies import compute_policy_expectations, policies_satisfied

GroundTruthPayload = dict[str, Any]


def compute_issue_age_hours(snapshot: TicketSnapshot, now: datetime) -> float:
    first_timestamp = snapshot.thread[0].timestamp
    return round((now - first_timestamp).total_seconds() / 3600, 2)


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


FILLER_PATTERNS = re.compile(
    r"^(can you (share|provide|tell)|please (confirm|clarify)|can you clarify|please share)\b.*$",
    re.IGNORECASE,
)
LIST_PATTERN = re.compile(r"\b(\w+,\s*){3,}\w+\b")
QUESTION_TOPIC_WORDS = {
    "invoice",
    "charge",
    "refund",
    "account",
    "order",
    "date",
    "amount",
    "reason",
    "issue",
    "card",
    "payment",
    "error",
    "contract",
    "migration",
    "outage",
    "plan",
}
AGENT_VERBS = {"can", "will", "do", "is", "are", "have", "need", "could", "would"}


def build_ground_truth_payload(
    scenario: TaskScenario,
    snapshot: TicketSnapshot,
    *,
    policy_version: PolicyVersion | None = None,
) -> GroundTruthPayload:
    return {
        "difficulty": scenario.difficulty,
        "policy_version": policy_version or scenario.policy_version,
        "policy_transition_step": scenario.policy_transition_step,
        "policy_transition_to": scenario.policy_transition_to,
        "expected_category": scenario.ground_truth.expected_category,
        "expected_priority": scenario.ground_truth.expected_priority,
        "expected_escalation": scenario.ground_truth.expected_escalation,
        "expected_escalation_reason": scenario.ground_truth.expected_escalation_reason,
        "expected_flag_fraud": scenario.ground_truth.expected_flag_fraud,
        "fraud_keywords": scenario.ground_truth.fraud_keywords,
        "requires_request_info": scenario.ground_truth.requires_request_info,
        "request_info_first_required": scenario.ground_truth.request_info_first_required,
        "clarification_keywords": scenario.ground_truth.clarification_keywords,
        "response_keywords": scenario.ground_truth.response_keywords,
        "history_keywords": scenario.ground_truth.history_keywords,
        "conflict_keywords": scenario.ground_truth.conflict_keywords,
        "required_action_order": scenario.ground_truth.required_action_order,
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


def _tokenize(text: str | None) -> list[str]:
    if not text:
        return []
    return re.findall(r"[a-z0-9']+", text.lower())


def _request_info_keyword_score(action: Action | None, keywords: list[str]) -> float:
    if action is None or not action.clarifying_question:
        return 0.0
    text = action.clarifying_question.lower()
    if not keywords:
        return 1.0
    hits = sum(1 for keyword in keywords if keyword.lower() in text)
    if hits:
        return min(1.0, hits / len(keywords))
    return 0.0


def is_substantive_question(text: str | None) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if FILLER_PATTERNS.match(stripped):
        return False
    if LIST_PATTERN.search(stripped):
        return False
    tokens = _tokenize(text)
    unique_tokens = set(tokens)
    topic_hit_count = len(unique_tokens & QUESTION_TOPIC_WORDS)
    if topic_hit_count < 2:
        return False
    has_verb = bool(unique_tokens & AGENT_VERBS)
    return len(tokens) >= 10 and has_verb and "?" in stripped


def request_info_quality(action: Action | None, ground_truth: GroundTruthPayload) -> float:
    if action is None:
        return 0.0
    keyword_score = _request_info_keyword_score(action, ground_truth["clarification_keywords"])
    substantive_bonus = 1.0 if is_substantive_question(action.clarifying_question) else 0.0
    return round(0.7 * keyword_score + 0.3 * substantive_bonus, 4)


def _keyword_score(text: str | None, keywords: list[str]) -> float:
    if not text:
        return 0.0
    lowered = text.lower()
    if not keywords:
        return 1.0
    hits = sum(1 for keyword in keywords if keyword.lower() in lowered)
    return min(1.0, hits / len(keywords))


def _signal_score(text: str | None, signals: list[str]) -> float:
    if not text:
        return 0.0
    lowered = text.lower()
    hits = sum(1 for signal in signals if signal in lowered)
    return min(1.0, hits / max(1, len(signals)))


def _anti_stuffing_factor(
    response_text: str | None,
    response_keywords: list[str],
    history_keywords: list[str],
) -> float:
    if not response_text:
        return 0.0

    tokens = _tokenize(response_text)
    if not tokens:
        return 0.0

    lowered = response_text.lower()
    keyword_hits = sum(1 for keyword in response_keywords + history_keywords if keyword.lower() in lowered)
    unique_ratio = len(set(tokens)) / len(tokens)
    keyword_density = keyword_hits / len(tokens)
    sentence_markers = sum(response_text.count(marker) for marker in [".", "!", "?"])

    factor = 1.0
    if len(tokens) < 8:
        factor = min(factor, 0.35)
    elif len(tokens) < 12:
        factor = min(factor, 0.6)

    if unique_ratio < 0.5:
        factor = min(factor, 0.7)
    if keyword_density > 0.35:
        factor = min(factor, 0.65)
    if sentence_markers == 0:
        factor = min(factor, 0.75)

    return round(factor, 4)


def _bounded_ratio(value: float, target: float) -> float:
    if target <= 0:
        return 1.0
    return max(0.0, min(1.0, round(value / target, 4)))


def _coherence_gate(response_text: str, difficulty: str | None = None) -> float:
    sentences = [segment.strip() for segment in re.split(r"[.!?]", response_text) if len(segment.strip()) > 8]
    tokens = _tokenize(response_text)
    if not tokens:
        return 0.0

    lowered = response_text.lower()
    connector_hits = sum(
        lowered.count(marker)
        for marker in [" because ", " and ", " after ", " before ", " while ", " if ", " when ", " but "]
    )
    unique_ratio = len(set(tokens)) / len(tokens)
    profile = {
        "medium": {
            "sentence_target": 1.5,
            "token_target": 12.0,
            "variety_target": 0.58,
            "connector_target": 1.0,
        },
        "hard": {
            "sentence_target": 2.2,
            "token_target": 18.0,
            "variety_target": 0.62,
            "connector_target": 2.0,
        },
    }.get(difficulty or "hard", {
        "sentence_target": 2.2,
        "token_target": 18.0,
        "variety_target": 0.62,
        "connector_target": 2.0,
    })

    sentence_score = _bounded_ratio(float(len(sentences)), profile["sentence_target"])
    length_score = _bounded_ratio(float(len(tokens)), profile["token_target"])
    variety_score = _bounded_ratio(unique_ratio, profile["variety_target"])
    connector_score = _bounded_ratio(float(connector_hits), profile["connector_target"])

    coherence = round(
        0.4 * sentence_score + 0.25 * length_score + 0.2 * variety_score + 0.15 * connector_score,
        4,
    )
    if difficulty == "hard" and len(sentences) < 2:
        single_sentence_cap = 0.26 + 0.12 * _bounded_ratio(float(len(tokens)), 20.0)
        coherence = min(coherence, round(single_sentence_cap, 4))
    return max(0.15, coherence)


def _response_rubric(
    response_text: str | None,
    response_keywords: list[str],
    history_keywords: list[str],
    ground_truth: GroundTruthPayload,
) -> dict[str, float]:
    if not response_text:
        return {
            "case_specific_facts": 0.0,
            "history_acknowledgment": 0.0,
            "next_step_actionability": 0.0,
            "tone_acknowledgment": 0.0,
            "anti_stuffing": 0.0,
            "coherence_gate": 0.0,
            "llm_judge_used": 0.0,
            "response_quality": 0.0,
        }

    fact_score = _keyword_score(response_text, response_keywords)
    history_score = _keyword_score(response_text, history_keywords)
    next_step_score = _signal_score(
        response_text,
        ["will", "follow", "update", "review", "escalat", "investigat", "check", "resolve", "today", "next step"],
    )
    tone_score = _signal_score(
        response_text,
        ["sorry", "apolog", "understand", "recogn", "appreciate", "thanks", "noted", "aware"],
    )
    anti_stuffing = _anti_stuffing_factor(response_text, response_keywords, history_keywords)
    coherence_gate = _coherence_gate(response_text, str(ground_truth.get("difficulty")))
    heuristic_quality = round(
        0.4 * fact_score + 0.25 * next_step_score + 0.2 * tone_score + 0.15 * history_score,
        4,
    )
    llm_score = score_response_with_optional_llm(
        response_text,
        ground_truth,
        force=(ground_truth.get("difficulty") == "hard"),
    )
    response_quality = round(
        (llm_score if llm_score is not None else heuristic_quality) * anti_stuffing * coherence_gate,
        4,
    )
    return {
        "case_specific_facts": round(fact_score, 4),
        "history_acknowledgment": round(history_score, 4),
        "next_step_actionability": round(next_step_score, 4),
        "tone_acknowledgment": round(tone_score, 4),
        "anti_stuffing": anti_stuffing,
        "coherence_gate": round(coherence_gate, 4),
        "llm_judge_used": 1.0 if llm_score is not None else 0.0,
        "response_quality": response_quality,
    }


def _contradiction_detection_score(actions: list[Action], ground_truth: GroundTruthPayload) -> float:
    conflict_keywords: list[str] = ground_truth.get("conflict_keywords", [])
    if not conflict_keywords:
        snapshot = TicketSnapshot.model_validate(ground_truth["snapshot"])
        policy_version = cast(PolicyVersion, ground_truth["policy_version"])
        expectations = compute_policy_expectations(snapshot, float(ground_truth["issue_age_hours"]), policy_version)
        derived_signals: list[list[str]] = []

        if ground_truth.get("expected_flag_fraud") or expectations["requires_fraud_flag"]:
            fraud_keywords = cast(list[str], ground_truth.get("fraud_keywords", []))
            derived_signals.append(["fraud", *fraud_keywords[:2]])

        if expectations["requires_escalation"]:
            escalation_keywords = ["escalation"]
            latest_text = f"{snapshot.thread[-1].subject} {snapshot.thread[-1].body}".lower()
            if any(term in latest_text for term in ["legal", "lawyer", "counsel", "lawsuit"]):
                escalation_keywords.extend(["legal", "lawyer"])
            elif snapshot.refund_amount is not None:
                escalation_keywords.extend(["refund", "billing"])
            derived_signals.append(escalation_keywords)

        if snapshot.sender_tier in {"vip", "premier"}:
            derived_signals.append([snapshot.sender_tier, "priority"])

        if "suspended" in snapshot.account_flags:
            derived_signals.append(["suspended", "billing"])

        if snapshot.visible_problem_type:
            visible_problem_tokens = snapshot.visible_problem_type.replace("_", " ").split()
            derived_signals.append(visible_problem_tokens)

        if float(ground_truth["issue_age_hours"]) > 72 or ground_truth.get("expected_priority") == "urgent":
            derived_signals.append(["urgent", "timeline"])

        if ground_truth.get("policy_transition_to"):
            derived_signals.append(["policy", str(ground_truth["policy_transition_to"])])

        if len(derived_signals) >= 2:
            conflict_keywords = list(
                dict.fromkeys(keyword for signal in derived_signals for keyword in signal if keyword)
            )
    if not conflict_keywords:
        return 1.0
    draft_action = latest_action(actions, "draft_response")
    escalate_action = latest_action(actions, "escalate")
    fraud_action = latest_action(actions, "flag_fraud")
    text_fragments = [
        draft_action.response_text if draft_action and draft_action.response_text else "",
        draft_action.reasoning if draft_action and draft_action.reasoning else "",
        escalate_action.escalation_reason if escalate_action and escalate_action.escalation_reason else "",
        fraud_action.fraud_reason if fraud_action and fraud_action.fraud_reason else "",
    ]
    combined = " ".join(fragment for fragment in text_fragments if fragment).lower()
    hits = sum(1 for keyword in conflict_keywords if keyword.lower() in combined)
    return round(min(1.0, hits / len(conflict_keywords)), 4)


def _ordering_correctness_score(actions: list[Action], ground_truth: GroundTruthPayload) -> float:
    required_order = cast(list[ActionType], ground_truth.get("required_action_order", []))
    if not required_order and ground_truth.get("difficulty") == "hard":
        if ground_truth.get("expected_flag_fraud") and ground_truth.get("expected_escalation"):
            required_order = ["flag_fraud", "escalate", "draft_response"]
        elif ground_truth.get("expected_flag_fraud"):
            required_order = ["flag_fraud", "draft_response"]
        elif ground_truth.get("expected_escalation"):
            required_order = ["escalate", "draft_response"]
    if not required_order:
        return 1.0
    action_types = [action.action_type for action in actions]
    positions: list[int] = []
    for required in required_order:
        try:
            positions.append(action_types.index(required))
        except ValueError:
            return 0.0
    return 1.0 if positions == sorted(positions) else 0.3


def _sequencing_penalty_factor(actions: list[Action], ground_truth: GroundTruthPayload) -> float:
    if not ground_truth["request_info_first_required"]:
        return 1.0
    if actions and actions[0].action_type == "request_info":
        return 1.0
    return 0.72 if ground_truth.get("difficulty") == "medium" else 0.35


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


def _fraud_score(actions: list[Action], expected_flag_fraud: bool) -> float:
    flagged = any(action.action_type == "flag_fraud" for action in actions)
    return 1.0 if flagged == expected_flag_fraud else 0.0


def _policy_score(actions: list[Action], ground_truth: GroundTruthPayload) -> float:
    snapshot = TicketSnapshot.model_validate(ground_truth["snapshot"])
    return (
        1.0
        if policies_satisfied(
            actions,
            snapshot,
            float(ground_truth["issue_age_hours"]),
            ground_truth["policy_version"],
        )
        else 0.0
    )


def easy_grader(actions: list[Action], ground_truth: GroundTruthPayload) -> float:
    components = easy_components(actions, ground_truth)
    return round(
        0.35 * components["category_correct"]
        + 0.3 * components["priority_correct"]
        + 0.2 * components["policy_compliance"]
        + 0.15 * components["fraud_handling"],
        4,
    )


def easy_components(actions: list[Action], ground_truth: GroundTruthPayload) -> dict[str, float]:
    return {
        "category_correct": _categorize_score(actions, ground_truth["expected_category"]),
        "priority_correct": _priority_score(actions, ground_truth["expected_priority"]),
        "policy_compliance": _policy_score(actions, ground_truth),
        "fraud_handling": _fraud_score(actions, bool(ground_truth["expected_flag_fraud"])),
    }


def medium_grader(actions: list[Action], ground_truth: GroundTruthPayload) -> float:
    components = medium_components(actions, ground_truth)
    base_score = round(
        0.2 * components["ambiguity_recognition"]
        + 0.15 * components["clarifying_question_quality"]
        + 0.15 * components["policy_compliance"]
        + 0.1 * components["category_correct"]
        + 0.1 * components["priority_correct"]
        + 0.2 * components["response_appropriateness"]
        + 0.1 * components["fraud_handling"],
        4,
    )
    return round(base_score * components["sequence_penalty_factor"], 4)


def medium_components(actions: list[Action], ground_truth: GroundTruthPayload) -> dict[str, float]:
    request_info_action = latest_action(actions, "request_info")
    draft_action = latest_action(actions, "draft_response")
    response_rubric = _response_rubric(
        draft_action.response_text if draft_action else None,
        ground_truth["response_keywords"],
        ground_truth["history_keywords"],
        ground_truth,
    )
    return {
        "ambiguity_recognition": 1.0 if request_info_action else 0.0,
        "clarifying_question_quality": request_info_quality(request_info_action, ground_truth),
        "policy_compliance": _policy_score(actions, ground_truth),
        "category_correct": _categorize_score(actions, ground_truth["expected_category"]),
        "priority_correct": _priority_score(actions, ground_truth["expected_priority"]),
        "response_appropriateness": response_rubric["response_quality"],
        "case_specific_facts": response_rubric["case_specific_facts"],
        "history_acknowledgment": response_rubric["history_acknowledgment"],
        "next_step_actionability": response_rubric["next_step_actionability"],
        "tone_acknowledgment": response_rubric["tone_acknowledgment"],
        "anti_stuffing": response_rubric["anti_stuffing"],
        "coherence_gate": response_rubric["coherence_gate"],
        "sequence_penalty_factor": _sequencing_penalty_factor(actions, ground_truth),
        "fraud_handling": _fraud_score(actions, bool(ground_truth["expected_flag_fraud"])),
    }


def hard_grader(actions: list[Action], ground_truth: GroundTruthPayload) -> float:
    components = hard_components(actions, ground_truth)
    base_score = round(
        0.1 * components["temporal_reasoning"]
        + 0.1 * components["policy_compliance"]
        + 0.1 * components["escalation_accuracy"]
        + 0.1 * components["contradiction_detection"]
        + 0.05 * components["ordering_correctness"]
        + 0.2 * components["history_acknowledgment"]
        + 0.25 * components["response_completeness"]
        + 0.1 * components["fraud_handling"],
        4,
    )
    return round(base_score * components["sequence_penalty_factor"], 4)


def hard_components(actions: list[Action], ground_truth: GroundTruthPayload) -> dict[str, float]:
    draft_action = latest_action(actions, "draft_response")
    response_rubric = _response_rubric(
        draft_action.response_text if draft_action else None,
        ground_truth["response_keywords"],
        ground_truth["history_keywords"],
        ground_truth,
    )
    category_score = _categorize_score(actions, ground_truth["expected_category"])
    policy_score = 1.0 if _policy_score(actions, ground_truth) == 1.0 and category_score == 1.0 else 0.0
    response_completeness = round(
        (
            0.45 * response_rubric["case_specific_facts"]
            + 0.35 * response_rubric["next_step_actionability"]
            + 0.2 * response_rubric["tone_acknowledgment"]
        )
        * response_rubric["anti_stuffing"],
        4,
    )
    response_completeness = round(response_completeness * response_rubric["coherence_gate"], 4)
    if response_rubric["llm_judge_used"]:
        response_completeness = round(
            (response_completeness + response_rubric["response_quality"]) / 2,
            4,
        )
    return {
        "temporal_reasoning": _priority_score(actions, ground_truth["expected_priority"]),
        "policy_compliance": policy_score,
        "escalation_accuracy": _escalation_score(actions, ground_truth["expected_escalation"]),
        "contradiction_detection": _contradiction_detection_score(actions, ground_truth),
        "ordering_correctness": _ordering_correctness_score(actions, ground_truth),
        "history_acknowledgment": response_rubric["history_acknowledgment"],
        "response_completeness": response_completeness,
        "case_specific_facts": response_rubric["case_specific_facts"],
        "next_step_actionability": response_rubric["next_step_actionability"],
        "tone_acknowledgment": response_rubric["tone_acknowledgment"],
        "anti_stuffing": response_rubric["anti_stuffing"],
        "coherence_gate": response_rubric["coherence_gate"],
        "sequence_penalty_factor": _sequencing_penalty_factor(actions, ground_truth),
        "fraud_handling": _fraud_score(actions, bool(ground_truth["expected_flag_fraud"])),
    }


def grade_actions(actions: list[Action], ground_truth: GroundTruthPayload) -> float:
    difficulty = ground_truth["difficulty"]
    if difficulty == "easy":
        return easy_grader(actions, ground_truth)
    if difficulty == "medium":
        return medium_grader(actions, ground_truth)
    return hard_grader(actions, ground_truth)


def component_scores(actions: list[Action], ground_truth: GroundTruthPayload) -> dict[str, float]:
    difficulty = ground_truth["difficulty"]
    if difficulty == "easy":
        return easy_components(actions, ground_truth)
    if difficulty == "medium":
        return medium_components(actions, ground_truth)
    return hard_components(actions, ground_truth)
