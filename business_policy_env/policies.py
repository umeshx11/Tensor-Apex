from __future__ import annotations

from collections.abc import Iterable

from .models import Action, Priority, TicketSnapshot

POLICY_RULES = [
    "Refunds over $500 require escalation.",
    "VIP customers require high or urgent priority.",
    "Issues open over 72 hours require urgent priority.",
    "Complaints mentioning legal action or lawsuits require immediate escalation.",
    "Suspended accounts must be routed to the billing team.",
]

PRIORITY_ORDER: dict[Priority, int] = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "urgent": 3,
}


def compute_policy_expectations(snapshot: TicketSnapshot, issue_age_hours: float) -> dict[str, object]:
    latest_message = snapshot.thread[-1]
    body_text = latest_message.body.lower()
    expectations: dict[str, object] = {
        "requires_escalation": False,
        "min_priority": None,
        "forced_category": None,
        "triggered_rules": [],
    }

    if snapshot.refund_amount and snapshot.refund_amount > 500:
        expectations["requires_escalation"] = True
        expectations["triggered_rules"].append("refund_gt_500")

    if snapshot.sender_tier == "vip":
        expectations["min_priority"] = "high"
        expectations["triggered_rules"].append("vip_priority")

    if issue_age_hours > 72:
        expectations["min_priority"] = "urgent"
        expectations["triggered_rules"].append("sla_breach")

    if "lawsuit" in body_text or "legal action" in body_text or "lawyer" in body_text:
        expectations["requires_escalation"] = True
        expectations["triggered_rules"].append("legal_threat")

    if "suspended" in snapshot.account_flags:
        expectations["forced_category"] = "billing"
        expectations["triggered_rules"].append("suspended_to_billing")

    return expectations


def check_policy_violations(action: Action, snapshot: TicketSnapshot, issue_age_hours: float) -> list[str]:
    expectations = compute_policy_expectations(snapshot, issue_age_hours)
    violations: list[str] = []

    min_priority = expectations["min_priority"]
    if action.action_type == "set_priority" and min_priority:
        if PRIORITY_ORDER[action.priority] < PRIORITY_ORDER[min_priority]:  # type: ignore[index]
            violations.append(f"Priority must be at least {min_priority}.")

    if action.action_type == "categorize" and expectations["forced_category"]:
        if action.category != expectations["forced_category"]:
            violations.append(f"Category must be {expectations['forced_category']}.")

    if action.action_type in {"draft_response", "mark_spam"} and expectations["requires_escalation"]:
        violations.append("The ticket requires escalation before resolution.")

    return violations


def policies_satisfied(actions: Iterable[Action], snapshot: TicketSnapshot, issue_age_hours: float) -> bool:
    expectations = compute_policy_expectations(snapshot, issue_age_hours)
    latest_priority = None
    latest_category = None
    escalated = False

    for action in actions:
        if action.action_type == "set_priority":
            latest_priority = action.priority
        elif action.action_type == "categorize":
            latest_category = action.category
        elif action.action_type == "escalate":
            escalated = True

    min_priority = expectations["min_priority"]
    if min_priority and latest_priority is None:
        return False
    if min_priority and PRIORITY_ORDER[latest_priority] < PRIORITY_ORDER[min_priority]:  # type: ignore[index]
        return False

    forced_category = expectations["forced_category"]
    if forced_category and latest_category != forced_category:
        return False

    if expectations["requires_escalation"] and not escalated:
        return False

    return True
