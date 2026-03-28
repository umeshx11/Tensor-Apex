from __future__ import annotations

from datetime import datetime, timedelta

from faker import Faker

from .models import EmailMessage, GroundTruth, TaskScenario, TicketSnapshot

FAKER_SEED = 20260328


def _faker() -> Faker:
    fake = Faker()
    fake.seed_instance(FAKER_SEED)
    return fake


def _message(
    fake: Faker,
    message_id: str,
    timestamp: datetime,
    subject: str,
    body: str,
    direction: str = "customer",
    sender_name: str | None = None,
    sender_email: str | None = None,
) -> EmailMessage:
    name = sender_name or fake.name()
    email = sender_email or fake.email()
    return EmailMessage(
        message_id=message_id,
        direction=direction,
        sender_name=name,
        sender_email=email,
        timestamp=timestamp,
        subject=subject,
        body=body,
    )


def build_scenarios() -> list[TaskScenario]:
    fake = _faker()
    base_now = datetime(2026, 3, 28, 10, 0, 0)

    scenarios: list[TaskScenario] = []
    scenarios.extend(_build_easy(fake, base_now))
    scenarios.extend(_build_medium(fake, base_now))
    scenarios.extend(_build_hard(fake, base_now))
    return scenarios


def _build_easy(fake: Faker, base_now: datetime) -> list[TaskScenario]:
    scenarios: list[TaskScenario] = []
    templates = [
        {
            "scenario_id": "easy_vip_refund",
            "title": "VIP refund request over threshold",
            "age_hours": 18,
            "sender_tier": "vip",
            "account_flags": [],
            "refund_amount": 650.0,
            "subject": "Refund for duplicate annual charge",
            "body": "I was charged twice for the enterprise annual plan and need the $650 duplicate payment refunded.",
            "expected_category": "billing",
            "expected_priority": "high",
            "expected_escalation": True,
            "expected_escalation_reason": "Refund exceeds $500.",
        },
        {
            "scenario_id": "easy_suspended_account",
            "title": "Suspended account billing route",
            "age_hours": 9,
            "sender_tier": "standard",
            "account_flags": ["suspended"],
            "refund_amount": None,
            "subject": "Why is my payment access blocked?",
            "body": "My account was suspended after yesterday's invoice and I need help updating the card on file.",
            "expected_category": "billing",
            "expected_priority": "medium",
            "expected_escalation": False,
            "expected_escalation_reason": None,
        },
        {
            "scenario_id": "easy_legal_threat",
            "title": "Explicit legal escalation",
            "age_hours": 6,
            "sender_tier": "standard",
            "account_flags": [],
            "refund_amount": 120.0,
            "subject": "Final warning before legal action",
            "body": "If this charge is not corrected today, I will pursue legal action and contact my lawyer.",
            "expected_category": "legal",
            "expected_priority": "urgent",
            "expected_escalation": True,
            "expected_escalation_reason": "Customer mentioned legal action.",
        },
        {
            "scenario_id": "easy_sla_breach",
            "title": "Aged low-tone billing ticket",
            "age_hours": 132,
            "sender_tier": "standard",
            "account_flags": [],
            "refund_amount": None,
            "subject": "Quick question about the March invoice",
            "body": "Could you explain the service adjustment listed on line 4 of my invoice?",
            "expected_category": "billing",
            "expected_priority": "urgent",
            "expected_escalation": False,
            "expected_escalation_reason": None,
        },
    ]

    for index, template in enumerate(templates, start=1):
        first_message_time = base_now - timedelta(hours=template["age_hours"])
        sender_name = fake.name()
        sender_email = fake.email()
        snapshot = TicketSnapshot(
            ticket_id=f"T-EASY-{index:03d}",
            thread=[
                _message(
                    fake,
                    f"{template['scenario_id']}_m1",
                    first_message_time,
                    template["subject"],
                    template["body"],
                    sender_name=sender_name,
                    sender_email=sender_email,
                )
            ],
            sender_tier=template["sender_tier"],
            account_flags=template["account_flags"],
            refund_amount=template["refund_amount"],
            order_id=f"ORD-{8000 + index}",
            visible_problem_type="billing",
        )
        scenarios.append(
            TaskScenario(
                scenario_id=template["scenario_id"],
                difficulty="easy",
                title=template["title"],
                objective="Classify the ticket, set the correct priority, and honor mandatory business policy rules.",
                max_steps=4,
                now=base_now,
                initial_snapshot=snapshot,
                ground_truth=GroundTruth(
                    expected_category=template["expected_category"],
                    expected_priority=template["expected_priority"],
                    expected_escalation=template["expected_escalation"],
                    expected_escalation_reason=template["expected_escalation_reason"],
                    completion_action_types=["categorize", "set_priority"]
                    + (["escalate"] if template["expected_escalation"] else []),
                ),
            )
        )

    return scenarios


def _build_medium(fake: Faker, base_now: datetime) -> list[TaskScenario]:
    scenarios: list[TaskScenario] = []
    templates = [
        {
            "scenario_id": "medium_charge_or_bug",
            "title": "Ambiguous charge issue for VIP customer",
            "age_hours": 14,
            "sender_tier": "vip",
            "account_flags": [],
            "subject": "The charge issue is still happening",
            "initial_body": "Can you fix the charge problem on my account?",
            "clarification_body": "It is the duplicate renewal charge from invoice INV-882. Please refund the extra $720.",
            "refund_amount": 720.0,
            "expected_category": "billing",
            "expected_priority": "high",
            "expected_escalation": True,
            "response_keywords": ["refund", "escalated", "duplicate"],
            "clarification_keywords": ["charge", "invoice", "account"],
        },
        {
            "scenario_id": "medium_same_problem",
            "title": "Ambiguous repeated problem without visible history",
            "age_hours": 11,
            "sender_tier": "standard",
            "account_flags": [],
            "subject": "Same problem as last time",
            "initial_body": "I am hitting the same problem again.",
            "clarification_body": "The desktop app will not load past the spinning logo after the update.",
            "refund_amount": None,
            "expected_category": "technical_support",
            "expected_priority": "medium",
            "expected_escalation": False,
            "response_keywords": ["troubleshoot", "app", "update"],
            "clarification_keywords": ["problem", "last time", "error"],
        },
        {
            "scenario_id": "medium_exchange_or_refund",
            "title": "Return request that becomes a high-value refund",
            "age_hours": 90,
            "sender_tier": "standard",
            "account_flags": [],
            "subject": "Need help with the laptop order",
            "initial_body": "I need help with my laptop order because the same issue keeps coming back.",
            "clarification_body": "I do not want another replacement. I need a refund for the $680 device instead.",
            "refund_amount": 680.0,
            "expected_category": "billing",
            "expected_priority": "urgent",
            "expected_escalation": True,
            "response_keywords": ["refund", "escalated", "wait"],
            "clarification_keywords": ["order", "refund", "replacement"],
        },
        {
            "scenario_id": "medium_suspended_not_working",
            "title": "Suspended account with unclear billing issue",
            "age_hours": 20,
            "sender_tier": "standard",
            "account_flags": ["suspended"],
            "subject": "My account is not working",
            "initial_body": "My account is not working and I cannot fix it.",
            "clarification_body": "I need to update the card on my suspended account because the invoice payment failed.",
            "refund_amount": None,
            "expected_category": "billing",
            "expected_priority": "medium",
            "expected_escalation": False,
            "response_keywords": ["billing", "card", "suspended"],
            "clarification_keywords": ["account", "invoice", "payment"],
        },
    ]

    for index, template in enumerate(templates, start=1):
        first_message_time = base_now - timedelta(hours=template["age_hours"])
        clarification_time = first_message_time + timedelta(hours=2)
        sender_name = fake.name()
        sender_email = fake.email()
        initial_snapshot = TicketSnapshot(
            ticket_id=f"T-MED-{index:03d}",
            thread=[
                _message(
                    fake,
                    f"{template['scenario_id']}_m1",
                    first_message_time,
                    template["subject"],
                    template["initial_body"],
                    sender_name=sender_name,
                    sender_email=sender_email,
                )
            ],
            sender_tier=template["sender_tier"],
            account_flags=template["account_flags"],
            refund_amount=None,
            order_id=f"MED-{9100 + index}",
            visible_problem_type=None,
        )
        clarification_snapshot = TicketSnapshot(
            ticket_id=f"T-MED-{index:03d}",
            thread=initial_snapshot.thread
            + [
                _message(
                    fake,
                    f"{template['scenario_id']}_m2",
                    clarification_time,
                    f"Re: {template['subject']}",
                    template["clarification_body"],
                    sender_name=sender_name,
                    sender_email=sender_email,
                )
            ],
            sender_tier=template["sender_tier"],
            account_flags=template["account_flags"],
            refund_amount=template["refund_amount"],
            order_id=f"MED-{9100 + index}",
            visible_problem_type="billing" if template["expected_category"] == "billing" else "technical_support",
        )
        scenarios.append(
            TaskScenario(
                scenario_id=template["scenario_id"],
                difficulty="medium",
                title=template["title"],
                objective="Recognize ambiguity, request clarification, then resolve the issue while following policy rules.",
                max_steps=6,
                now=base_now,
                initial_snapshot=initial_snapshot,
                clarification_snapshot=clarification_snapshot,
                ground_truth=GroundTruth(
                    expected_category=template["expected_category"],
                    expected_priority=template["expected_priority"],
                    expected_escalation=template["expected_escalation"],
                    expected_escalation_reason="Escalation required by policy." if template["expected_escalation"] else None,
                    requires_request_info=True,
                    request_info_first_required=True,
                    clarification_keywords=template["clarification_keywords"],
                    response_keywords=template["response_keywords"],
                    completion_action_types=["request_info", "categorize", "set_priority", "draft_response"]
                    + (["escalate"] if template["expected_escalation"] else []),
                    ambiguous=True,
                ),
            )
        )

    return scenarios


def _build_hard(fake: Faker, base_now: datetime) -> list[TaskScenario]:
    scenarios: list[TaskScenario] = []
    templates = [
        {
            "scenario_id": "hard_vip_refund_lawyer",
            "title": "VIP refund thread with legal pressure",
            "age_hours": 124,
            "sender_tier": "vip",
            "account_flags": [],
            "refund_amount": 700.0,
            "expected_category": "legal",
            "expected_priority": "urgent",
            "expected_escalation": True,
            "response_keywords": ["refund", "escalated", "review", "delay"],
            "history_keywords": ["waiting", "five days", "follow-up"],
            "thread_bodies": [
                "I was charged $700 twice for our annual renewal. Please process the duplicate refund.",
                "We are reviewing the billing transaction and will update you soon.",
                "It has been five days with no refund. If this keeps dragging on I will speak with my lawyer.",
            ],
        },
        {
            "scenario_id": "hard_old_invoice_question",
            "title": "Low-tone invoice question that breached SLA",
            "age_hours": 146,
            "sender_tier": "standard",
            "account_flags": [],
            "refund_amount": None,
            "expected_category": "billing",
            "expected_priority": "urgent",
            "expected_escalation": False,
            "response_keywords": ["invoice", "review", "update"],
            "history_keywords": ["waiting", "follow-up"],
            "thread_bodies": [
                "Could you clarify the platform fee on my March invoice when you have a moment?",
                "Following up on the invoice fee question from earlier this week.",
            ],
        },
        {
            "scenario_id": "hard_ambiguous_device_resolution",
            "title": "Multi-turn device issue that still needs clarification",
            "age_hours": 98,
            "sender_tier": "standard",
            "account_flags": [],
            "refund_amount": 640.0,
            "expected_category": "billing",
            "expected_priority": "urgent",
            "expected_escalation": True,
            "response_keywords": ["refund", "escalated", "device", "delay"],
            "history_keywords": ["waiting", "replacement", "follow-up"],
            "thread_bodies": [
                "The replacement laptop is failing just like the first one.",
                "Support asked whether I wanted another swap or money back and I said I still needed help.",
                "This replacement issue has been unresolved for four days and I need this fixed now.",
            ],
            "clarification_body": "Please stop the replacement process and refund the $640 order instead.",
            "clarification_keywords": ["replacement", "refund", "want"],
        },
        {
            "scenario_id": "hard_suspended_payment_thread",
            "title": "Suspended account thread with repeated billing friction",
            "age_hours": 96,
            "sender_tier": "standard",
            "account_flags": ["suspended"],
            "refund_amount": None,
            "expected_category": "billing",
            "expected_priority": "urgent",
            "expected_escalation": False,
            "response_keywords": ["billing", "payment", "review"],
            "history_keywords": ["waiting", "follow-up", "suspended"],
            "thread_bodies": [
                "My account was suspended after the automatic payment failed.",
                "I sent the new card details yesterday but the account is still locked.",
                "Following up again because I cannot access billing settings and this is now urgent for our team.",
            ],
        },
    ]

    for index, template in enumerate(templates, start=1):
        start_time = base_now - timedelta(hours=template["age_hours"])
        sender_name = fake.name()
        sender_email = fake.email()
        thread = []
        for message_index, body in enumerate(template["thread_bodies"], start=1):
            thread.append(
                _message(
                    fake,
                    f"{template['scenario_id']}_m{message_index}",
                    start_time + timedelta(hours=message_index * 6),
                    template["title"],
                    body,
                    direction="customer" if message_index != 2 else "agent",
                    sender_name=sender_name,
                    sender_email=sender_email,
                )
            )
        initial_snapshot = TicketSnapshot(
            ticket_id=f"T-HARD-{index:03d}",
            thread=thread,
            sender_tier=template["sender_tier"],
            account_flags=template["account_flags"],
            refund_amount=None if template["scenario_id"] == "hard_ambiguous_device_resolution" else template["refund_amount"],
            order_id=f"HARD-{9900 + index}",
            visible_problem_type=None,
        )

        clarification_snapshot = None
        requires_request_info = template["scenario_id"] == "hard_ambiguous_device_resolution"
        if requires_request_info:
            clarification_snapshot = TicketSnapshot(
                ticket_id=f"T-HARD-{index:03d}",
                thread=thread
                + [
                    _message(
                        fake,
                        f"{template['scenario_id']}_m4",
                        start_time + timedelta(hours=32),
                        f"Re: {template['title']}",
                        template["clarification_body"],
                        sender_name=sender_name,
                        sender_email=sender_email,
                    )
                ],
                sender_tier=template["sender_tier"],
                account_flags=template["account_flags"],
                refund_amount=template["refund_amount"],
                order_id=f"HARD-{9900 + index}",
                visible_problem_type="billing",
            )

        scenarios.append(
            TaskScenario(
                scenario_id=template["scenario_id"],
                difficulty="hard",
                title=template["title"],
                objective="Resolve a realistic multi-turn thread with temporal pressure, policy constraints, and customer-history awareness.",
                max_steps=7,
                now=base_now,
                initial_snapshot=initial_snapshot,
                clarification_snapshot=clarification_snapshot,
                ground_truth=GroundTruth(
                    expected_category=template["expected_category"],
                    expected_priority=template["expected_priority"],
                    expected_escalation=template["expected_escalation"],
                    expected_escalation_reason="Escalation required by policy." if template["expected_escalation"] else None,
                    requires_request_info=requires_request_info,
                    request_info_first_required=requires_request_info,
                    clarification_keywords=template.get("clarification_keywords", []),
                    response_keywords=template["response_keywords"],
                    history_keywords=template["history_keywords"],
                    completion_action_types=(( ["request_info"] if requires_request_info else [] )
                    + ["categorize", "set_priority", "draft_response"]
                    + (["escalate"] if template["expected_escalation"] else [])),
                    ambiguous=requires_request_info,
                ),
            )
        )

    return scenarios
