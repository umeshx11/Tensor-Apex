from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

from faker import Faker

from .models import (
    ActionType,
    Category,
    Difficulty,
    EmailMessage,
    GroundTruth,
    PolicyVersion,
    Priority,
    SenderTier,
    TaskScenario,
    TicketSnapshot,
)


@dataclass
class ScenarioTemplate:
    scenario_id: str
    title: str
    subject: str
    difficulty: Difficulty
    sender_tier: SenderTier
    account_flags: list[str]
    refund_amount: float | None
    age_hours: float
    thread_bodies: list[str]
    expected_category: Category
    expected_priority: Priority
    expected_escalation: bool
    requires_request_info: bool
    request_info_first_required: bool
    response_keywords: list[str]
    history_keywords: list[str]
    clarification_keywords: list[str]
    conflict_keywords: list[str] = field(default_factory=list)
    required_action_order: list[ActionType] = field(default_factory=list)
    clarification_body: str | None = None
    legal_language: bool = False
    suspended_account: bool = False
    expected_flag_fraud: bool = False
    fraud_keywords: list[str] = field(default_factory=list)
    refund_range: tuple[float, float] | None = None
    age_range: tuple[float, float] | None = None
    policy_version: PolicyVersion | Literal["random"] = "v1"
    max_steps: int = 6
    objective: str | None = None
    expected_escalation_reason: str | None = None
    thread_directions: list[Literal["customer", "agent", "system"]] = field(default_factory=list)
    visible_problem_type: str | None = None
    partial_observability: bool = False
    hidden_account_flag_count: int = 0
    policy_transition_step: int | None = None
    policy_transition_to: PolicyVersion | None = None


EASY_TEMPLATES: list[ScenarioTemplate] = [
    ScenarioTemplate(
        scenario_id="easy_vip_refund",
        title="VIP refund request over threshold",
        subject="Refund for duplicate annual charge",
        difficulty="easy",
        sender_tier="vip",
        account_flags=[],
        refund_amount=650.0,
        age_hours=18,
        thread_bodies=["I was charged twice and need the $650 duplicate payment refunded."],
        expected_category="billing",
        expected_priority="high",
        expected_escalation=True,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=[],
        history_keywords=[],
        clarification_keywords=[],
        expected_escalation_reason="Refund exceeds $500.",
        max_steps=5,
        visible_problem_type="billing",
    ),
    ScenarioTemplate(
        scenario_id="easy_suspended_account",
        title="Suspended account billing route",
        subject="Why is my payment access blocked?",
        difficulty="easy",
        sender_tier="standard",
        account_flags=["suspended"],
        refund_amount=None,
        age_hours=9,
        thread_bodies=["My account was suspended after yesterday's invoice and I need billing help."],
        expected_category="billing",
        expected_priority="medium",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=[],
        history_keywords=[],
        clarification_keywords=[],
        suspended_account=True,
        max_steps=5,
        visible_problem_type="billing",
    ),
    ScenarioTemplate(
        scenario_id="easy_legal_threat",
        title="Explicit legal escalation",
        subject="Final warning before legal action",
        difficulty="easy",
        sender_tier="standard",
        account_flags=[],
        refund_amount=120.0,
        age_hours=6,
        thread_bodies=["If this charge is not corrected today, I will take legal action and contact my lawyer."],
        expected_category="legal",
        expected_priority="urgent",
        expected_escalation=True,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=[],
        history_keywords=[],
        clarification_keywords=[],
        legal_language=True,
        expected_escalation_reason="Customer mentioned legal action.",
        max_steps=5,
        visible_problem_type="legal",
    ),
    ScenarioTemplate(
        scenario_id="easy_sla_breach",
        title="Aged low-tone billing ticket",
        subject="Quick question about the March invoice",
        difficulty="easy",
        sender_tier="standard",
        account_flags=[],
        refund_amount=None,
        age_hours=132,
        thread_bodies=["Could you explain the service adjustment listed on line 4 of my invoice?"],
        expected_category="billing",
        expected_priority="urgent",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=[],
        history_keywords=[],
        clarification_keywords=[],
        max_steps=5,
        visible_problem_type="billing",
    ),
    ScenarioTemplate(
        scenario_id="easy_standard_small_refund",
        title="Standard refund below escalation threshold",
        subject="Please refund an accidental duplicate add-on",
        difficulty="easy",
        sender_tier="standard",
        account_flags=[],
        refund_amount=85.0,
        age_hours=16,
        thread_bodies=["I accidentally bought the same add-on twice and need the extra $85 refunded."],
        expected_category="billing",
        expected_priority="medium",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=[],
        history_keywords=[],
        clarification_keywords=[],
        max_steps=5,
        visible_problem_type="billing",
    ),
    ScenarioTemplate(
        scenario_id="easy_vip_technical_issue",
        title="VIP technical issue without escalation",
        subject="Desktop app crash after patch",
        difficulty="easy",
        sender_tier="vip",
        account_flags=[],
        refund_amount=None,
        age_hours=7,
        thread_bodies=["Our desktop app crashes right after login since today's patch."],
        expected_category="technical_support",
        expected_priority="high",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=[],
        history_keywords=[],
        clarification_keywords=[],
        max_steps=5,
        visible_problem_type="technical_support",
    ),
    ScenarioTemplate(
        scenario_id="easy_vip_legal_dual_trigger",
        title="VIP legal threat with overlapping rules",
        subject="Counsel review if this is not corrected today",
        difficulty="easy",
        sender_tier="vip",
        account_flags=[],
        refund_amount=210.0,
        age_hours=28,
        thread_bodies=[
            "I am a VIP account holder and our legal counsel will proceed if this charge remains unresolved."
        ],
        expected_category="legal",
        expected_priority="urgent",
        expected_escalation=True,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=[],
        history_keywords=[],
        clarification_keywords=[],
        legal_language=True,
        expected_escalation_reason="Customer mentioned legal action.",
        max_steps=5,
        visible_problem_type="legal",
    ),
    ScenarioTemplate(
        scenario_id="easy_suspended_sla_breach",
        title="Suspended account and SLA breach",
        subject="Still locked out of billing controls",
        difficulty="easy",
        sender_tier="standard",
        account_flags=["suspended"],
        refund_amount=None,
        age_hours=88,
        thread_bodies=["My account is still suspended and I cannot update billing settings after several days."],
        expected_category="billing",
        expected_priority="urgent",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=[],
        history_keywords=[],
        clarification_keywords=[],
        suspended_account=True,
        max_steps=5,
        visible_problem_type="billing",
    ),
    ScenarioTemplate(
        scenario_id="easy_spam_detection",
        title="Obvious spam outreach",
        subject="Guaranteed growth hack click now",
        difficulty="easy",
        sender_tier="standard",
        account_flags=[],
        refund_amount=None,
        age_hours=2,
        thread_bodies=["Buy fake reviews and guaranteed traffic now. Click this short link to activate your bonus."],
        expected_category="spam",
        expected_priority="low",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=[],
        history_keywords=[],
        clarification_keywords=[],
        max_steps=5,
        visible_problem_type="spam",
    ),
    ScenarioTemplate(
        scenario_id="easy_sla_marginal",
        title="Clean baseline close to SLA threshold",
        subject="Need onboarding help",
        difficulty="easy",
        sender_tier="standard",
        account_flags=[],
        refund_amount=None,
        age_hours=71,
        thread_bodies=["Could someone point me to the right setup guide for our new workspace?"],
        expected_category="customer_success",
        expected_priority="medium",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=[],
        history_keywords=[],
        clarification_keywords=[],
        max_steps=5,
        visible_problem_type="customer_success",
    ),
]


MEDIUM_TEMPLATES: list[ScenarioTemplate] = [
    ScenarioTemplate(
        scenario_id="medium_charge_or_bug",
        title="Ambiguous charge issue for VIP customer",
        subject="The charge issue is still happening",
        difficulty="medium",
        sender_tier="vip",
        account_flags=[],
        refund_amount=720.0,
        age_hours=14,
        thread_bodies=["Can you fix the charge problem on my account?"],
        expected_category="billing",
        expected_priority="high",
        expected_escalation=True,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["refund", "escalated", "duplicate"],
        history_keywords=["charge", "problem"],
        clarification_keywords=["charge", "invoice", "account"],
        clarification_body="It is the duplicate renewal charge from invoice INV-882. Please refund the extra $720.",
        expected_escalation_reason="Refund exceeds $500.",
        max_steps=7,
    ),
    ScenarioTemplate(
        scenario_id="medium_same_problem",
        title="Ambiguous repeated problem without visible history",
        subject="Same problem as last time",
        difficulty="medium",
        sender_tier="standard",
        account_flags=[],
        refund_amount=None,
        age_hours=11,
        thread_bodies=["I am hitting the same problem again."],
        expected_category="technical_support",
        expected_priority="medium",
        expected_escalation=False,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["troubleshoot", "app", "update"],
        history_keywords=["same problem", "again"],
        clarification_keywords=["problem", "last time", "error"],
        clarification_body="The desktop app will not load past the spinning logo after the update.",
        max_steps=7,
    ),
    ScenarioTemplate(
        scenario_id="medium_exchange_or_refund",
        title="Return request that becomes a high-value refund",
        subject="Need help with the laptop order",
        difficulty="medium",
        sender_tier="standard",
        account_flags=[],
        refund_amount=680.0,
        age_hours=90,
        thread_bodies=["I need help with my laptop order because the same issue keeps coming back."],
        expected_category="billing",
        expected_priority="urgent",
        expected_escalation=True,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["refund", "escalated", "wait"],
        history_keywords=["issue", "coming back"],
        clarification_keywords=["order", "refund", "replacement"],
        clarification_body="I do not want another replacement. I need a refund for the $680 device instead.",
        expected_escalation_reason="Refund exceeds $500.",
        max_steps=7,
    ),
    ScenarioTemplate(
        scenario_id="medium_vip_small_refund_after_clarification",
        title="Ambiguous VIP complaint resolved below escalation threshold",
        subject="You charged us incorrectly again",
        difficulty="medium",
        sender_tier="vip",
        account_flags=[],
        refund_amount=130.0,
        age_hours=22,
        thread_bodies=["This charge problem happened again and we need it corrected."],
        expected_category="billing",
        expected_priority="high",
        expected_escalation=False,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["refund", "invoice", "confirmed"],
        history_keywords=["again", "charge"],
        clarification_keywords=["amount", "invoice", "outcome"],
        clarification_body="It is a duplicate $130 line item on invoice INV-909 and a refund is all we need.",
        max_steps=7,
    ),
    ScenarioTemplate(
        scenario_id="medium_returns_exchange_ambiguity",
        title="Ambiguous return versus exchange request",
        subject="Need a different size",
        difficulty="medium",
        sender_tier="standard",
        account_flags=[],
        refund_amount=None,
        age_hours=19,
        thread_bodies=["The jacket did not fit and I need a different option."],
        expected_category="returns",
        expected_priority="medium",
        expected_escalation=False,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["exchange", "size", "return label"],
        history_keywords=["different option"],
        clarification_keywords=["refund", "exchange", "size"],
        clarification_body="Please exchange it for one size up; I do not need a refund.",
        max_steps=7,
    ),
    ScenarioTemplate(
        scenario_id="medium_suspended_tech_vs_billing",
        title="Suspended account with technical-vs-billing ambiguity",
        subject="Cannot sign in to manage my account",
        difficulty="medium",
        sender_tier="standard",
        account_flags=["suspended"],
        refund_amount=None,
        age_hours=26,
        thread_bodies=["I cannot log in and my account is unusable right now."],
        expected_category="billing",
        expected_priority="medium",
        expected_escalation=False,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["billing", "payment", "suspended"],
        history_keywords=["cannot log in"],
        clarification_keywords=["payment", "invoice", "suspended"],
        clarification_body="The login issue started after a failed invoice payment and the account was suspended.",
        suspended_account=True,
        max_steps=7,
    ),
    ScenarioTemplate(
        scenario_id="medium_multilingual_signal",
        title="Non-English message with weak keyword overlap",
        subject="Necesito ayuda urgente",
        difficulty="medium",
        sender_tier="standard",
        account_flags=[],
        refund_amount=None,
        age_hours=13,
        thread_bodies=["Necesito ayuda con mi cuenta, no funciona desde ayer."],
        expected_category="technical_support",
        expected_priority="medium",
        expected_escalation=False,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["app", "access", "support"],
        history_keywords=["desde ayer"],
        clarification_keywords=["error", "pantalla", "acceso"],
        clarification_body="La app muestra un error de acceso despues de iniciar sesion.",
        max_steps=7,
    ),
    ScenarioTemplate(
        scenario_id="medium_one_word_help",
        title="One-word inbound message",
        subject="Help",
        difficulty="medium",
        sender_tier="standard",
        account_flags=[],
        refund_amount=None,
        age_hours=4,
        thread_bodies=["Help"],
        expected_category="customer_success",
        expected_priority="medium",
        expected_escalation=False,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["support", "details", "assist"],
        history_keywords=["help"],
        clarification_keywords=["order", "account", "issue"],
        clarification_body="I need help enabling SSO for our workspace settings.",
        max_steps=7,
    ),
    ScenarioTemplate(
        scenario_id="medium_misleading_subject",
        title="Misleading subject line with conflicting body",
        subject="Refund status needed",
        difficulty="medium",
        sender_tier="standard",
        account_flags=[],
        refund_amount=None,
        age_hours=17,
        thread_bodies=["Ignore the subject line, my issue is that the mobile app freezes on launch."],
        expected_category="technical_support",
        expected_priority="medium",
        expected_escalation=False,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["mobile", "troubleshoot", "crash"],
        history_keywords=["subject line"],
        clarification_keywords=["device", "os", "error"],
        clarification_body="It crashes on iOS 18 right after I tap Sign in.",
        max_steps=7,
    ),
    ScenarioTemplate(
        scenario_id="medium_premier_same_day",
        title="Premier account policy-v2 same-day scenario",
        subject="Need setup guidance",
        difficulty="medium",
        sender_tier="premier",
        account_flags=[],
        refund_amount=None,
        age_hours=10,
        thread_bodies=["Can someone guide us on dashboard setup?"],
        expected_category="customer_success",
        expected_priority="high",
        expected_escalation=False,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["setup", "guide", "today"],
        history_keywords=["guidance"],
        clarification_keywords=["goal", "workspace", "configuration"],
        clarification_body="We need same-day onboarding guidance for three new workspace admins.",
        policy_version="v2",
        max_steps=7,
    ),
]


HARD_TEMPLATES: list[ScenarioTemplate] = [
    ScenarioTemplate(
        scenario_id="hard_vip_refund_lawyer",
        title="VIP refund thread with legal pressure",
        subject="Escalating duplicate renewal refund",
        difficulty="hard",
        sender_tier="vip",
        account_flags=[],
        refund_amount=700.0,
        age_hours=124,
        thread_bodies=[
            "I was charged $700 twice for our annual renewal. Please process the duplicate refund.",
            "We are reviewing the billing transaction and will update you soon.",
            "It has been five days with no refund. If this keeps dragging on I will speak with my lawyer.",
        ],
        thread_directions=["customer", "agent", "customer"],
        expected_category="legal",
        expected_priority="urgent",
        expected_escalation=True,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["refund", "escalated", "review", "delay"],
        history_keywords=["waiting", "five days", "follow-up"],
        clarification_keywords=[],
        legal_language=True,
        expected_escalation_reason="Customer mentioned legal action.",
        max_steps=8,
    ),
    ScenarioTemplate(
        scenario_id="hard_old_invoice_question",
        title="Low-tone invoice question that breached SLA",
        subject="Following up on invoice fee",
        difficulty="hard",
        sender_tier="standard",
        account_flags=[],
        refund_amount=None,
        age_hours=146,
        thread_bodies=[
            "Could you clarify the platform fee on my March invoice when you have a moment?",
            "Following up on the invoice fee question from earlier this week.",
        ],
        expected_category="billing",
        expected_priority="urgent",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["invoice", "review", "update"],
        history_keywords=["waiting", "follow-up"],
        clarification_keywords=[],
        max_steps=8,
    ),
    ScenarioTemplate(
        scenario_id="hard_ambiguous_device_resolution",
        title="Multi-turn device issue that still needs clarification",
        subject="Replacement device still failing",
        difficulty="hard",
        sender_tier="standard",
        account_flags=[],
        refund_amount=640.0,
        age_hours=98,
        thread_bodies=[
            "The replacement laptop is failing just like the first one.",
            "Support asked whether I wanted another swap or money back and I said I still needed help.",
            "This replacement issue has been unresolved for four days and I need this fixed now.",
        ],
        expected_category="billing",
        expected_priority="urgent",
        expected_escalation=True,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["refund", "escalated", "device", "delay"],
        history_keywords=["waiting", "replacement", "follow-up"],
        clarification_keywords=["replacement", "refund", "want"],
        clarification_body="Please stop the replacement process and refund the $640 order instead.",
        expected_escalation_reason="Refund exceeds $500.",
        max_steps=8,
    ),
    ScenarioTemplate(
        scenario_id="hard_suspended_payment_thread",
        title="Suspended account thread with repeated billing friction",
        subject="Still locked after payment update",
        difficulty="hard",
        sender_tier="standard",
        account_flags=["suspended"],
        refund_amount=None,
        age_hours=96,
        thread_bodies=[
            "My account was suspended after the automatic payment failed.",
            "I sent the new card details yesterday but the account is still locked.",
            "Following up again because I cannot access billing settings and this is now urgent for our team.",
        ],
        expected_category="billing",
        expected_priority="urgent",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["billing", "payment", "review"],
        history_keywords=["waiting", "follow-up", "suspended"],
        clarification_keywords=[],
        suspended_account=True,
        max_steps=8,
    ),
    ScenarioTemplate(
        scenario_id="hard_sla_cross_mid_thread",
        title="Priority changes mid-thread after crossing 72h",
        subject="Still unresolved after multiple follow-ups",
        difficulty="hard",
        sender_tier="standard",
        account_flags=[],
        refund_amount=None,
        age_hours=74,
        thread_bodies=[
            "I asked about this invoice adjustment yesterday and still need help.",
            "Checking in again because this was opened before the weekend.",
            "Now this has been over three days with no resolution.",
            "Please prioritize this immediately and confirm next steps.",
        ],
        expected_category="billing",
        expected_priority="urgent",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["priority", "urgent", "update"],
        history_keywords=["three days", "follow-up", "weekend"],
        clarification_keywords=[],
        max_steps=8,
    ),
    ScenarioTemplate(
        scenario_id="hard_previous_agent_failed_escalation",
        title="Customer escalating prior agent failure",
        subject="Previous response ignored escalation need",
        difficulty="hard",
        sender_tier="standard",
        account_flags=[],
        refund_amount=780.0,
        age_hours=84,
        thread_bodies=[
            "I requested a refund for duplicate charges last week.",
            "Agent reply: This does not need escalation and will be handled routinely.",
            "That was incorrect and I am escalating this failure now.",
        ],
        thread_directions=["customer", "agent", "customer"],
        expected_category="billing",
        expected_priority="urgent",
        expected_escalation=True,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["escalated", "review", "incorrect", "refund"],
        history_keywords=["last week", "failed", "agent"],
        clarification_keywords=[],
        expected_escalation_reason="Refund exceeds $500.",
        max_steps=8,
    ),
    ScenarioTemplate(
        scenario_id="hard_vip_became_suspended",
        title="VIP account became suspended mid-thread",
        subject="Urgent access interruption",
        difficulty="hard",
        sender_tier="vip",
        account_flags=["suspended"],
        refund_amount=None,
        age_hours=80,
        thread_bodies=[
            "As a VIP account we saw intermittent access errors yesterday.",
            "The issue got worse and now billing features are inaccessible.",
            "This morning our account shows suspended status and we need restoration.",
        ],
        expected_category="billing",
        expected_priority="urgent",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["billing", "restore", "urgent"],
        history_keywords=["vip", "suspended", "morning"],
        clarification_keywords=[],
        suspended_account=True,
        max_steps=8,
    ),
    ScenarioTemplate(
        scenario_id="hard_three_signal_precedence",
        title="Three conflicting policy signals with precedence",
        subject="Legal complaint about suspicious billing activity",
        difficulty="hard",
        sender_tier="premier",
        account_flags=["suspended", "fraud_risk"],
        refund_amount=930.0,
        age_hours=92,
        thread_bodies=[
            "Our suspended account shows multiple unauthorized card attempts and duplicate charges.",
            "If this is not escalated today our legal team will intervene.",
            "We need immediate fraud investigation and billing remediation.",
        ],
        expected_category="billing",
        expected_priority="urgent",
        expected_escalation=True,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["fraud", "escalated", "billing", "urgent"],
        history_keywords=["unauthorized", "legal", "suspended"],
        clarification_keywords=[],
        conflict_keywords=["fraud", "legal", "suspended"],
        required_action_order=["flag_fraud", "escalate", "draft_response"],
        legal_language=True,
        suspended_account=True,
        expected_flag_fraud=True,
        fraud_keywords=["unauthorized", "card attempts", "duplicate charges"],
        expected_escalation_reason="Policy escalation required by legal and high-value billing risk.",
        policy_version="v2",
        max_steps=8,
    ),
    ScenarioTemplate(
        scenario_id="hard_fraud_chargeback",
        title="Chargeback fraud pattern detection",
        subject="Bank reversals across multiple cards",
        difficulty="hard",
        sender_tier="standard",
        account_flags=["chargeback_risk"],
        refund_amount=120.0,
        age_hours=36,
        thread_bodies=[
            "We see repeated bank reversals tied to multiple cards and suspicious rapid transactions.",
            "Please investigate this as a potential chargeback fraud pattern.",
        ],
        expected_category="billing",
        expected_priority="medium",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["fraud", "investigate", "chargeback"],
        history_keywords=["multiple cards", "reversals"],
        clarification_keywords=[],
        expected_flag_fraud=True,
        fraud_keywords=["chargeback", "multiple cards"],
        policy_version="v2",
        max_steps=8,
    ),
    ScenarioTemplate(
        scenario_id="hard_fraud_account_takeover",
        title="Account takeover indicators in access complaints",
        subject="Unauthorized admin sessions detected",
        difficulty="hard",
        sender_tier="standard",
        account_flags=["ato_watch"],
        refund_amount=None,
        age_hours=48,
        thread_bodies=[
            "We noticed unknown admin logins and password resets from unfamiliar locations.",
            "This looks like account takeover behavior and needs fraud handling.",
        ],
        expected_category="technical_support",
        expected_priority="medium",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["fraud", "secure", "investigate"],
        history_keywords=["unknown logins", "password resets"],
        clarification_keywords=[],
        expected_flag_fraud=True,
        fraud_keywords=["account takeover", "unknown logins"],
        policy_version="v2",
        max_steps=8,
    ),
    ScenarioTemplate(
        scenario_id="hard_fraud_policy_abuse",
        title="Refund abuse and card-testing risk",
        subject="Suspicious rapid refund requests",
        difficulty="hard",
        sender_tier="standard",
        account_flags=["fraud_risk"],
        refund_amount=910.0,
        age_hours=30,
        thread_bodies=[
            "We received several high-value refund requests tied to newly added cards in one hour.",
            "This resembles card testing and policy abuse tied to chargeback fraud.",
        ],
        expected_category="billing",
        expected_priority="medium",
        expected_escalation=True,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["flagged", "fraud", "escalated", "review"],
        history_keywords=["card testing", "refund requests"],
        clarification_keywords=[],
        expected_flag_fraud=True,
        fraud_keywords=["card testing", "policy abuse", "chargeback"],
        expected_escalation_reason="Refund exceeds $500.",
        policy_version="v2",
        max_steps=8,
    ),
    ScenarioTemplate(
        scenario_id="hard_premier_same_day_v2",
        title="Premier account requiring same-day handling",
        subject="Need same-day workflow guidance",
        difficulty="hard",
        sender_tier="premier",
        account_flags=[],
        refund_amount=None,
        age_hours=12,
        thread_bodies=[
            "We are a premier account and need same-day guidance on workflow approvals.",
            "Following up to ensure this is being prioritized today.",
        ],
        expected_category="customer_success",
        expected_priority="high",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["today", "priority", "guide"],
        history_keywords=["same-day", "following up"],
        clarification_keywords=[],
        policy_version="v2",
        max_steps=8,
    ),
]


EXTENDED_EASY_TEMPLATES: list[ScenarioTemplate] = [
    ScenarioTemplate(
        scenario_id="easy_cross_domain_credit_request",
        title="Service outage with billing-credit request",
        subject="Outage credit for yesterday's dashboard downtime",
        difficulty="easy",
        sender_tier="vip",
        account_flags=[],
        refund_amount=240.0,
        age_hours=20,
        thread_bodies=["Our dashboard was down for hours yesterday and we need a billing credit for the disruption."],
        expected_category="billing",
        expected_priority="high",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=[],
        history_keywords=[],
        clarification_keywords=[],
        refund_range=(180.0, 320.0),
        age_range=(10.0, 28.0),
        visible_problem_type="billing",
        max_steps=5,
    ),
    ScenarioTemplate(
        scenario_id="easy_standard_return_label",
        title="Straightforward return label request",
        subject="Need a return label for the wrong headset size",
        difficulty="easy",
        sender_tier="standard",
        account_flags=[],
        refund_amount=None,
        age_hours=15,
        thread_bodies=["The headset size was wrong and I just need a return label so I can send it back."],
        expected_category="returns",
        expected_priority="medium",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=[],
        history_keywords=[],
        clarification_keywords=[],
        age_range=(8.0, 24.0),
        visible_problem_type="returns",
        max_steps=5,
    ),
    ScenarioTemplate(
        scenario_id="easy_premier_same_day_question",
        title="Premier same-day onboarding guidance",
        subject="Need same-day approval workflow guidance",
        difficulty="easy",
        sender_tier="premier",
        account_flags=[],
        refund_amount=None,
        age_hours=8,
        thread_bodies=["We are a premier customer and need someone to guide us through approval workflows today."],
        expected_category="customer_success",
        expected_priority="high",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=[],
        history_keywords=[],
        clarification_keywords=[],
        policy_version="v2",
        age_range=(4.0, 18.0),
        visible_problem_type="customer_success",
        max_steps=5,
    ),
    ScenarioTemplate(
        scenario_id="easy_spam_with_legal_bait",
        title="Spam message posing as a legal notice",
        subject="Urgent legal review required click for settlement",
        difficulty="easy",
        sender_tier="standard",
        account_flags=[],
        refund_amount=None,
        age_hours=2,
        thread_bodies=[
            "Click this external form to avoid legal trouble and unlock a guaranteed settlement for your account."
        ],
        expected_category="spam",
        expected_priority="low",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=[],
        history_keywords=[],
        clarification_keywords=[],
        visible_problem_type="spam",
        max_steps=5,
    ),
]


EXTENDED_MEDIUM_TEMPLATES: list[ScenarioTemplate] = [
    ScenarioTemplate(
        scenario_id="medium_vip_downgrade_dispute",
        title="Customer disputes account tier downgrade",
        subject="Why was our VIP support tier removed?",
        difficulty="medium",
        sender_tier="standard",
        account_flags=["vip_dispute"],
        refund_amount=None,
        age_hours=18,
        thread_bodies=["We were told we are no longer VIP, but our contract says we should be. Please fix this."],
        expected_category="customer_success",
        expected_priority="medium",
        expected_escalation=False,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["support tier", "contract", "review"],
        history_keywords=["vip", "contract"],
        clarification_keywords=["contract", "renewal", "account"],
        clarification_body=(
            "Our renewal invoice should include premium support and the downgrade happened after "
            "that change."
        ),
        age_range=(12.0, 30.0),
        max_steps=7,
    ),
    ScenarioTemplate(
        scenario_id="medium_cross_domain_billing_bug",
        title="App failure tied to a disputed charge",
        subject="App stopped working right after the add-on charge",
        difficulty="medium",
        sender_tier="vip",
        account_flags=[],
        refund_amount=260.0,
        age_hours=27,
        thread_bodies=["The app broke right after we were charged for the add-on and I need this sorted out."],
        expected_category="technical_support",
        expected_priority="high",
        expected_escalation=False,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["app", "billing", "investigate"],
        history_keywords=["charge", "broke"],
        clarification_keywords=["error", "screen", "charge"],
        clarification_body=(
            "The charge was expected; the real issue is that the desktop app crashes after the "
            "add-on unlocks."
        ),
        refund_range=(180.0, 320.0),
        age_range=(18.0, 36.0),
        max_steps=7,
    ),
    ScenarioTemplate(
        scenario_id="medium_refund_or_credit_review",
        title="Credit note versus refund ambiguity",
        subject="Need the right fix for the duplicate invoice",
        difficulty="medium",
        sender_tier="standard",
        account_flags=[],
        refund_amount=480.0,
        age_hours=34,
        thread_bodies=["We were billed twice and need the right correction applied to the account."],
        expected_category="billing",
        expected_priority="medium",
        expected_escalation=False,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["invoice", "credit", "refund"],
        history_keywords=["twice", "correction"],
        clarification_keywords=["credit", "refund", "invoice"],
        clarification_body="A refund is fine, but a credit memo on the current invoice would also solve it for us.",
        refund_range=(320.0, 520.0),
        age_range=(24.0, 44.0),
        max_steps=7,
    ),
    ScenarioTemplate(
        scenario_id="medium_migration_or_refund",
        title="Cross-domain migration delay with refund threat",
        subject="Migration failed and we may need our money back",
        difficulty="medium",
        sender_tier="premier",
        account_flags=[],
        refund_amount=540.0,
        age_hours=21,
        thread_bodies=["Our migration failed and I need to know whether you can fix it or refund the service."],
        expected_category="technical_support",
        expected_priority="high",
        expected_escalation=False,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["migration", "fix", "timeline"],
        history_keywords=["refund", "failed"],
        clarification_keywords=["error", "migration", "refund"],
        clarification_body=(
            "We still want the migration completed today; if that is impossible we would discuss "
            "a refund later."
        ),
        refund_range=(500.0, 760.0),
        age_range=(16.0, 30.0),
        policy_version="random",
        max_steps=7,
    ),
    ScenarioTemplate(
        scenario_id="medium_false_legal_subject",
        title="Alarmist subject with ordinary billing body",
        subject="Legal escalation if invoice is wrong",
        difficulty="medium",
        sender_tier="standard",
        account_flags=[],
        refund_amount=None,
        age_hours=16,
        thread_bodies=["Ignore my dramatic subject line, I only need help understanding a tax line on the invoice."],
        expected_category="billing",
        expected_priority="medium",
        expected_escalation=False,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["invoice", "tax", "explain"],
        history_keywords=["subject line"],
        clarification_keywords=["invoice", "line item", "tax"],
        clarification_body="It is the tax adjustment on invoice INV-441 that I need explained.",
        max_steps=7,
    ),
    ScenarioTemplate(
        scenario_id="medium_account_lock_or_chargeback",
        title="Account lock could be fraud or billing dispute",
        subject="Locked out after bank reversal notice",
        difficulty="medium",
        sender_tier="standard",
        account_flags=["chargeback_risk"],
        refund_amount=None,
        age_hours=25,
        thread_bodies=["Our account locked right after a bank reversal notice and I do not know what happened."],
        expected_category="billing",
        expected_priority="medium",
        expected_escalation=False,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["review", "account", "chargeback"],
        history_keywords=["bank reversal", "locked"],
        clarification_keywords=["chargeback", "card", "bank"],
        clarification_body="The bank reversed a subscription payment and now the billing console is locked.",
        expected_flag_fraud=True,
        fraud_keywords=["chargeback", "bank reversal"],
        policy_version="v2",
        partial_observability=True,
        hidden_account_flag_count=1,
        max_steps=7,
    ),
    ScenarioTemplate(
        scenario_id="medium_snooze_sla_trap",
        title="Customer asks for delay near SLA boundary",
        subject="Can this wait until next week?",
        difficulty="medium",
        sender_tier="standard",
        account_flags=[],
        refund_amount=None,
        age_hours=68,
        thread_bodies=[
            "I know this invoice problem is open, but can you just wait until next week before "
            "handling it?"
        ],
        expected_category="billing",
        expected_priority="medium",
        expected_escalation=False,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["timeline", "invoice", "today"],
        history_keywords=["wait", "next week"],
        clarification_keywords=["invoice", "deadline", "timeline"],
        clarification_body=(
            "I can send the invoice number today, but I still need to understand the charge "
            "before next week."
        ),
        age_range=(66.0, 71.0),
        max_steps=7,
    ),
    ScenarioTemplate(
        scenario_id="medium_contract_or_outage_credit",
        title="Contract dispute mixed with service credit ask",
        subject="Need help with the contract and outage credit",
        difficulty="medium",
        sender_tier="vip",
        account_flags=[],
        refund_amount=300.0,
        age_hours=29,
        thread_bodies=["We need help with our contract and the outage credit from yesterday."],
        expected_category="customer_success",
        expected_priority="high",
        expected_escalation=False,
        requires_request_info=True,
        request_info_first_required=True,
        response_keywords=["contract", "credit", "review"],
        history_keywords=["outage", "contract"],
        clarification_keywords=["contract", "credit", "plan"],
        clarification_body=(
            "The real blocker is that our plan terms changed unexpectedly, and the outage credit "
            "is secondary."
        ),
        refund_range=(220.0, 420.0),
        max_steps=7,
    ),
]


EXTENDED_HARD_TEMPLATES: list[ScenarioTemplate] = [
    ScenarioTemplate(
        scenario_id="hard_concurrent_violations",
        title="Legal threat, high refund, and fraud signals in one ticket",
        subject="Unauthorized duplicate charge and legal escalation",
        difficulty="hard",
        sender_tier="premier",
        account_flags=["fraud_risk"],
        refund_amount=920.0,
        age_hours=86,
        thread_bodies=[
            "We found an unauthorized duplicate charge for $920 on a newly added card.",
            "If this is not escalated today our legal counsel will get involved.",
            "Please open a fraud review and confirm the billing escalation immediately.",
        ],
        expected_category="billing",
        expected_priority="urgent",
        expected_escalation=True,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["fraud", "escalated", "duplicate charge", "review"],
        history_keywords=["unauthorized", "legal counsel", "today"],
        clarification_keywords=[],
        conflict_keywords=["fraud", "billing", "legal"],
        required_action_order=["flag_fraud", "escalate", "draft_response"],
        expected_flag_fraud=True,
        fraud_keywords=["unauthorized", "newly added card", "fraud review"],
        expected_escalation_reason="Policy escalation required by legal pressure and high-value billing risk.",
        refund_range=(860.0, 1100.0),
        age_range=(80.0, 104.0),
        policy_version="v2",
        partial_observability=True,
        hidden_account_flag_count=1,
        max_steps=8,
    ),
    ScenarioTemplate(
        scenario_id="hard_snooze_sla_race",
        title="Customer-requested delay crosses urgent SLA boundary",
        subject="Can we hold this invoice case for six more hours?",
        difficulty="hard",
        sender_tier="standard",
        account_flags=[],
        refund_amount=None,
        age_hours=68,
        thread_bodies=[
            "This invoice dispute has been open for a while, but can you snooze it for six "
            "hours until our finance lead returns?",
            "I still need a real answer as soon as possible after that.",
        ],
        expected_category="billing",
        expected_priority="medium",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["invoice", "review", "today"],
        history_keywords=["finance lead", "snooze", "open for a while"],
        clarification_keywords=[],
        age_range=(67.0, 71.0),
        max_steps=8,
    ),
    ScenarioTemplate(
        scenario_id="hard_policy_shift_fraud_upgrade",
        title="Policy version upgrades mid-episode and adds fraud requirement",
        subject="Premier account with suspicious card activity",
        difficulty="hard",
        sender_tier="premier",
        account_flags=["fraud_risk"],
        refund_amount=320.0,
        age_hours=18,
        thread_bodies=[
            "We noticed suspicious card activity after a failed top-up and need help restoring access.",
            "This is for our premier workspace and we need next steps today.",
        ],
        expected_category="billing",
        expected_priority="high",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["fraud", "review", "today", "account"],
        history_keywords=["suspicious card", "premier workspace"],
        clarification_keywords=[],
        conflict_keywords=["fraud", "premier", "today"],
        required_action_order=["flag_fraud", "draft_response"],
        expected_flag_fraud=True,
        fraud_keywords=["suspicious card", "failed top-up"],
        policy_version="v1",
        policy_transition_step=3,
        policy_transition_to="v2",
        partial_observability=True,
        hidden_account_flag_count=1,
        max_steps=8,
    ),
    ScenarioTemplate(
        scenario_id="hard_contradictory_claim",
        title="Customer claim contradicts account records",
        subject="Refund the cancelled order you still shipped",
        difficulty="hard",
        sender_tier="standard",
        account_flags=["no_cancellation_on_record"],
        refund_amount=300.0,
        age_hours=40,
        thread_bodies=[
            "I was charged $300 for an order I cancelled and I want the refund now.",
            "Your records are wrong because the item should never have shipped.",
        ],
        expected_category="billing",
        expected_priority="medium",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["review", "order", "investigate"],
        history_keywords=["cancelled", "shipped"],
        clarification_keywords=[],
        conflict_keywords=["cancelled", "shipped", "records"],
        required_action_order=["flag_fraud", "draft_response"],
        expected_flag_fraud=True,
        fraud_keywords=["cancelled", "records are wrong", "shipped"],
        policy_version="v2",
        partial_observability=True,
        hidden_account_flag_count=1,
        max_steps=8,
    ),
    ScenarioTemplate(
        scenario_id="hard_cross_domain_legal_outage",
        title="Technical outage becomes legal and billing risk",
        subject="Outage losses and invoice dispute need urgent review",
        difficulty="hard",
        sender_tier="vip",
        account_flags=[],
        refund_amount=610.0,
        age_hours=76,
        thread_bodies=[
            "Our production workflow was down and the invoice still charged us in full.",
            "We already followed up twice and need the outage credit corrected immediately.",
            "If this keeps dragging on our legal team will review the contract impact.",
        ],
        expected_category="legal",
        expected_priority="urgent",
        expected_escalation=True,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["outage", "invoice", "escalated", "review"],
        history_keywords=["followed up twice", "legal team", "contract"],
        clarification_keywords=[],
        legal_language=True,
        expected_escalation_reason="Customer mentioned legal action.",
        refund_range=(560.0, 760.0),
        age_range=(72.0, 96.0),
        max_steps=8,
    ),
    ScenarioTemplate(
        scenario_id="hard_hidden_flag_dispute",
        title="Fraud dispute with partially hidden account risk flags",
        subject="Why was our account frozen after these card updates?",
        difficulty="hard",
        sender_tier="standard",
        account_flags=["fraud_risk", "chargeback_risk"],
        refund_amount=None,
        age_hours=52,
        thread_bodies=[
            "Our account froze after we updated payment cards and saw several reversals in the dashboard.",
            "We need to know whether this is fraud review or a billing issue.",
        ],
        expected_category="billing",
        expected_priority="medium",
        expected_escalation=False,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["review", "fraud", "billing"],
        history_keywords=["payment cards", "reversals"],
        clarification_keywords=[],
        conflict_keywords=["fraud", "billing", "reversals"],
        required_action_order=["flag_fraud", "draft_response"],
        expected_flag_fraud=True,
        fraud_keywords=["reversals", "payment cards"],
        policy_version="v2",
        partial_observability=True,
        hidden_account_flag_count=1,
        max_steps=8,
    ),
    ScenarioTemplate(
        scenario_id="hard_premier_migration_with_refund_pressure",
        title="Premier migration issue with refund pressure and same-day rule",
        subject="Migration broke and we may need a same-day refund decision",
        difficulty="hard",
        sender_tier="premier",
        account_flags=[],
        refund_amount=740.0,
        age_hours=28,
        thread_bodies=[
            "Our enterprise migration failed halfway through and we already lost a full day.",
            "If you cannot recover it today we need a refund decision for the $740 service block.",
            "Please confirm whether this is being escalated right now.",
        ],
        expected_category="technical_support",
        expected_priority="high",
        expected_escalation=True,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["migration", "today", "escalated", "refund"],
        history_keywords=["lost a full day", "right now"],
        clarification_keywords=[],
        conflict_keywords=["migration", "refund", "today"],
        required_action_order=["escalate", "draft_response"],
        expected_escalation_reason="Refund exceeds $500.",
        policy_version="v2",
        refund_range=(680.0, 860.0),
        age_range=(20.0, 40.0),
        max_steps=8,
    ),
    ScenarioTemplate(
        scenario_id="hard_billing_technical_contract_precedence",
        title="Billing, technical, and contract signals with precedence rules",
        subject="Contract breach claim after admin lockout and duplicate charges",
        difficulty="hard",
        sender_tier="vip",
        account_flags=["suspended"],
        refund_amount=830.0,
        age_hours=89,
        thread_bodies=[
            "Our admins were locked out after duplicate renewal charges posted to the account.",
            "The contract guarantees access and this suspension is causing material damage.",
            "We need billing, access restoration, and an urgent escalation path today.",
        ],
        expected_category="billing",
        expected_priority="urgent",
        expected_escalation=True,
        requires_request_info=False,
        request_info_first_required=False,
        response_keywords=["billing", "restore", "escalated", "contract"],
        history_keywords=["locked out", "duplicate renewal", "today"],
        clarification_keywords=[],
        conflict_keywords=["billing", "contract", "access"],
        required_action_order=["escalate", "draft_response"],
        legal_language=True,
        suspended_account=True,
        expected_escalation_reason="Policy escalation required by legal and high-value billing risk.",
        refund_range=(760.0, 980.0),
        age_range=(82.0, 102.0),
        max_steps=8,
    ),
]


ALL_TEMPLATES: tuple[ScenarioTemplate, ...] = tuple(
    EASY_TEMPLATES
    + EXTENDED_EASY_TEMPLATES
    + MEDIUM_TEMPLATES
    + EXTENDED_MEDIUM_TEMPLATES
    + HARD_TEMPLATES
    + EXTENDED_HARD_TEMPLATES
)
TEMPLATE_REGISTRY: dict[str, ScenarioTemplate] = {template.scenario_id: template for template in ALL_TEMPLATES}


def scenario_ids_for_task(task_name: Difficulty | None = None) -> list[str]:
    templates = ALL_TEMPLATES if task_name is None else tuple(
        template for template in ALL_TEMPLATES if template.difficulty == task_name
    )
    return [template.scenario_id for template in templates]


class ScenarioFactory:
    def __init__(self, seed: int = 20260328):
        self._seed = seed
        self._base_now = datetime(2026, 3, 28, 10, 0, 0)

    def build_all(self) -> list[TaskScenario]:
        return [self.build_canonical_scenario(template.scenario_id) for template in ALL_TEMPLATES]

    def build_canonical_scenario(self, scenario_id: str) -> TaskScenario:
        return self._build(TEMPLATE_REGISTRY[scenario_id], variant_key=None)

    def build_variant_scenario(self, scenario_id: str, variant_key: str | int) -> TaskScenario:
        return self._build(TEMPLATE_REGISTRY[scenario_id], variant_key=str(variant_key))

    def _build(self, t: ScenarioTemplate, *, variant_key: str | None) -> TaskScenario:
        rng = self._rng_for_key(t.scenario_id if variant_key is None else f"{t.scenario_id}:{variant_key}")
        fake = Faker()
        fake.seed_instance(
            self._stable_seed(
                t.scenario_id if variant_key is None else f"{t.scenario_id}:{variant_key}:faker"
            )
        )

        refund_amount = t.refund_amount if variant_key is None else self._variant_refund_amount(t, rng)
        age_hours = t.age_hours if variant_key is None else self._variant_age_hours(t, rng)
        policy_version = self._resolved_policy_version(t, rng)
        subject = t.subject if variant_key is None else self._paraphrase_text(t.subject, rng)
        thread_bodies = (
            list(t.thread_bodies)
            if variant_key is None
            else self._variant_thread_bodies(t, rng, refund_amount)
        )
        clarification_body = (
            t.clarification_body
            if variant_key is None or t.clarification_body is None
            else self._variant_text(t.clarification_body, t, rng, refund_amount)
        )

        sender_name = fake.name()
        sender_email = fake.email()
        first_message_time = self._base_now - timedelta(hours=age_hours)
        directions = self._message_directions(t)
        thread = self._build_thread(
            t,
            subject,
            thread_bodies,
            sender_name,
            sender_email,
            first_message_time,
            directions,
            age_hours=age_hours,
            rng=None if variant_key is None else rng,
        )

        account_flags = list(t.account_flags)
        if t.suspended_account and "suspended" not in account_flags:
            account_flags.append("suspended")
        hidden_account_flags = self._hidden_account_flags(t, account_flags, rng)

        initial_refund_amount = None if t.requires_request_info else refund_amount
        scenario_token = t.scenario_id if variant_key is None else f"{t.scenario_id}:{variant_key}"
        deterministic_id = self._stable_seed(scenario_token) % 1000
        order_seed = self._stable_seed(f"{scenario_token}:order") % 10000
        ticket_id = f"T-{t.difficulty.upper()}-{deterministic_id:03d}"
        scenario_id = (
            t.scenario_id
            if variant_key is None
            else f"{t.scenario_id}__variant_{self._stable_seed(scenario_token) % 100000:05d}"
        )

        initial_snapshot = TicketSnapshot(
            ticket_id=ticket_id,
            thread=thread,
            sender_tier=t.sender_tier,
            account_flags=account_flags,
            refund_amount=initial_refund_amount,
            order_id=f"ORD-{order_seed:04d}",
            visible_problem_type=t.visible_problem_type,
        )

        clarification_snapshot = None
        if clarification_body:
            clarification_time = thread[-1].timestamp + timedelta(hours=2)
            clarification_snapshot = TicketSnapshot(
                ticket_id=ticket_id,
                thread=thread
                + [
                    self._message(
                        message_id=f"{t.scenario_id}_clarification",
                        timestamp=clarification_time,
                        subject=f"Re: {subject}",
                        body=clarification_body,
                        sender_name=sender_name,
                        sender_email=sender_email,
                        direction="customer",
                    )
                ],
                sender_tier=t.sender_tier,
                account_flags=account_flags,
                refund_amount=refund_amount,
                order_id=f"ORD-{order_seed:04d}",
                visible_problem_type=t.expected_category,
            )

        completion_actions: list[ActionType] = []
        if t.requires_request_info:
            completion_actions.append("request_info")
        if t.expected_flag_fraud:
            completion_actions.append("flag_fraud")
        completion_actions.extend(["categorize", "set_priority"])
        if t.expected_escalation:
            completion_actions.append("escalate")
        if t.difficulty in {"medium", "hard"}:
            completion_actions.append("draft_response")

        default_objective = {
            "easy": "Classify the ticket, set priority, and follow active business policy rules.",
            "medium": "Recognize ambiguity, ask clarifying questions first, then resolve policy-safely.",
            "hard": (
                "Resolve a realistic multi-turn thread with policy precedence, time pressure, and "
                "history-aware communication."
            ),
        }[t.difficulty]

        return TaskScenario(
            scenario_id=scenario_id,
            difficulty=t.difficulty,
            title=t.title,
            objective=t.objective or default_objective,
            max_steps=t.max_steps,
            now=self._base_now,
            policy_version=policy_version,
            policy_transition_step=t.policy_transition_step,
            policy_transition_to=t.policy_transition_to,
            initial_snapshot=initial_snapshot,
            clarification_snapshot=clarification_snapshot,
            hidden_account_flags=hidden_account_flags,
            ground_truth=GroundTruth(
                expected_category=t.expected_category,
                expected_priority=t.expected_priority,
                expected_escalation=t.expected_escalation,
                expected_escalation_reason=t.expected_escalation_reason,
                expected_flag_fraud=t.expected_flag_fraud,
                fraud_keywords=t.fraud_keywords,
                requires_request_info=t.requires_request_info,
                request_info_first_required=t.request_info_first_required,
                clarification_keywords=t.clarification_keywords,
                response_keywords=t.response_keywords,
                history_keywords=t.history_keywords,
                conflict_keywords=t.conflict_keywords,
                required_action_order=t.required_action_order,
                completion_action_types=completion_actions,
                ambiguous=t.requires_request_info,
            ),
        )

    def _stable_seed(self, key: str) -> int:
        digest = hashlib.sha256(f"{self._seed}:{key}".encode()).digest()
        return int.from_bytes(digest[:8], "big") % (2**31)

    def _rng_for_key(self, key: str) -> random.Random:
        return random.Random(self._stable_seed(key))

    def _variant_refund_amount(self, template: ScenarioTemplate, rng: random.Random) -> float | None:
        if template.refund_amount is None:
            return None
        if template.refund_range is not None:
            low, high = template.refund_range
            return round(rng.uniform(low, high), 2)
        amount = float(template.refund_amount)
        if amount > 500:
            varied = amount * rng.uniform(0.9, 1.12)
            return round(max(525.0, varied), 2)
        if amount >= 100:
            varied = amount * rng.uniform(0.88, 1.12)
            return round(min(495.0, max(50.0, varied)), 2)
        varied = amount * rng.uniform(0.82, 1.18)
        return round(min(495.0, max(15.0, varied)), 2)

    def _variant_age_hours(self, template: ScenarioTemplate, rng: random.Random) -> float:
        if template.age_range is not None:
            low, high = template.age_range
            return round(rng.uniform(low, high), 2)
        age = float(template.age_hours)
        if age < 24:
            return round(min(23.5, max(1.0, age + rng.uniform(-4.0, 4.0))), 2)
        if age < 72:
            return round(min(71.5, max(24.5, age + rng.uniform(-6.0, 6.0))), 2)
        return round(max(72.5, age + rng.uniform(-12.0, 12.0)), 2)

    def _resolved_policy_version(
        self,
        template: ScenarioTemplate,
        rng: random.Random,
    ) -> PolicyVersion:
        if template.policy_version == "random":
            return rng.choice(["v1", "v2"])
        return template.policy_version

    def _hidden_account_flags(
        self,
        template: ScenarioTemplate,
        account_flags: list[str],
        rng: random.Random,
    ) -> list[str]:
        if not template.partial_observability or not account_flags:
            return []
        hide_count = template.hidden_account_flag_count or 1
        hide_count = min(hide_count, len(account_flags))
        if hide_count == len(account_flags):
            hide_count = max(0, len(account_flags) - 1)
        if hide_count <= 0:
            return []
        return sorted(rng.sample(account_flags, hide_count))

    def _variant_thread_bodies(
        self,
        template: ScenarioTemplate,
        rng: random.Random,
        refund_amount: float | None,
    ) -> list[str]:
        return [self._variant_text(body, template, rng, refund_amount) for body in template.thread_bodies]

    def _variant_text(
        self,
        text: str,
        template: ScenarioTemplate,
        rng: random.Random,
        refund_amount: float | None,
    ) -> str:
        updated = self._paraphrase_text(text, rng)
        if template.refund_amount is not None and refund_amount is not None:
            updated = self._replace_amount_mentions(updated, template.refund_amount, refund_amount)
        return updated

    def _replace_amount_mentions(self, text: str, original_amount: float, new_amount: float) -> str:
        rounded_original = int(round(original_amount))
        rounded_new = int(round(new_amount))
        replacements = {
            f"${rounded_original}": f"${rounded_new}",
            str(rounded_original): str(rounded_new),
            f"${original_amount:.2f}": f"${new_amount:.2f}",
        }
        updated = text
        for source, target in replacements.items():
            updated = updated.replace(source, target)
        return updated

    def _paraphrase_text(self, text: str, rng: random.Random) -> str:
        replacements = {
            "need help": ["need help", "need assistance", "need support"],
            "help": ["help", "assistance", "support"],
            "urgent": ["urgent", "time-sensitive", "pressing"],
            "refund": ["refund", "reimbursement"],
            "duplicate": ["duplicate", "extra", "double"],
            "charge": ["charge", "billing item", "billing entry"],
            "issue": ["issue", "problem", "situation"],
            "reviewing": ["reviewing", "looking into", "checking"],
            "update": ["update", "follow-up", "status update"],
            "still": ["still", "currently", "at the moment"],
            "account": ["account", "workspace account", "profile"],
            "today": ["today", "this business day", "as soon as possible"],
            "app": ["app", "application"],
        }

        updated = text
        for source, options in replacements.items():
            pattern = re.compile(re.escape(source), re.IGNORECASE)
            if not pattern.search(updated):
                continue
            replacement = rng.choice(options)
            updated = pattern.sub(lambda _: replacement, updated, count=1)
        return updated

    def _message_directions(self, t: ScenarioTemplate) -> list[Literal["customer", "agent", "system"]]:
        if not t.thread_directions:
            return ["customer"] * len(t.thread_bodies)
        if len(t.thread_directions) >= len(t.thread_bodies):
            return t.thread_directions[: len(t.thread_bodies)]
        return t.thread_directions + ["customer"] * (len(t.thread_bodies) - len(t.thread_directions))

    def _build_thread(
        self,
        t: ScenarioTemplate,
        subject: str,
        thread_bodies: list[str],
        sender_name: str,
        sender_email: str,
        first_message_time: datetime,
        directions: list[Literal["customer", "agent", "system"]],
        *,
        age_hours: float,
        rng: random.Random | None,
    ) -> list[EmailMessage]:
        if len(thread_bodies) == 1:
            timestamps = [first_message_time]
        else:
            spacing = max(1.0, age_hours / len(thread_bodies))
            if rng is not None:
                spacing = max(1.0, spacing * rng.uniform(0.92, 1.08))
            timestamps = [
                first_message_time + timedelta(hours=index * spacing) for index in range(len(thread_bodies))
            ]

        thread: list[EmailMessage] = []
        for index, body in enumerate(thread_bodies, start=1):
            message_body = body
            if t.legal_language and index == len(t.thread_bodies):
                lowered = message_body.lower()
                if "legal action" not in lowered and "lawyer" not in lowered and "lawsuit" not in lowered:
                    message_body = f"{message_body} We are considering legal action if this is not resolved promptly."
            thread.append(
                self._message(
                    message_id=f"{t.scenario_id}_m{index}",
                    timestamp=timestamps[index - 1],
                    subject=subject,
                    body=message_body,
                    sender_name=sender_name,
                    sender_email=sender_email,
                    direction=directions[index - 1],
                )
            )
        return thread

    def _message(
        self,
        message_id: str,
        timestamp: datetime,
        subject: str,
        body: str,
        sender_name: str,
        sender_email: str,
        direction: Literal["customer", "agent", "system"],
    ) -> EmailMessage:
        return EmailMessage(
            message_id=message_id,
            direction=direction,
            sender_name=sender_name,
            sender_email=sender_email,
            timestamp=timestamp,
            subject=subject,
            body=body,
        )


def build_scenarios() -> list[TaskScenario]:
    return ScenarioFactory().build_all()
