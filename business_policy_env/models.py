from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

Category = Literal["billing", "technical_support", "returns", "legal", "customer_success", "spam"]
Priority = Literal["low", "medium", "high", "urgent"]
ActionType = Literal["categorize", "set_priority", "draft_response", "escalate", "mark_spam", "request_info"]
Difficulty = Literal["easy", "medium", "hard"]
SenderTier = Literal["standard", "vip"]


class EmailMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str
    direction: Literal["customer", "agent", "system"] = "customer"
    sender_name: str
    sender_email: str
    timestamp: datetime
    subject: str
    body: str


class TicketSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    thread: list[EmailMessage]
    sender_tier: SenderTier
    account_flags: list[str] = Field(default_factory=list)
    refund_amount: Optional[float] = None
    order_id: Optional[str] = None
    product_name: Optional[str] = None
    visible_problem_type: Optional[str] = None


class GroundTruth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_category: Optional[Category] = None
    expected_priority: Optional[Priority] = None
    expected_escalation: bool = False
    expected_escalation_reason: Optional[str] = None
    requires_request_info: bool = False
    request_info_first_required: bool = False
    clarification_keywords: list[str] = Field(default_factory=list)
    response_keywords: list[str] = Field(default_factory=list)
    history_keywords: list[str] = Field(default_factory=list)
    completion_action_types: list[ActionType] = Field(default_factory=list)
    ambiguous: bool = False


class TaskScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    difficulty: Difficulty
    title: str
    objective: str
    max_steps: int
    now: datetime
    initial_snapshot: TicketSnapshot
    clarification_snapshot: Optional[TicketSnapshot] = None
    ground_truth: GroundTruth


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: ActionType
    reasoning: str = Field(min_length=3)
    category: Optional[Category] = None
    priority: Optional[Priority] = None
    response_text: Optional[str] = None
    escalation_reason: Optional[str] = None
    clarifying_question: Optional[str] = None

    @model_validator(mode="after")
    def validate_action_payload(self) -> "Action":
        required_fields = {
            "categorize": self.category,
            "set_priority": self.priority,
            "draft_response": self.response_text,
            "escalate": self.escalation_reason,
            "request_info": self.clarifying_question,
        }
        value = required_fields.get(self.action_type)
        if self.action_type in required_fields and not value:
            field_name = {
                "categorize": "category",
                "set_priority": "priority",
                "draft_response": "response_text",
                "escalate": "escalation_reason",
                "request_info": "clarifying_question",
            }[self.action_type]
            raise ValueError(f"{field_name} is required for action_type={self.action_type}")
        return self


class ActionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_index: int
    action: Action
    timestamp: datetime
    valid: bool = True


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    difficulty: Difficulty
    current_email: EmailMessage
    thread: list[EmailMessage]
    sender_tier: SenderTier
    account_flags: list[str]
    refund_amount: Optional[float] = None
    issue_age_hours: float
    emails_remaining: int
    steps_taken: int
    max_steps: int
    action_history: list[ActionRecord]
    policy_rules: list[str]
    task_objective: str
    clarification_received: bool = False


class StepInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid_action: bool
    final_score: Optional[float] = None
    partial_score: Optional[float] = None
    policy_violations: list[str] = Field(default_factory=list)
    reward_breakdown: dict[str, float] = Field(default_factory=dict)
    explanation: str


class StepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any]


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_name: Optional[Difficulty] = None
    scenario_id: Optional[str] = None


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Action


class RewardBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reward: float
    components: dict[str, float]
    explanation: str
