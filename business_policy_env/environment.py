from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, cast

from pydantic import ValidationError

from .data_generation import ScenarioFactory, scenario_ids_for_task
from .models import (
    Action,
    ActionRecord,
    Difficulty,
    EpisodePhase,
    Observation,
    PolicyVersion,
    SpecialistTeam,
    TaskName,
    TaskScenario,
    TicketSnapshot,
)
from .policies import check_policy_violations, compute_policy_expectations, policy_rules_for
from .rewards import current_progress, invalid_action_breakdown, shaped_reward
from .tasks import (
    build_ground_truth_payload,
    compute_issue_age_hours,
    is_substantive_question,
    request_info_quality,
)


class BusinessPolicyComplianceEnv:
    def __init__(self, seed: int = 20260328) -> None:
        self._seed = seed
        self._scenario_factory = ScenarioFactory(seed=seed)
        task_names: tuple[Difficulty, ...] = ("easy", "medium", "hard")
        self._task_family_ids = {task: scenario_ids_for_task(task) for task in task_names}
        self._shuffle_bags: dict[str, list[str]] = defaultdict(list)
        self._selection_rngs = {
            task: random.Random(self._stable_seed(f"shuffle:{task}")) for task in self._task_family_ids
        }
        self._variant_counters: dict[str, int] = defaultdict(int)
        self._recent_final_scores: deque[float] = deque(maxlen=6)
        self._connection = self._create_connection()
        self.current_scenario: TaskScenario | None = None
        self.action_history: list[ActionRecord] = []
        self.clarification_received = False
        self.episode_phase = EpisodePhase.initial
        self._simulated_offset_hours = 0.0
        self._snooze_sla_violations = 0
        self._active_policy_version: PolicyVersion = "v1"
        self._difficulty_mode: str = "task"
        self._specialist_consults_used = 0
        self._specialist_consult_budget_remaining = 0
        self._specialist_notes: list[str] = []
        self._last_final_score: float | None = None
        self.done = False

    def _stable_seed(self, key: str) -> int:
        digest = hashlib.sha256(f"{self._seed}:{key}".encode()).digest()
        return int.from_bytes(digest[:8], "big") % (2**31)

    def _create_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS episode_actions (
                step_index INTEGER NOT NULL,
                action_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                valid INTEGER NOT NULL
            )
            """
        )
        connection.commit()
        return connection

    def close(self) -> None:
        try:
            self._connection.close()
        except sqlite3.Error:
            return

    def _reset_connection(self) -> None:
        self.close()
        self._connection = self._create_connection()

    def available_tasks(self) -> dict[str, list[str]]:
        tasks: dict[str, list[str]] = {task: list(ids) for task, ids in self._task_family_ids.items()}
        tasks["adaptive"] = []
        return tasks

    def _initial_specialist_budget(self, difficulty: Difficulty) -> int:
        if difficulty == "hard":
            return 2
        return 1

    def _select_adaptive_task(self) -> Difficulty:
        if not self._recent_final_scores:
            return "easy"
        window = list(self._recent_final_scores)[-3:]
        trailing_mean = sum(window) / len(window)
        if trailing_mean >= 0.78:
            return "hard"
        if trailing_mean >= 0.5:
            return "medium"
        return "easy"

    def _next_family_id(self, task_name: Difficulty) -> str:
        bag = self._shuffle_bags[task_name]
        if not bag:
            bag = list(self._task_family_ids[task_name])
            self._selection_rngs[task_name].shuffle(bag)
            self._shuffle_bags[task_name] = bag
        return self._shuffle_bags[task_name].pop()

    def _select_scenario(self, task_name: Difficulty | None, scenario_id: str | None) -> TaskScenario:
        if scenario_id:
            return self._scenario_factory.build_canonical_scenario(scenario_id)

        selected_task: Difficulty = "easy" if task_name is None else task_name
        family_id = self._next_family_id(selected_task)
        variant_index = self._variant_counters[selected_task]
        self._variant_counters[selected_task] += 1
        return self._scenario_factory.build_variant_scenario(family_id, variant_key=f"{selected_task}:{variant_index}")

    def _active_snapshot(self) -> TaskScenario:
        if self.current_scenario is None:
            raise RuntimeError("Environment has not been reset.")
        return self.current_scenario

    def _current_snapshot(self) -> TicketSnapshot:
        scenario = self._active_snapshot()
        if self.clarification_received and scenario.clarification_snapshot is not None:
            return scenario.clarification_snapshot
        return scenario.initial_snapshot

    def _grade_snapshot(self) -> TicketSnapshot:
        scenario = self._active_snapshot()
        return scenario.clarification_snapshot or scenario.initial_snapshot

    def _base_issue_age_hours(self) -> float:
        scenario = self._active_snapshot()
        return compute_issue_age_hours(self._current_snapshot(), scenario.now)

    def _issue_age_hours(self) -> float:
        return round(self._base_issue_age_hours() + self._simulated_offset_hours, 2)

    def _emails_remaining(self) -> int:
        scenario = self._active_snapshot()
        if self.clarification_received or scenario.clarification_snapshot is None:
            return 0
        return max(0, len(scenario.clarification_snapshot.thread) - len(scenario.initial_snapshot.thread))

    def _policy_version(self) -> PolicyVersion:
        return self._active_policy_version

    def _steps_until_policy_change(self) -> int | None:
        scenario = self._active_snapshot()
        if scenario.policy_transition_step is None or scenario.policy_transition_to is None:
            return None
        remaining = scenario.policy_transition_step - len(self.action_history)
        return remaining if remaining > 0 else None

    def _visible_account_flags(self, snapshot: TicketSnapshot) -> list[str]:
        scenario = self._active_snapshot()
        hidden = set(scenario.hidden_account_flags)
        return [flag for flag in snapshot.account_flags if flag not in hidden]

    def _build_specialist_note(
        self,
        specialist_team: SpecialistTeam,
        snapshot: TicketSnapshot,
        policy_version: PolicyVersion,
    ) -> str:
        expectations = compute_policy_expectations(snapshot, self._issue_age_hours(), policy_version)
        team_notes = {
            "billing_ops": "Billing ops recommends verifying invoice history and adjustment rules before closure.",
            "technical_ops": (
                "Technical ops recommends confirming the product surface, timeline, and any recent changes."
            ),
            "returns_ops": "Returns ops recommends confirming the fulfillment state and desired remedy before closure.",
            "legal_ops": "Legal ops recommends preserving the thread and escalating if legal language is present.",
            "customer_success_ops": (
                "Customer success recommends acknowledging the history and clarifying the desired outcome."
            ),
            "fraud_ops": "Fraud ops recommends checking account risk indicators before any resolution step.",
        }
        signals: list[str] = []
        if expectations["requires_fraud_flag"]:
            signals.append("Fraud review is likely relevant.")
        if expectations["requires_escalation"]:
            signals.append("Escalation may be required before closure.")
        if snapshot.visible_problem_type:
            signals.append(f"Primary surface appears to be {snapshot.visible_problem_type.replace('_', ' ')}.")
        return " ".join([team_notes[specialist_team], *signals]).strip()

    def _maybe_transition_policy_version(self) -> None:
        scenario = self._active_snapshot()
        if scenario.policy_transition_step is None or scenario.policy_transition_to is None:
            return
        if len(self.action_history) >= scenario.policy_transition_step:
            self._active_policy_version = scenario.policy_transition_to

    def _step_timestamp(self, step_index: int) -> datetime:
        scenario = self._active_snapshot()
        return scenario.now + timedelta(seconds=step_index)

    def _log_action(self, record: ActionRecord) -> None:
        self._connection.execute(
            "INSERT INTO episode_actions(step_index, action_type, payload, timestamp, valid) VALUES (?, ?, ?, ?, ?)",
            (
                record.step_index,
                record.action.action_type,
                json.dumps(record.action.model_dump(mode="json")),
                record.timestamp.isoformat(),
                int(record.valid),
            ),
        )
        self._connection.commit()

    def _episode_log(self) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT step_index, action_type, payload, timestamp, valid FROM episode_actions ORDER BY step_index"
        ).fetchall()
        return [dict(row) for row in rows]

    def _observation(self) -> Observation:
        scenario = self._active_snapshot()
        snapshot = self._current_snapshot()
        visible_flags = self._visible_account_flags(snapshot)
        return Observation(
            scenario_id=scenario.scenario_id,
            difficulty=scenario.difficulty,
            current_email=snapshot.thread[-1],
            thread=snapshot.thread,
            sender_tier=snapshot.sender_tier,
            account_flags=visible_flags,
            hidden_flags=len(scenario.hidden_account_flags),
            refund_amount=snapshot.refund_amount,
            issue_age_hours=self._issue_age_hours(),
            emails_remaining=self._emails_remaining(),
            steps_taken=len(self.action_history),
            max_steps=scenario.max_steps,
            action_history=self.action_history,
            policy_rules=policy_rules_for(self._policy_version()),
            policy_version=self._policy_version(),
            policy_transition_step=scenario.policy_transition_step,
            policy_transition_to=scenario.policy_transition_to,
            steps_until_policy_change=self._steps_until_policy_change(),
            task_objective=scenario.objective,
            clarification_received=self.clarification_received,
            episode_phase=self.episode_phase,
            difficulty_mode="adaptive" if self._difficulty_mode == "adaptive" else "task",
            specialist_consult_budget_remaining=self._specialist_consult_budget_remaining,
            specialist_consults_used=self._specialist_consults_used,
            specialist_notes=list(self._specialist_notes),
        )

    def _completion_reached(self) -> bool:
        scenario = self._active_snapshot()
        completed_types = {record.action.action_type for record in self.action_history}
        required_types = set(scenario.ground_truth.completion_action_types)
        return required_types.issubset(completed_types)

    def _advance_phase(self, action: Action) -> None:
        phase = self.episode_phase
        resolving_actions = {
            "categorize",
            "set_priority",
            "escalate",
            "flag_fraud",
            "draft_response",
            "mark_spam",
            "consult_specialist",
        }

        if phase == EpisodePhase.initial:
            if action.action_type == "request_info":
                self.episode_phase = (
                    EpisodePhase.post_clarification
                    if self.clarification_received
                    else EpisodePhase.awaiting_clarification
                )
            elif action.action_type in resolving_actions:
                self.episode_phase = EpisodePhase.resolving
        elif phase == EpisodePhase.awaiting_clarification:
            if self.clarification_received:
                self.episode_phase = EpisodePhase.post_clarification
        elif phase == EpisodePhase.post_clarification:
            if action.action_type in resolving_actions:
                self.episode_phase = EpisodePhase.resolving

        if self._completion_reached() or self.done:
            self.episode_phase = EpisodePhase.complete

    def _should_unlock_clarification(self, action: Action, scenario: TaskScenario) -> bool:
        if (
            action.action_type != "request_info"
            or scenario.clarification_snapshot is None
            or self.clarification_received
        ):
            return False
        if not is_substantive_question(action.clarifying_question):
            return False
        ground_truth = build_ground_truth_payload(
            scenario,
            scenario.clarification_snapshot,
            policy_version=self._policy_version(),
        )
        return request_info_quality(action, ground_truth) >= 0.5

    def reset(self, task_name: TaskName | None = None, scenario_id: str | None = None) -> Observation:
        selected_task: Difficulty | None = None
        if task_name in {"easy", "medium", "hard"}:
            selected_task = cast(Difficulty, task_name)
        self._difficulty_mode = "task"
        if scenario_id is None and task_name in {None, "adaptive"}:
            selected_task = self._select_adaptive_task()
            self._difficulty_mode = "adaptive"

        self.current_scenario = self._select_scenario(selected_task, scenario_id)
        self.action_history = []
        self.clarification_received = False
        self.episode_phase = EpisodePhase.initial
        self._simulated_offset_hours = 0.0
        self._snooze_sla_violations = 0
        self._active_policy_version = self.current_scenario.policy_version
        self._specialist_consults_used = 0
        self._specialist_consult_budget_remaining = self._initial_specialist_budget(self.current_scenario.difficulty)
        self._specialist_notes = []
        self._last_final_score = None
        self.done = False
        self._reset_connection()
        return self._observation()

    def step(self, action_input: Action | dict[str, Any]) -> tuple[Observation, float, bool, dict[str, Any]]:
        if self.current_scenario is None:
            self.reset()

        if self.done:
            schema_valid = True
            explanation = "Episode is already complete. Call reset() to start a new ticket."
            if not isinstance(action_input, Action):
                try:
                    Action.model_validate(action_input)
                except ValidationError as exc:
                    schema_valid = False
                    explanation = str(exc)
            observation = self._observation()
            info_done: dict[str, Any] = {
                "valid_action": schema_valid,
                "action_accepted": False,
                "episode_complete": True,
                "final_score": self._last_final_score,
                "partial_score": None,
                "policy_violations": [],
                "reward_breakdown": {"already_done": 0.0},
                "component_scores": {},
                "explanation": explanation,
            }
            return observation, 0.0, True, info_done

        try:
            action = action_input if isinstance(action_input, Action) else Action.model_validate(action_input)
        except ValidationError as exc:
            observation = self._observation()
            breakdown = invalid_action_breakdown(str(exc))
            info_invalid: dict[str, Any] = {
                "valid_action": False,
                "action_accepted": False,
                "episode_complete": False,
                "final_score": None,
                "partial_score": None,
                "policy_violations": [],
                "reward_breakdown": breakdown.components,
                "component_scores": {},
                "explanation": breakdown.explanation,
            }
            return observation, breakdown.reward, False, info_invalid

        scenario = self._active_snapshot()
        snapshot_before = self._current_snapshot()
        previous_age = self._issue_age_hours()
        prior_actions = [item.action for item in self.action_history]
        active_policy_version = self._policy_version()
        policy_violations = check_policy_violations(
            action,
            snapshot_before,
            previous_age,
            active_policy_version,
            prior_actions=prior_actions,
        )

        if action.action_type == "consult_specialist":
            if self._specialist_consult_budget_remaining <= 0:
                observation = self._observation()
                breakdown = invalid_action_breakdown("No specialist consult budget remaining.")
                info_consult_denied: dict[str, Any] = {
                    "valid_action": True,
                    "action_accepted": False,
                    "episode_complete": False,
                    "final_score": None,
                    "partial_score": None,
                    "policy_violations": [],
                    "reward_breakdown": breakdown.components,
                    "component_scores": {},
                    "explanation": breakdown.explanation,
                }
                return observation, breakdown.reward, False, info_consult_denied
            assert action.specialist_team is not None
            self._specialist_consult_budget_remaining -= 1
            self._specialist_consults_used += 1
            self._specialist_notes.append(
                self._build_specialist_note(action.specialist_team, snapshot_before, active_policy_version)
            )

        if action.action_type == "snooze" and action.snooze_hours:
            self._simulated_offset_hours += float(action.snooze_hours)
            new_age = self._issue_age_hours()
            if previous_age <= 72 < new_age:
                self._snooze_sla_violations += 1
            elif previous_age > 72:
                self._snooze_sla_violations += 1

        record = ActionRecord(
            step_index=len(self.action_history) + 1,
            action=action,
            timestamp=self._step_timestamp(len(self.action_history) + 1),
            valid=True,
        )
        self.action_history.append(record)
        self._log_action(record)

        if self._should_unlock_clarification(action, scenario):
            self.clarification_received = True

        if len(self.action_history) >= scenario.max_steps or self._completion_reached():
            self.done = True

        self._advance_phase(action)

        grading_payload = build_ground_truth_payload(
            scenario,
            self._grade_snapshot(),
            policy_version=active_policy_version,
        )
        actions = [item.action for item in self.action_history]
        reward_breakdown = shaped_reward(
            actions,
            grading_payload,
            self.done,
            scenario.max_steps,
            policy_violations,
            snooze_sla_violations=self._snooze_sla_violations,
            specialist_consults_used=self._specialist_consults_used,
            fraud_expected=scenario.ground_truth.expected_flag_fraud,
        )
        progress_score, components = current_progress(actions, grading_payload)
        if self.done:
            self._last_final_score = progress_score
            self._recent_final_scores.append(progress_score)
        else:
            self._maybe_transition_policy_version()
        observation = self._observation()
        info_step: dict[str, Any] = {
            "valid_action": True,
            "action_accepted": True,
            "episode_complete": self.done,
            "final_score": progress_score if self.done else None,
            "partial_score": None if self.done else progress_score,
            "policy_violations": policy_violations,
            "reward_breakdown": reward_breakdown.components,
            "component_scores": components,
            "explanation": reward_breakdown.explanation,
        }
        return observation, reward_breakdown.reward, self.done, info_step

    def state(self) -> dict[str, Any]:
        if self.current_scenario is None:
            return {"active": False, "detail": "Environment has not been reset."}

        scenario = self._active_snapshot()
        observation = self._observation()
        return {
            "active": True,
            "observation": observation.model_dump(mode="json"),
            "episode_log": self._episode_log(),
            "current_task_configuration": {
                "scenario_id": observation.scenario_id,
                "difficulty": scenario.difficulty,
                "max_steps": scenario.max_steps,
                "objective": scenario.objective,
                "title": scenario.title,
                "policy_version": self._policy_version(),
                "policy_transition_step": scenario.policy_transition_step,
                "policy_transition_to": scenario.policy_transition_to,
            },
            "policy_rules": policy_rules_for(self._policy_version()),
            "internal_variables": {
                "clarification_received": self.clarification_received,
                "episode_phase": self.episode_phase,
                "simulated_offset_hours": self._simulated_offset_hours,
                "snooze_sla_violations": self._snooze_sla_violations,
                "done": self.done,
                "steps_taken": len(self.action_history),
                "active_policy_version": self._policy_version(),
                "difficulty_mode": self._difficulty_mode,
                "specialist_consults_used": self._specialist_consults_used,
                "specialist_consult_budget_remaining": self._specialist_consult_budget_remaining,
            },
        }

    def debug_state(self) -> dict[str, Any]:
        if self.current_scenario is None:
            return {
                "active": False,
                "ground_truth": None,
                "dataset_reference": None,
                "episode_log": [],
                "current_task_configuration": None,
                "policy_rules": [],
                "internal_variables": {},
            }

        scenario = self._active_snapshot()
        active_snapshot = self._grade_snapshot()
        state = self.state()
        state["ground_truth"] = build_ground_truth_payload(
            scenario,
            active_snapshot,
            policy_version=self._policy_version(),
        )
        state["dataset_reference"] = scenario.model_dump(mode="json")
        return state

    def render(self, mode: str = "human") -> str | None:
        if mode != "human":
            return None
        if self.current_scenario is None:
            return "Environment not reset."
        observation = self._observation()
        return (
            f"Scenario: {observation.scenario_id} ({observation.difficulty}) | "
            f"Phase: {observation.episode_phase} | Step {observation.steps_taken}/{observation.max_steps}\n"
            f"Subject: {observation.current_email.subject}\n"
            f"Policy: {observation.policy_version} | Age: {observation.issue_age_hours}h | "
            f"Consult budget: {observation.specialist_consult_budget_remaining}"
        )
