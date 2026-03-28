import unittest

from fastapi.testclient import TestClient

import baseline as root_baseline
from business_policy_env.baseline import RuleBasedAgent, run_baseline
from business_policy_env.data_generation import scenario_ids_for_task
from business_policy_env.environment import BusinessPolicyComplianceEnv
from business_policy_env.models import Action
from business_policy_env.server import app
from business_policy_env.session_manager import (
    RateLimitError,
    SessionCapacityError,
    SessionManager,
    get_session_manager,
)
from business_policy_env.tasks import (
    build_ground_truth_payload,
    component_scores,
    grade_actions,
    is_substantive_question,
    scenario_registry,
)
from gradio_app import reset_episode, take_action


class EnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        get_session_manager().close_all()
        self.env = BusinessPolicyComplianceEnv()

    def tearDown(self) -> None:
        self.env.close()
        get_session_manager().close_all()

    def _expected_actions_for(self, scenario_id: str) -> tuple[list[Action], dict]:
        scenario = scenario_registry()[scenario_id]
        snapshot = scenario.clarification_snapshot or scenario.initial_snapshot
        ground_truth = build_ground_truth_payload(scenario, snapshot)
        actions: list[Action] = []
        if scenario.ground_truth.requires_request_info:
            actions.append(
                Action(
                    action_type="request_info",
                    reasoning="Need clarification before routing.",
                    clarifying_question="Can you confirm the invoice, the charge, and whether you want a refund?",
                )
            )
        if scenario.ground_truth.expected_flag_fraud:
            actions.append(
                Action(
                    action_type="flag_fraud",
                    reasoning="Fraud signal detected.",
                    fraud_reason="Suspicious pattern indicates fraud risk.",
                )
            )
        actions.append(
            Action(
                action_type="categorize",
                reasoning="Route to the expected department.",
                category=scenario.ground_truth.expected_category,
            )
        )
        actions.append(
            Action(
                action_type="set_priority",
                reasoning="Use the expected SLA priority.",
                priority=scenario.ground_truth.expected_priority,
            )
        )
        if scenario.ground_truth.expected_escalation:
            actions.append(
                Action(
                    action_type="escalate",
                    reasoning="Escalate per policy.",
                    escalation_reason=scenario.ground_truth.expected_escalation_reason or "Policy escalation.",
                )
            )
        if scenario.difficulty != "easy":
            response = (
                "We understand the delay, have reviewed the history, and will send an update after the review "
                "is escalated to the right team."
            )
            actions.append(
                Action(
                    action_type="draft_response",
                    reasoning="Send a policy-safe customer reply.",
                    response_text=response,
                )
            )
        return actions, ground_truth

    def test_graders_are_deterministic(self) -> None:
        for scenario_id in ["easy_vip_refund", "medium_charge_or_bug", "hard_vip_refund_lawyer"]:
            actions, ground_truth = self._expected_actions_for(scenario_id)
            first = grade_actions(actions, ground_truth)
            second = grade_actions(actions, ground_truth)
            self.assertEqual(first, second)

    def test_scenario_bank_expanded_beyond_original_catalog(self) -> None:
        self.assertGreaterEqual(len(scenario_ids_for_task()), 52)

    def test_substantive_question_rejects_generic_filler(self) -> None:
        self.assertFalse(is_substantive_question("Can you provide more?"))
        self.assertFalse(is_substantive_question("What is the invoice, charge, date, reason, and account?"))
        self.assertTrue(
            is_substantive_question("Can you confirm the invoice amount and the card used for this charge?")
        )

    def test_invalid_action_penalty_is_negative_and_state_is_stable(self) -> None:
        self.env.reset(scenario_id="easy_vip_refund")
        observation, reward, done, info = self.env.step({"action_type": "categorize", "reasoning": "Missing field"})
        self.assertEqual(reward, -0.1)
        self.assertFalse(done)
        self.assertFalse(info["valid_action"])
        self.assertEqual(observation.steps_taken, 0)
        self.assertEqual(self.env.debug_state()["episode_log"], [])

    def test_policy_violation_penalty_can_make_step_reward_negative(self) -> None:
        self.env.reset(scenario_id="easy_vip_refund")
        observation, reward, done, info = self.env.step(
            Action(
                action_type="set_priority",
                reasoning="Incorrectly low priority.",
                priority="low",
            )
        )
        self.assertEqual(observation.steps_taken, 1)
        self.assertLessEqual(reward, 0.0)
        self.assertFalse(done)
        self.assertTrue(info["policy_violations"])
        self.assertEqual(info["reward_breakdown"]["policy_penalty"], -0.2)

    def test_public_state_is_sanitized_but_debug_state_keeps_answer_key(self) -> None:
        self.env.reset(scenario_id="easy_vip_refund")
        public_state = self.env.state()
        debug_state = self.env.debug_state()
        self.assertIn("observation", public_state)
        self.assertNotIn("ground_truth", public_state)
        self.assertNotIn("dataset_reference", public_state)
        self.assertIn("ground_truth", debug_state)
        self.assertIn("dataset_reference", debug_state)

    def test_emails_remaining_tracks_hidden_clarification(self) -> None:
        observation = self.env.reset(scenario_id="medium_charge_or_bug")
        self.assertEqual(observation.emails_remaining, 1)

        low_quality_obs, _, _, _ = self.env.step(
            Action(
                action_type="request_info",
                reasoning="Need more detail.",
                clarifying_question="?",
            )
        )
        self.assertFalse(low_quality_obs.clarification_received)
        self.assertEqual(low_quality_obs.emails_remaining, 1)

        self.env.reset(scenario_id="medium_charge_or_bug")
        clarified_obs, _, _, _ = self.env.step(
            Action(
                action_type="request_info",
                reasoning="Need more detail before routing.",
                clarifying_question="Can you confirm the invoice, the account, and whether you want a refund?",
            )
        )
        self.assertTrue(clarified_obs.clarification_received)
        self.assertEqual(clarified_obs.emails_remaining, 0)
        self.assertIn("duplicate renewal charge", clarified_obs.current_email.body.lower())

    def test_partial_observability_hides_some_account_flags(self) -> None:
        scenario = scenario_registry()["hard_hidden_flag_dispute"]
        observation = self.env.reset(scenario_id="hard_hidden_flag_dispute")
        self.assertEqual(observation.hidden_flags, 1)
        self.assertLess(len(observation.account_flags), len(scenario.initial_snapshot.account_flags))
        self.assertNotEqual(sorted(observation.account_flags), sorted(scenario.initial_snapshot.account_flags))

    def test_ambiguous_ticket_retains_partial_credit_when_request_info_is_skipped(self) -> None:
        self.env.reset(scenario_id="medium_charge_or_bug")
        final_info = None
        done = False
        scripted_actions = [
            Action(action_type="categorize", reasoning="Billing ticket.", category="billing"),
            Action(action_type="set_priority", reasoning="VIP and policy-sensitive.", priority="high"),
            Action(
                action_type="escalate",
                reasoning="Refund threshold requires escalation.",
                escalation_reason="Policy.",
            ),
            Action(
                action_type="draft_response",
                reasoning="Send update.",
                response_text="We understand the charge issue and will review the account and send an update.",
            ),
        ]
        while not done:
            step_index = self.env.debug_state()["internal_variables"]["steps_taken"]
            action = scripted_actions[min(step_index, len(scripted_actions) - 1)]
            _, _, done, final_info = self.env.step(action)
        self.assertIsNotNone(final_info)
        self.assertGreater(final_info["final_score"], 0.0)

    def test_keyword_dump_scores_lower_than_coherent_response(self) -> None:
        scenario = scenario_registry()["hard_vip_refund_lawyer"]
        ground_truth = build_ground_truth_payload(scenario, scenario.initial_snapshot)
        base_actions = [
            Action(action_type="categorize", reasoning="Legal issue.", category="legal"),
            Action(action_type="set_priority", reasoning="Urgent due to age and legal risk.", priority="urgent"),
            Action(action_type="escalate", reasoning="Escalate now.", escalation_reason="Legal threat."),
        ]
        keyword_dump_actions = base_actions + [
            Action(
                action_type="draft_response",
                reasoning="Reply.",
                response_text="refund escalated review delay waiting follow-up",
            )
        ]
        coherent_actions = base_actions + [
            Action(
                action_type="draft_response",
                reasoning="Reply.",
                response_text=(
                    "We understand the refund delay and the follow-up history. "
                    "We have escalated the review and will send you an update today after the team finishes its review."
                ),
            )
        ]
        self.assertGreater(
            grade_actions(coherent_actions, ground_truth),
            grade_actions(keyword_dump_actions, ground_truth),
        )
        coherent_components = component_scores(coherent_actions, ground_truth)
        keyword_components = component_scores(keyword_dump_actions, ground_truth)
        self.assertGreater(coherent_components["response_completeness"], keyword_components["response_completeness"])

    def test_policy_version_can_transition_mid_episode(self) -> None:
        observation = self.env.reset(scenario_id="hard_policy_shift_fraud_upgrade")
        self.assertEqual(observation.policy_version, "v1")
        for action in [
            Action(action_type="categorize", reasoning="Billing issue.", category="billing"),
            Action(action_type="set_priority", reasoning="Premier issue.", priority="high"),
            Action(
                action_type="draft_response",
                reasoning="Send an update before the new policy applies.",
                response_text="We are reviewing the account and will send an update today.",
            ),
        ]:
            observation, _, _, _ = self.env.step(action)
        self.assertEqual(observation.policy_version, "v2")
        _, _, _, info = self.env.step(
            Action(
                action_type="draft_response",
                reasoning="Still responding without a fraud flag.",
                response_text="We are still reviewing the account and will update you again.",
            )
        )
        self.assertIn("Fraud indicators require flag_fraud before resolution actions.", info["policy_violations"])

    def test_task_resets_build_variants_from_shuffled_families(self) -> None:
        env = BusinessPolicyComplianceEnv(seed=11)
        canonical_order = scenario_ids_for_task("hard")
        observed_families: list[str] = []
        for _ in range(len(canonical_order)):
            observation = env.reset(task_name="hard")
            self.assertIn("__variant_", observation.scenario_id)
            observed_families.append(observation.scenario_id.split("__variant_")[0])
        env.close()
        self.assertCountEqual(observed_families, canonical_order)
        self.assertNotEqual(observed_families, canonical_order)

    def test_fastapi_endpoints_return_sanitized_state(self) -> None:
        client = TestClient(app)
        reset_response = client.post("/reset", json={"scenario_id": "easy_sla_breach"})
        self.assertEqual(reset_response.status_code, 200)
        step_response = client.post(
            "/step",
            json={
                "action": {
                    "action_type": "set_priority",
                    "reasoning": "SLA breach requires urgent handling.",
                    "priority": "urgent",
                }
            },
        )
        self.assertEqual(step_response.status_code, 200)
        state_response = client.get("/state")
        self.assertEqual(state_response.status_code, 200)
        body = state_response.json()
        self.assertTrue(body["active"])
        self.assertIn("observation", body)
        self.assertNotIn("ground_truth", body)
        self.assertNotIn("dataset_reference", body)

    def test_session_isolation(self) -> None:
        client = TestClient(app)
        client.post("/reset", json={"scenario_id": "easy_vip_refund"}, headers={"X-Session-Id": "session_a"})
        client.post("/reset", json={"scenario_id": "easy_sla_breach"}, headers={"X-Session-Id": "session_b"})
        state_a = client.get("/state", headers={"X-Session-Id": "session_a"}).json()
        state_b = client.get("/state", headers={"X-Session-Id": "session_b"}).json()
        self.assertNotEqual(
            state_a["current_task_configuration"]["title"],
            state_b["current_task_configuration"]["title"],
        )

    def test_snooze_crosses_sla_threshold(self) -> None:
        self.env.reset(scenario_id="easy_sla_marginal")
        obs, _, _, info = self.env.step(
            Action(
                action_type="snooze",
                reasoning="Waiting for customer reply.",
                snooze_hours=2,
            )
        )
        self.assertGreater(obs.issue_age_hours, 72)
        self.assertEqual(info["reward_breakdown"]["snooze_sla_penalty"], -0.1)

    def test_snooze_penalty_accumulates_after_sla_crossing(self) -> None:
        self.env.reset(scenario_id="easy_sla_marginal")
        self.env.step(
            Action(
                action_type="snooze",
                reasoning="Waiting for customer reply.",
                snooze_hours=2,
            )
        )
        _, _, _, info = self.env.step(
            Action(
                action_type="snooze",
                reasoning="Still waiting for customer reply.",
                snooze_hours=1,
            )
        )
        self.assertEqual(info["reward_breakdown"]["snooze_sla_penalty"], -0.2)

    def test_flag_fraud_scores_correctly(self) -> None:
        self.env.reset(scenario_id="hard_fraud_chargeback")
        _, reward, _, _ = self.env.step(
            Action(
                action_type="flag_fraud",
                reasoning="Multiple rapid refund requests from different cards.",
                fraud_reason="Chargeback pattern consistent with card testing fraud.",
            )
        )
        self.assertGreater(reward, 0.0)

    def test_rule_baseline_runs_one_episode(self) -> None:
        scenario_id = "hard_old_invoice_question"
        observation = self.env.reset(scenario_id=scenario_id)
        agent = RuleBasedAgent()
        done = False
        info = {}
        while not done:
            observation, _, done, info = self.env.step(agent.next_action(observation))
        self.assertIn("final_score", info)

    def test_root_baseline_entrypoint_exports_main(self) -> None:
        self.assertTrue(callable(root_baseline.main))

    def test_rule_baseline_reports_min_and_max_scores(self) -> None:
        result = run_baseline(agent_name="rule")
        self.assertIn("easy", result["results"])
        self.assertIn("min_final_score", result["results"]["easy"])
        self.assertIn("max_final_score", result["results"]["easy"])

    def test_hard_conflict_and_ordering_components_reward_better_reasoning(self) -> None:
        scenario = scenario_registry()["hard_concurrent_violations"]
        ground_truth = build_ground_truth_payload(scenario, scenario.initial_snapshot)
        weak_actions = [
            Action(action_type="flag_fraud", reasoning="Fraud issue.", fraud_reason="Flagging risk."),
            Action(action_type="categorize", reasoning="Billing issue.", category="billing"),
            Action(action_type="set_priority", reasoning="Urgent issue.", priority="urgent"),
            Action(action_type="escalate", reasoning="Policy escalation.", escalation_reason="Needs escalation."),
            Action(
                action_type="draft_response",
                reasoning="Send an update.",
                response_text="We are reviewing this now.",
            ),
        ]
        strong_actions = [
            Action(
                action_type="flag_fraud",
                reasoning="Fraud risk conflicts with billing and legal pressure.",
                fraud_reason="Unauthorized duplicate charge requires fraud review.",
            ),
            Action(
                action_type="categorize",
                reasoning="Billing issue with overlapping fraud and legal signals.",
                category="billing",
            ),
            Action(
                action_type="set_priority",
                reasoning="Urgent due to fraud, billing, and legal urgency.",
                priority="urgent",
            ),
            Action(
                action_type="escalate",
                reasoning="Escalate after fraud review because legal counsel is involved.",
                escalation_reason="Legal pressure and high-value billing risk.",
            ),
            Action(
                action_type="draft_response",
                reasoning="Acknowledge the fraud, billing, and legal conflict clearly.",
                response_text=(
                    "We understand the unauthorized billing charge and the legal urgency. "
                    "Our fraud and billing teams are reviewing the issue now, and we will send a concrete update today."
                ),
            ),
        ]
        weak_components = component_scores(weak_actions, ground_truth)
        strong_components = component_scores(strong_actions, ground_truth)
        self.assertGreater(strong_components["contradiction_detection"], weak_components["contradiction_detection"])
        self.assertGreater(grade_actions(strong_actions, ground_truth), grade_actions(weak_actions, ground_truth))

    def test_hard_ordering_component_penalizes_wrong_sequence(self) -> None:
        scenario = scenario_registry()["hard_three_signal_precedence"]
        ground_truth = build_ground_truth_payload(scenario, scenario.initial_snapshot)
        ordered_actions = [
            Action(action_type="flag_fraud", reasoning="Fraud first.", fraud_reason="Unauthorized activity."),
            Action(action_type="categorize", reasoning="Billing issue.", category="billing"),
            Action(action_type="set_priority", reasoning="Urgent issue.", priority="urgent"),
            Action(action_type="escalate", reasoning="Legal escalation.", escalation_reason="Legal and policy risk."),
            Action(
                action_type="draft_response",
                reasoning="Respond after the required actions.",
                response_text="We have flagged the fraud issue, escalated the case, and will update you today.",
            ),
        ]
        wrong_order_actions = [
            Action(
                action_type="escalate",
                reasoning="Escalate immediately.",
                escalation_reason="Legal and policy risk.",
            ),
            Action(
                action_type="flag_fraud",
                reasoning="Fraud second, which is wrong.",
                fraud_reason="Unauthorized activity.",
            ),
            Action(action_type="categorize", reasoning="Billing issue.", category="billing"),
            Action(action_type="set_priority", reasoning="Urgent issue.", priority="urgent"),
            Action(
                action_type="draft_response",
                reasoning="Respond after the required actions.",
                response_text="We have escalated the case and flagged the fraud issue for review today.",
            ),
        ]
        ordered_components = component_scores(ordered_actions, ground_truth)
        wrong_components = component_scores(wrong_order_actions, ground_truth)
        self.assertGreater(ordered_components["ordering_correctness"], wrong_components["ordering_correctness"])

    def test_done_branch_reports_episode_complete_for_valid_follow_on_actions(self) -> None:
        self.env.reset(scenario_id="easy_vip_refund")
        for action in [
            Action(action_type="categorize", reasoning="Billing ticket.", category="billing"),
            Action(action_type="set_priority", reasoning="VIP refund.", priority="high"),
            Action(
                action_type="escalate",
                reasoning="Refund threshold.",
                escalation_reason="Refund exceeds threshold.",
            ),
        ]:
            _, _, done, _ = self.env.step(action)
        self.assertTrue(done)
        _, _, done_again, info = self.env.step(
            Action(
                action_type="draft_response",
                reasoning="Episode already ended.",
                response_text="Following up with an additional note.",
            )
        )
        self.assertTrue(done_again)
        self.assertTrue(info["valid_action"])
        self.assertTrue(info["episode_complete"])
        self.assertFalse(info["action_accepted"])

    def test_render_returns_human_summary(self) -> None:
        self.env.reset(scenario_id="easy_sla_breach")
        rendered = self.env.render()
        self.assertIsInstance(rendered, str)
        self.assertIn("Scenario:", rendered)
        self.assertIn("Policy:", rendered)


class SessionManagerTests(unittest.TestCase):
    def test_session_ttl_eviction(self) -> None:
        manager = SessionManager(session_ttl_seconds=10)
        manager.get_or_create("session-a", now=0.0)
        self.assertIsNotNone(manager.get("session-a", now=5.0))
        self.assertIsNone(manager.get("session-a", now=16.0))
        manager.close_all()

    def test_max_session_cap_is_enforced(self) -> None:
        manager = SessionManager(max_sessions=1)
        manager.get_or_create("session-a", now=0.0)
        with self.assertRaises(SessionCapacityError):
            manager.get_or_create("session-b", now=0.0)
        manager.close_all()

    def test_rate_limit_is_enforced(self) -> None:
        manager = SessionManager(rate_limit_per_minute=2)
        manager.enforce_rate_limit("client", "session-a", now=0.0)
        manager.enforce_rate_limit("client", "session-a", now=10.0)
        with self.assertRaises(RateLimitError):
            manager.enforce_rate_limit("client", "session-a", now=20.0)
        manager.close_all()


class GradioIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        get_session_manager().close_all()

    def tearDown(self) -> None:
        get_session_manager().close_all()

    def test_gradio_handlers_use_isolated_sessions(self) -> None:
        _, _, status_a, _, session_a = reset_episode("easy_vip_refund", None)
        _, _, status_b, _, session_b = reset_episode("easy_sla_breach", None)
        self.assertIn("Ready for actions", status_a)
        self.assertIn("Ready for actions", status_b)
        self.assertNotEqual(session_a, session_b)

        take_action(
            session_a,
            "set_priority",
            "",
            "high",
            "",
            "",
            "",
            "",
            0,
            "Setting priority for the first UI session.",
        )
        manager = get_session_manager()
        state_a = manager.get(session_a).state()
        state_b = manager.get(session_b).state()
        self.assertEqual(state_a["internal_variables"]["steps_taken"], 1)
        self.assertEqual(state_b["internal_variables"]["steps_taken"], 0)


if __name__ == "__main__":
    unittest.main()
