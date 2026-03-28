from __future__ import annotations

import argparse
import json
import os
from statistics import mean

from .environment import BusinessPolicyComplianceEnv
from .models import Action, Observation
from .tasks import scenarios_for_task


class RuleBasedAgent:
    def next_action(self, observation: Observation) -> Action:
        action_types = [record.action.action_type for record in observation.action_history]
        body = observation.current_email.body.lower()
        subject = observation.current_email.subject.lower()
        combined_text = f"{subject} {body}"

        if not observation.clarification_received and "request_info" not in action_types and self._needs_clarification(body):
            return Action(
                action_type="request_info",
                reasoning="The message is too vague to safely resolve without a follow-up question.",
                clarifying_question="Can you confirm the order or invoice involved and what outcome you want?",
            )

        if "categorize" not in action_types:
            return Action(
                action_type="categorize",
                reasoning="Route using simple keyword rules.",
                category=self._category(combined_text),
            )

        if "set_priority" not in action_types:
            return Action(
                action_type="set_priority",
                reasoning="Priority follows age first, then VIP metadata.",
                priority=self._priority(observation),
            )

        if "escalate" not in action_types and self._needs_escalation(observation):
            return Action(
                action_type="escalate",
                reasoning="Refund amount exceeds the automatic approval threshold.",
                escalation_reason="Refund exceeds $500.",
            )

        return Action(
            action_type="draft_response",
            reasoning="Send a short acknowledgement.",
            response_text="Thanks for the message. We will follow up soon.",
        )

    def _needs_clarification(self, body: str) -> bool:
        clear_signals = [
            "refund",
            "invoice",
            "charge",
            "payment",
            "billing",
            "login",
            "password",
            "app",
            "error",
            "update",
            "replacement",
            "return",
            "exchange",
        ]
        has_clear_signal = any(signal in body for signal in clear_signals)
        return len(body.split()) < 20 and not has_clear_signal

    def _category(self, combined_text: str) -> str:
        if any(keyword in combined_text for keyword in ["refund", "charge", "invoice", "payment", "billing", "card"]):
            return "billing"
        if any(keyword in combined_text for keyword in ["login", "password", "app", "error", "update", "load"]):
            return "technical_support"
        if any(keyword in combined_text for keyword in ["replacement", "return", "exchange"]):
            return "returns"
        return "customer_success"

    def _priority(self, observation: Observation) -> str:
        if observation.issue_age_hours > 72:
            return "urgent"
        if observation.sender_tier == "vip":
            return "high"
        return "medium"

    def _needs_escalation(self, observation: Observation) -> bool:
        return bool(observation.refund_amount and observation.refund_amount > 500)


class OpenAIBaselineAgent:
    def __init__(self, model: str) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional path
            raise RuntimeError("openai package is not installed.") from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the OpenAI baseline.")

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def next_action(self, observation: Observation) -> Action:  # pragma: no cover - optional path
        prompt = {
            "task_objective": observation.task_objective,
            "policy_rules": observation.policy_rules,
            "issue_age_hours": observation.issue_age_hours,
            "sender_tier": observation.sender_tier,
            "account_flags": observation.account_flags,
            "refund_amount": observation.refund_amount,
            "thread": [message.model_dump(mode="json") for message in observation.thread],
            "action_history": [record.model_dump(mode="json") for record in observation.action_history],
        }
        response = self._client.responses.create(
            model=self._model,
            input=[
                {
                    "role": "system",
                    "content": "You are a customer-support policy agent. Return exactly one JSON object that matches the Action schema.",
                },
                {"role": "user", "content": json.dumps(prompt)},
            ],
        )
        return Action.model_validate_json(response.output_text)



def run_episode(env: BusinessPolicyComplianceEnv, agent, scenario_id: str) -> dict:
    observation = env.reset(scenario_id=scenario_id)
    reward = 0.0
    done = False
    info: dict = {}
    while not done:
        action = agent.next_action(observation)
        observation, reward, done, info = env.step(action)
    return {
        "scenario_id": scenario_id,
        "reward": reward,
        "final_score": info.get("final_score", 0.0),
        "component_scores": info.get("component_scores", {}),
    }



def run_baseline(agent_name: str = "rule", model: str = "gpt-4.1-mini") -> dict:
    env = BusinessPolicyComplianceEnv()
    agent = RuleBasedAgent() if agent_name == "rule" else OpenAIBaselineAgent(model=model)
    summary: dict[str, object] = {"agent": agent_name, "results": {}}

    for task_name in ["easy", "medium", "hard"]:
        task_results = [run_episode(env, agent, scenario.scenario_id) for scenario in scenarios_for_task(task_name)]
        summary["results"][task_name] = {
            "mean_final_score": round(mean(result["final_score"] for result in task_results), 4),
            "scenarios": task_results,
        }

    return summary



def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline agents against the Business Policy Compliance environment.")
    parser.add_argument("--agent", choices=["rule", "openai"], default="rule")
    parser.add_argument("--model", default="gpt-4.1-mini")
    args = parser.parse_args()
    print(json.dumps(run_baseline(agent_name=args.agent, model=args.model), indent=2))


if __name__ == "__main__":
    main()
