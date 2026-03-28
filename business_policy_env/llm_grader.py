from __future__ import annotations

import json
import os
from typing import Any

import httpx

JUDGE_PROMPT = """
You are grading a customer support response. Score 0.0 to 1.0 on:
- Does it address the specific issue? (40%)
- Does it state a clear next step? (35%)
- Is the tone appropriate? (25%)

Ticket context: {context}
Expected resolution: {ground_truth}
Agent response: {response}

Reply with ONLY a JSON object: {{"score": 0.0, "reason": "..."}}
""".strip()

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _llm_judge_enabled() -> bool:
    return os.getenv("BUSINESS_POLICY_ENV_USE_LLM_JUDGE", "").strip().lower() in _TRUE_VALUES


def _extract_content_text(payload: dict[str, Any]) -> str:
    content = payload.get("content", [])
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    return text
    output_text = payload.get("output_text")
    if isinstance(output_text, str):
        return output_text
    return ""


def score_response_with_optional_llm(response_text: str, ground_truth: dict[str, Any]) -> float | None:
    if not _llm_judge_enabled():
        return None

    api_key = os.getenv("JUDGE_API_KEY")
    if not api_key:
        return None

    context = {
        "difficulty": ground_truth.get("difficulty"),
        "policy_version": ground_truth.get("policy_version"),
        "snapshot": ground_truth.get("snapshot"),
        "issue_age_hours": ground_truth.get("issue_age_hours"),
    }
    expected_resolution = {
        "expected_category": ground_truth.get("expected_category"),
        "expected_priority": ground_truth.get("expected_priority"),
        "expected_escalation": ground_truth.get("expected_escalation"),
        "expected_flag_fraud": ground_truth.get("expected_flag_fraud"),
        "response_keywords": ground_truth.get("response_keywords"),
        "history_keywords": ground_truth.get("history_keywords"),
    }

    payload = {
        "model": os.getenv("JUDGE_MODEL", "claude-3-5-haiku-latest"),
        "max_tokens": 128,
        "messages": [
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    context=json.dumps(context, ensure_ascii=True),
                    ground_truth=json.dumps(expected_resolution, ensure_ascii=True),
                    response=response_text,
                ),
            }
        ],
    }

    headers = {
        "x-api-key": api_key,
        "anthropic-version": os.getenv("JUDGE_API_VERSION", "2023-06-01"),
        "content-type": "application/json",
    }
    url = os.getenv("JUDGE_API_URL", "https://api.anthropic.com/v1/messages")

    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
        response_text_payload = _extract_content_text(response.json())
        if not response_text_payload:
            return None
        parsed = json.loads(response_text_payload)
        score = float(parsed["score"])
        return max(0.0, min(1.0, round(score, 4)))
    except (httpx.HTTPError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
