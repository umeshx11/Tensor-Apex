---
title: Business Policy Compliance Environment
emoji: ??
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
short_description: Policy-aware customer resolution environment for OpenEnv agents.
---

# Business Policy Compliance and Customer Resolution Environment

This project is an OpenEnv-style environment for the Hugging Face and Meta AI competition. It is not an email assistant. It is the training world for an agent that must resolve customer tickets while following explicit business policy, handling ambiguity honestly, and respecting SLA-driven time pressure.

The core skill under test is policy-aware reasoning under uncertainty. Agents do not get rewarded for sounding helpful. They get rewarded for following documented rules, recognizing when they do not have enough information, and reacting differently when the same issue has aged past policy thresholds.

## Why This Environment Is Different

Most inbox environments stop at classification. This one adds three hard constraints that make grader logic objective and difficult to game:

1. Policy constraints: refund thresholds, VIP handling, SLA breaches, legal threats, and suspended-account routing are all explicit and deterministic.
2. Ambiguity handling: some tickets must be answered with `request_info` first or the episode scores `0.0`.
3. Temporal reasoning: issue age is computed from thread timestamps and directly changes the correct priority.

## Policy Rules

The agent sees every active rule in the observation:

- Refunds over `$500` require escalation.
- VIP customers require `high` or `urgent` priority.
- Issues open for more than `72` hours require `urgent` priority.
- Complaints mentioning legal action or lawsuits require immediate escalation.
- Suspended accounts must be routed to the billing team.

## Tasks

### Easy
Single-turn, unambiguous tickets with obvious policy signals. The grader checks correct department routing, priority assignment, and policy compliance.

### Medium
Ambiguous tickets that require `request_info` before the correct policy-aware action sequence can happen. The grader checks ambiguity recognition, clarifying-question quality, post-clarification policy compliance, and response quality.

### Hard
Multi-turn threads with SLA pressure, policy triggers, and response requirements that must acknowledge history. The grader weights history-aware response quality heavily so shallow keyword agents do poorly even when they catch some metadata rules.

## Observation Space

Each observation contains:

- The current visible email and full visible thread.
- Sender tier, account flags, and visible refund amount.
- Issue age in hours, computed from timestamps.
- Steps taken, max steps, and prior action history.
- Plain-English policy rules.
- A task objective describing the current ticket goal.

## Action Space

The environment accepts structured Pydantic actions:

- `categorize`
- `set_priority`
- `draft_response`
- `escalate`
- `mark_spam`
- `request_info`

Every action includes a `reasoning` field for debugging, but the field is not graded.

## Reward Design

| Component | Trigger | Range | Example |
| --- | --- | --- | --- |
| Valid action bonus | Any schema-valid action | `+0.05` | Agent submits a well-formed `categorize` action |
| Policy penalty | Current action violates an active policy rule | `-0.2` | Agent sets `low` priority for a VIP ticket |
| Partial progress | Correct work completed before episode end | `0.0` to task score so far | Correctly categorizing a ticket before final resolution |
| Final grader score | Episode ends and grader runs | `0.0` to `1.0` | Full task sequence matches ground truth |
| Efficiency bonus | Episode finishes in at most half the allowed steps | `+0.1` | Easy task finished in two steps |
| Redundancy penalty | Repeated action types in the same episode | `-0.05` each repeat | Drafting multiple redundant responses |

Final reward is clamped to `0.0` through `1.0`. Detailed reward breakdowns live in the `info` dictionary returned by `step()`.

## Verified Rule Baseline Scores

Running `python baseline.py --agent rule` on a fresh clone with the pinned requirements currently yields:

- Easy: `0.75`
- Medium: `0.4459`
- Hard: `0.2`

These are intentionally non-trivial: the rule baseline handles obvious policy metadata but struggles once ambiguity and history-aware response quality dominate.

## API

The FastAPI app exposes:

- `GET /health`
- `GET /tasks`
- `POST /reset`
- `POST /step`
- `GET /state`

`state()` returns internal environment state, including ground truth, the dataset reference, SQLite episode logs, task configuration, and policy configuration. `reset()` and `step()` expose only the agent-facing observation.

## Project Layout

- `business_policy_env/models.py`: Pydantic models only.
- `business_policy_env/environment.py`: reset, step, state, and environment execution.
- `business_policy_env/tasks.py`: deterministic task graders and scenario registry.
- `business_policy_env/rewards.py`: reward shaping only.
- `business_policy_env/policies.py`: policy rules and rule checker.
- `business_policy_env/data_generation.py`: deterministic synthetic ticket generation.
- `business_policy_env/server.py`: FastAPI routes.
- `business_policy_env/baseline.py`: rule baseline and optional OpenAI baseline.
- `openenv.yaml`: environment metadata.
- `tests/test_environment.py`: determinism, API, reset, invalid-action, ambiguity, and policy tests.

## Local Run

```bash
python -m pip install -r requirements.txt
uvicorn business_policy_env.server:app --host 0.0.0.0 --port 7860
```

Run the baseline:

```bash
python baseline.py --agent rule
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

## Docker

The included `Dockerfile` uses `python:3.11-slim`, installs the pinned requirements, exposes port `7860`, defines a `/health` health check, and starts the FastAPI server with `uvicorn`.
