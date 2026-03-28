from __future__ import annotations

from fastapi import FastAPI

from .environment import BusinessPolicyComplianceEnv
from .models import Observation, ResetRequest, StepRequest, StepResult

app = FastAPI(
    title="Business Policy Compliance and Customer Resolution Environment",
    description="An OpenEnv-style environment for policy-aware customer support reasoning under uncertainty.",
    version="1.0.0",
)
env = BusinessPolicyComplianceEnv()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks")
def tasks() -> dict[str, list[str]]:
    return env.available_tasks()


@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest | None = None) -> Observation:
    payload = request or ResetRequest()
    return env.reset(task_name=payload.task_name, scenario_id=payload.scenario_id)


@app.post("/step", response_model=StepResult)
def step(request: StepRequest) -> StepResult:
    observation, reward, done, info = env.step(request.action)
    return StepResult(observation=observation, reward=reward, done=done, info=info)


@app.get("/state")
def state() -> dict:
    return env.state()
