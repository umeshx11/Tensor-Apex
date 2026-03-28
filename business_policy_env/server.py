from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request

from .data_generation import scenario_ids_for_task
from .models import Observation, ResetRequest, StepRequest, StepResult
from .session_manager import RateLimitError, SessionCapacityError, get_session_manager

app = FastAPI(
    title="Business Policy Compliance and Customer Resolution Environment",
    description="An OpenEnv-style environment for policy-aware customer support reasoning under uncertainty.",
    version="1.0.0",
)


def _session_or_default(x_session_id: str | None) -> str:
    return x_session_id or "default"


def _client_host(request: Request) -> str:
    return request.client.host if request.client is not None else "unknown"


def _enforce_rate_limit(request: Request, session_id: str) -> None:
    try:
        get_session_manager().enforce_rate_limit(_client_host(request), session_id)
    except RateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc


def _get_or_create_env(session_id: str):
    try:
        return get_session_manager().get_or_create(session_id)
    except SessionCapacityError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.on_event("shutdown")
def shutdown() -> None:
    get_session_manager().close_all()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks")
def tasks(request: Request, x_session_id: str | None = Header(default=None)) -> dict[str, list[str]]:
    session_id = _session_or_default(x_session_id)
    _enforce_rate_limit(request, session_id)
    return {
        "easy": scenario_ids_for_task("easy"),
        "medium": scenario_ids_for_task("medium"),
        "hard": scenario_ids_for_task("hard"),
    }


@app.post("/reset", response_model=Observation)
def reset(
    request: Request,
    payload: ResetRequest | None = None,
    x_session_id: str | None = Header(default=None),
) -> Observation:
    session_id = _session_or_default(x_session_id)
    _enforce_rate_limit(request, session_id)
    env = _get_or_create_env(session_id)
    request_payload = payload or ResetRequest()
    return env.reset(task_name=request_payload.task_name, scenario_id=request_payload.scenario_id)


@app.post("/step", response_model=StepResult)
def step(
    request: Request,
    payload: StepRequest,
    x_session_id: str | None = Header(default=None),
) -> StepResult:
    session_id = _session_or_default(x_session_id)
    _enforce_rate_limit(request, session_id)
    env = get_session_manager().get(session_id)
    if env is None:
        raise HTTPException(status_code=400, detail="Session not found. Call /reset first.")
    observation, reward, done, info = env.step(payload.action)
    return StepResult(observation=observation, reward=reward, done=done, info=info)


@app.get("/state")
def state(request: Request, x_session_id: str | None = Header(default=None)) -> dict[str, Any]:
    session_id = _session_or_default(x_session_id)
    _enforce_rate_limit(request, session_id)
    env = get_session_manager().get(session_id)
    if env is None:
        return {"active": False, "detail": "Session not found. Call /reset first."}
    return env.state()


@app.delete("/session")
def close_session(request: Request, x_session_id: str | None = Header(default=None)) -> dict[str, str]:
    session_id = _session_or_default(x_session_id)
    _enforce_rate_limit(request, session_id)
    get_session_manager().close(session_id)
    return {"closed": session_id}
