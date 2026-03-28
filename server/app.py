"""ASGI entrypoint used by OpenEnv and direct uvicorn launches."""

from __future__ import annotations

from business_policy_env.server import app as app
from business_policy_env.server_main import main as run_server

__all__ = ["app", "main"]


def main() -> None:
    run_server()

if __name__ == "__main__":
    main()
