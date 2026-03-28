from __future__ import annotations

from business_policy_env.server import app as app


def main() -> None:
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, log_level="info")


if __name__ == "__main__":
    main()
