from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run("business_policy_env.server:app", host="0.0.0.0", port=7860, log_level="info")
