"""Minimal CLI to drive the workflow (mock-friendly)."""

from __future__ import annotations

import argparse
import json
import os
import sys

import uvicorn

from infra_ai.runner import invoke_until_interrupt, resume_run


def main() -> None:
    parser = argparse.ArgumentParser(prog="infra-ai")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_api = sub.add_parser("serve", help="Run FastAPI server")
    p_api.add_argument("--host", default="127.0.0.1")
    p_api.add_argument("--port", type=int, default=8000)

    p_run = sub.add_parser("run", help="Run graph once until first interrupt (stdout JSON)")
    p_run.add_argument("--text", default="", help="User requirement text")
    p_run.add_argument("--mock-llm", action="store_true", help="Set INFRA_AI_MOCK_LLM=1")

    args = parser.parse_args()
    if args.cmd == "serve":
        uvicorn.run("infra_ai.api.main:app", host=args.host, port=args.port, reload=False)
    elif args.cmd == "run":
        if args.mock_llm:
            os.environ["INFRA_AI_MOCK_LLM"] = "1"
        tid, state, interrupts = invoke_until_interrupt(
            {"raw_user_text": args.text, "raw_user_configs": {}},
        )
        json.dump({"thread_id": tid, "state": state, "interrupts": interrupts}, sys.stdout, indent=2)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
