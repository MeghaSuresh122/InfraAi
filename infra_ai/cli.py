"""Minimal CLI to drive the workflow (mock-friendly)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import uvicorn

from infra_ai.logging_config import get_logger, setup_logging
from infra_ai.runner import invoke_until_interrupt, resume_run

logger = get_logger(__name__)


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
    
    # Setup logging
    setup_logging(level=logging.INFO)
    logger.info("InfraAi CLI started with command: %s", args.cmd)
    
    if args.cmd == "serve":
        logger.info("Starting FastAPI server on %s:%s", args.host, args.port)
        uvicorn.run("infra_ai.api.main:app", host=args.host, port=args.port, reload=False)
    elif args.cmd == "run":
        logger.info("Running workflow with text: %s", args.text[:100] if args.text else "(empty)")
        if args.mock_llm:
            logger.info("Mock LLM mode enabled")
            os.environ["INFRA_AI_MOCK_LLM"] = "1"
        try:
            tid, state, interrupts = invoke_until_interrupt(
                {"raw_user_text": args.text, "raw_user_configs": {}},
            )
            logger.info("Workflow execution completed. Thread ID: %s, Interrupts: %d", tid, len(interrupts))
            json.dump({"thread_id": tid, "state": state, "interrupts": interrupts}, sys.stdout, indent=2)
            sys.stdout.write("\n")
        except Exception as e:
            logger.error("Workflow execution failed: %s", e, exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
