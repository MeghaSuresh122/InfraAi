from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from infra_ai.runner import invoke_until_interrupt, resume_run
from infra_ai.logging_config import setup_logging

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    yield

app = FastAPI(title="InfraAi", version="0.1.0", lifespan=lifespan)


class StartRunBody(BaseModel):
    raw_user_text: str = ""
    raw_user_configs: dict[str, Any] = Field(default_factory=dict)
    thread_id: str | None = None


class ResumeBody(BaseModel):
    resume: Any
    state_update: dict[str, Any] | None = None


@app.get("/health")
def health() -> dict[str, str]:
    import logging
    logging.getLogger("test_logger").info("HEALTH ENDPOINT CALLED")
    return {"status": "ok"}


@app.post("/v1/runs")
def start_run(body: StartRunBody) -> dict[str, Any]:
    tid, state, interrupts = invoke_until_interrupt(
        {"raw_user_text": body.raw_user_text, "raw_user_configs": body.raw_user_configs},
        thread_id=body.thread_id,
    )
    return {"thread_id": tid, "state": state, "interrupts": interrupts}


@app.post("/v1/runs/{thread_id}/resume")
def resume(thread_id: str, body: ResumeBody) -> dict[str, Any]:
    state, interrupts = resume_run(
        thread_id, body.resume, update=body.state_update
    )
    return {"thread_id": thread_id, "state": state, "interrupts": interrupts}
