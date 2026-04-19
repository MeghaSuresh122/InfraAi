# InfraAi

From requirements to ready-to-deploy infrastructure—validated, compliant, and automated: an intelligent workflow that interprets user requirements, applies validation and policy-style guardrails, and generates infrastructure configs with LangGraph-driven human-in-the-loop review.

## Prerequisites

- Python 3.11 or newer (see `requires-python` in `pyproject.toml`)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed (recommended package manager for this repo)

## Virtual environment and install (uv)

From the repository root:

**Windows (PowerShell)**

```powershell
uv venv
.\.venv\Scripts\Activate.ps1
uv sync --extra dev
```

**macOS / Linux**

```bash
uv venv
source .venv/bin/activate
uv sync --extra dev
```

- `uv venv` creates a `.venv` in the project directory (gitignored).
- `uv sync --extra dev` installs the project in editable mode, resolves dependencies from `uv.lock`, and includes the `dev` optional extra (pytest, ruff, etc.).
- Optional Milvus extras: `uv sync --extra dev --extra milvus`

After changing dependencies in `pyproject.toml`, refresh the lockfile and reinstall:

```bash
uv lock
uv sync --extra dev
```

Commit `uv.lock` with dependency changes so installs are reproducible.

`pyproject.toml` sets `[tool.uv] link-mode = "copy"` to reduce hardlink issues on cloud-synced folders (for example OneDrive on Windows). If installs look corrupted (missing imports despite `uv pip list`), delete the `.venv` folder and run `uv sync --extra dev` again.

## Quick start

```powershell
copy .env.example .env
# Set OPENROUTER_API_KEY for real codegen, or use mock mode:
$env:INFRA_AI_MOCK_LLM = "1"
uv run infra-ai run --text "Build a frontend application in react" --mock-llm
```

(`uv run` uses the project environment without activating the venv.)

Run the HTTP API:

```powershell
uv run infra-ai serve
# or: uv run uvicorn infra_ai.api.main:app --reload
```

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — components, checkpointing, extension points  
- [docs/WORKFLOW.md](docs/WORKFLOW.md) — LangGraph nodes, HITL payloads, JSON shapes  
- [docs/CODE_AND_TOOLS.md](docs/CODE_AND_TOOLS.md) — env vars, layout, testing, adding artifact types  

## Stack

LangGraph, LangChain (Ollama + OpenRouter), FastAPI, Pydantic, GitPython; optional Milvus hook for skill retrieval.

Default codegen model: **OpenRouter** `minimax/minimax-m2.5:free`. Other agents default to local **Ollama** models (configurable).
