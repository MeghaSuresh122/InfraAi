# InfraAi

From requirements to ready-to-deploy infrastructure—validated, compliant, and automated: an intelligent workflow that interprets user requirements, applies validation and policy-style guardrails, and generates infrastructure configs with LangGraph-driven human-in-the-loop review.

## Quick start

```bash
pip install -e ".[dev]"
copy .env.example .env
# Set OPENROUTER_API_KEY for real codegen, or use mock mode:
set INFRA_AI_MOCK_LLM=1
infra-ai run --text "Build a frontend application in react" --mock-llm
```

Run the HTTP API:

```bash
infra-ai serve
# or: uvicorn infra_ai.api.main:app --reload
```

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — components, checkpointing, extension points  
- [docs/WORKFLOW.md](docs/WORKFLOW.md) — LangGraph nodes, HITL payloads, JSON shapes  
- [docs/CODE_AND_TOOLS.md](docs/CODE_AND_TOOLS.md) — env vars, layout, testing, adding artifact types  

## Stack

LangGraph, LangChain (Ollama + OpenRouter), FastAPI, Pydantic, GitPython; optional Milvus hook for skill retrieval.

Default codegen model: **OpenRouter** `minimax/minimax-m2.5:free`. Other agents default to local **Ollama** models (configurable).
