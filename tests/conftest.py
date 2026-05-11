import os

import pytest

from infra_ai.config import get_settings
from infra_ai.graphs.main import reset_app_graph_cache


@pytest.fixture(autouse=True)
def _reset_graph_cache():
    get_settings.cache_clear()
    reset_app_graph_cache()
    os.environ["INFRA_AI_MOCK_LLM"] = "1"
    os.environ["INFRA_AI_SKIP_MCP"] = "1"
    yield
    reset_app_graph_cache()
    os.environ.pop("INFRA_AI_SKIP_MCP", None)
    get_settings.cache_clear()
