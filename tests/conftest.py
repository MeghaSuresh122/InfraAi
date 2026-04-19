import os

import pytest

from infra_ai.graphs.main import reset_app_graph_cache


@pytest.fixture(autouse=True)
def _reset_graph_cache():
    reset_app_graph_cache()
    os.environ["INFRA_AI_MOCK_LLM"] = "1"
    yield
    reset_app_graph_cache()
