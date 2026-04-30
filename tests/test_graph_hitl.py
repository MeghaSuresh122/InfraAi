import pytest

from infra_ai.schemas.config_plan import ConfigPlan, ConfigPlanItem
import infra_ai.nodes.analysis_nodes as wf
from infra_ai.runner import invoke_until_interrupt, resume_run


@pytest.fixture
def single_item_plan(monkeypatch):
    def _one_item(_req):
        return ConfigPlan(
            items=[
                ConfigPlanItem(
                    id="k1",
                    description="Kubernetes Deployment",
                    type="k8s_deployment",
                    environment="dev",
                ),
            ]
        )

    monkeypatch.setattr(wf, "_heuristic_plan", _one_item)


def test_interrupt_review_then_finish(single_item_plan):
    tid, state, interrupts = invoke_until_interrupt(
        {"raw_user_text": "Build react frontend", "raw_user_configs": {}},
    )
    assert interrupts
    assert interrupts[0]["value"]["kind"] == "review_fields"

    fields = state["config_fields_output"]
    state2, ints2 = resume_run(tid, {"config_fields": fields})
    assert ints2
    assert ints2[0]["value"]["kind"] == "confirm_repo"

    state3, ints3 = resume_run(
        tid,
        {"confirm": True, "repo_url": "", "target_branch": "main"},
    )
    assert state3.get("generated_files")
    assert ints3 and ints3[0]["value"]["kind"] == "continue_next"

    state4, ints4 = resume_run(tid, {"continue_next": False})
    assert state4.get("workflow_status") == "completed"
    assert not ints4
