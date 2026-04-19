from infra_ai.validation.deterministic import validate_config_fields


def test_k8s_deployment_requires_image_not_latest():
    fields = {
        "environment": {"value": "dev", "agent_generated": False, "confidence_score": 9.9},
        "namespace": {"value": "app-dev", "agent_generated": True, "confidence_score": 7.0},
        "app_name": {"value": "web", "agent_generated": False, "confidence_score": 9.9},
        "image": {"value": "nginx:latest", "agent_generated": True, "confidence_score": 7.0},
        "replicas": {"value": 2, "agent_generated": True, "confidence_score": 7.0},
    }
    ok, errs = validate_config_fields(fields, "k8s_deployment")
    assert not ok
    assert any("latest" in e for e in errs)


def test_k8s_deployment_ok():
    fields = {
        "environment": {"value": "dev", "agent_generated": False, "confidence_score": 9.9},
        "namespace": {"value": "app-dev", "agent_generated": True, "confidence_score": 7.0},
        "app_name": {"value": "web", "agent_generated": False, "confidence_score": 9.9},
        "image": {"value": "nginx:1.25.3", "agent_generated": True, "confidence_score": 7.0},
        "replicas": {"value": 2, "agent_generated": True, "confidence_score": 7.0},
    }
    ok, errs = validate_config_fields(fields, "k8s_deployment")
    assert ok
    assert errs == []
