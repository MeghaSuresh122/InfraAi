from infra_ai.nodes.codegen_nodes import _codegen_system_messages


def test_codegen_system_prompt_for_terraform() -> None:
    messages = _codegen_system_messages("terraform_eks_cluster")
    assert any("Terraform expert" in getattr(m, "content", "") for m in messages)
    assert any("terraform-aws-modules/eks/aws" in getattr(m, "content", "") for m in messages)
    assert any("No kubernetes_* resources in same apply" in getattr(m, "content", "") for m in messages)


def test_codegen_system_prompt_for_k8s_deployment() -> None:
    messages = _codegen_system_messages("k8s_deployment")
    assert any("Kubernetes manifest expert" in getattr(m, "content", "") for m in messages)
    assert any("Do not include Terraform or EKS cluster creation resources" in getattr(m, "content", "") for m in messages)
