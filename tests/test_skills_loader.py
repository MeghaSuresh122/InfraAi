from infra_ai.skills.loader import load_skill_markdown


def test_load_eks_skill_contains_required():
    text = load_skill_markdown("terraform_eks_cluster")
    assert "EKS" in text or "eks" in text.lower()
    assert "environment" in text.lower()
    assert "aws_availability_zones" in text
    assert "kubernetes" in text.lower()
