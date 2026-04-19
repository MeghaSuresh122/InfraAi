from pathlib import Path

from infra_ai.config import get_settings

TYPE_TO_SKILL: dict[str, str] = {
    "terraform_eks_cluster": "terraform/k8s_cluster/SKILL.md",
    "k8s_deployment": "k8s/deployment/SKILL.md",
    "terraform_storage": "terraform/storage/SKILL.md",
}


def _skills_root() -> Path:
    s = get_settings()
    root = Path(s.skills_dir)
    if not root.is_absolute():
        root = Path.cwd() / root
    return root


def load_skill_markdown(artifact_type: str) -> str:
    """Load SKILL.md for an artifact type; falls back to generic stub."""
    rel = TYPE_TO_SKILL.get(artifact_type)
    root = _skills_root()
    if rel:
        path = root / rel
        if path.is_file():
            return path.read_text(encoding="utf-8")
    generic = root / "_generic" / "SKILL.md"
    if generic.is_file():
        return generic.read_text(encoding="utf-8")
    return (
        f"# Skill not found for type {artifact_type}\n"
        "Define required fields: env, region, service_name, replicas, image, "
        "resources (cpu/memory), ports, labels."
    )
