import re
from typing import Any

REQUIRED_BY_TYPE: dict[str, tuple[str, ...]] = {
    "terraform_eks_cluster": ("environment", "region", "cluster_name", "kubernetes_version"),
    "k8s_deployment": ("environment", "namespace", "app_name", "image", "replicas"),
    "terraform_storage": ("environment", "region", "bucket_name"),
}


def _unwrap(field: Any) -> Any:
    if isinstance(field, dict) and "value" in field:
        return field["value"]
    return field


def validate_config_fields(
    fields: dict[str, Any], artifact_type: str
) -> tuple[bool, list[str]]:
    """
    Deterministic validation on flattened or envelope-shaped fields.
    Returns (ok, list of error messages).
    """
    errs: list[str] = []
    flat: dict[str, Any] = {}
    for k, v in fields.items():
        flat[k] = _unwrap(v)

    required = REQUIRED_BY_TYPE.get(
        artifact_type, ("environment", "region", "service_name")
    )
    for key in required:
        if key not in flat or flat[key] in (None, ""):
            errs.append(f"Missing required field: {key}")

    image = flat.get("image")
    if isinstance(image, str) and image.endswith(":latest"):
        errs.append("Image tag 'latest' is not allowed")

    replicas = flat.get("replicas")
    if replicas is not None:
        try:
            r = int(replicas)
            if r < 1:
                errs.append("replicas must be >= 1")
        except (TypeError, ValueError):
            errs.append("replicas must be an integer")

    for port_key in ("container_port", "port", "service_port"):
        p = flat.get(port_key)
        if p is not None:
            try:
                pv = int(p)
                if not (1 <= pv <= 65535):
                    errs.append(f"{port_key} must be between 1 and 65535")
            except (TypeError, ValueError):
                errs.append(f"{port_key} must be an integer")

    name = flat.get("cluster_name") or flat.get("app_name") or flat.get("service_name")
    if isinstance(name, str) and name:
        if not re.match(r"^[a-z0-9][a-z0-9-]{1,62}$", name):
            errs.append("Name must be DNS-like lowercase alphanumerics and hyphens")

    if artifact_type == "terraform_eks_cluster":
        for k in ("node_desired_size",):
            if k not in flat or flat[k] in (None, ""):
                errs.append(f"Missing recommended EKS field: {k}")

    return (len(errs) == 0, errs)
