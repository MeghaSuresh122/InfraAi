"""Future: load validator plugins from entry points or infra_ai.validation.plugins package."""

from collections.abc import Callable
from typing import Any

ValidatorFn = Callable[[dict[str, Any], str], tuple[bool, list[str]]]

_REGISTRY: list[ValidatorFn] = []


def register(fn: ValidatorFn) -> ValidatorFn:
    _REGISTRY.append(fn)
    return fn


def run_plugins(fields: dict[str, Any], artifact_type: str) -> tuple[bool, list[str]]:
    errs: list[str] = []
    for fn in _REGISTRY:
        ok, e = fn(fields, artifact_type)
        if not ok:
            errs.extend(e)
    return (len(errs) == 0, errs)
