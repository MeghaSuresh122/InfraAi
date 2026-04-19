# Skill: Kubernetes Deployment

## Purpose
Fields for a standard Deployment + Service manifest.

## Required fields
- `environment` (string)
- `namespace` (string): default `default` only for dev; otherwise env-specific
- `app_name` (string)
- `image` (string): must include explicit tag, not `latest`
- `replicas` (integer, >=1)
- `container_port` (integer)
- `cpu_request`, `cpu_limit`, `memory_request`, `memory_limit` (strings, k8s quantities)
- `labels` (map string->string)

## Guardrails
- Ban image tag `latest`.
- Set probes only if ports are defined.
