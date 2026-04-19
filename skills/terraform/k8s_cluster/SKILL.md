# Skill: Terraform EKS cluster

## Purpose
Produce variables for an EKS cluster module: VPC integration, node groups, and cluster identity.

## Required fields
- `environment` (string): dev | test | prod
- `region` (string): AWS region, e.g. `us-east-1`
- `cluster_name` (string): unique per env/account
- `kubernetes_version` (string): supported EKS version, not `latest`
- `node_instance_types` (list of strings)
- `node_desired_size`, `node_min_size`, `node_max_size` (integers)
- `vpc_cidr` (string, optional with default `10.0.0.0/16`)

## Naming
- Prefix with `environment` and application name from requirements when available.

## Guardrails
- Never use `:latest` container tags in examples; pin versions in separate deployment skills.
- Prefer smaller node groups for non-prod.
