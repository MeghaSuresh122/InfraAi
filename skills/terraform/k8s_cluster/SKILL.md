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
- Avoid hardcoded AZ names like `${var.region}a` / `${var.region}b`; use `data "aws_availability_zones" "available" {}` and slice the returned names.
- Prefer private-only cluster API access for security; if public access is enabled, restrict `cluster_endpoint_public_access_cidrs`.
- When using the `kubernetes` provider for post-cluster resources, configure `exec` auth and be aware the cluster must exist before Kubernetes operations.
- For dev clusters, treat `enable_nat_gateway = true` and `single_nat_gateway = true` as an optional cost tradeoff.
- Output names should match the returned value shape: do not name a map/list output `node_role_arn`.
