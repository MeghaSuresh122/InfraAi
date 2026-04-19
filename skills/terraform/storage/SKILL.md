# Skill: Terraform storage (S3 + optional KMS)

## Required fields
- `environment` (string)
- `region` (string)
- `bucket_name` (string): globally unique pattern
- `enable_versioning` (bool)
- `enable_kms` (bool)

## Guardrails
- Block public ACLs; default private.
