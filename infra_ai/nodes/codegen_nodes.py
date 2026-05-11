import json
import logging
import re
import shutil
import subprocess
import tempfile

from pathlib import Path
from typing import Any, Literal

from distro import name
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from langgraph.types import Command, interrupt

from infra_ai.llm.factory import get_chat_model
from infra_ai.nodes.llm_utils import extract_json_object, mock_llm_enabled
from infra_ai.nodes.tools import global_tools_loader
from infra_ai.services.git_service import GitService
from infra_ai.state import InfraGraphState

logger = logging.getLogger(__name__)

CODE_BLOCK = re.compile(r"```(?:\w+)?\s*\n([\s\S]*?)```", re.MULTILINE)

def _parse_generated_files(text: str) -> list[tuple[str, str]]:
    files: list[tuple[str, str]] = []
    extracted_json = extract_json_object(text)
    if extracted_json is not None and extracted_json.get("type", "") == "text" and "text" in extracted_json:
        text = extracted_json["text"]
    parts = re.split(r"^###\s+", text, flags=re.MULTILINE)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        lines = part.splitlines()
        path_line = lines[0].strip() if lines else ""
        body = "\n".join(lines[1:]) if len(lines) > 1 else part
        m = CODE_BLOCK.search(body)
        content = m.group(1).strip() if m else body.strip()
        if path_line and ("/" in path_line or path_line.endswith((".tf", ".yaml", ".yml"))):
            filename = path_line.split()[0].strip("'").strip("\"")
            files.append((filename, content))
    if not files:
        m = CODE_BLOCK.search(text)
        if m:
            files.append(("generated/main.tf", m.group(1).strip()))
    return files


def _repo_aware_system_messages(state: InfraGraphState) -> list[Any]:
    """Compact system guidance from analyzed repo (tree, summary, modules, folder context)."""
    rc = state.get("repo_context") or {}
    paths = rc.get("repo_tree_paths") or []
    if not paths:
        return []
    summary = rc.get("repo_summary") or {}
    mods = rc.get("reusable_modules") or {}
    head = (rc.get("repo_head_commit") or state.get("repo_head_commit") or "")[:16]
    top = summary.get("top_level_folders") or []
    if isinstance(top, list):
        top_str = ", ".join(str(x) for x in top[:25])
    else:
        top_str = str(top)
    mods_json = json.dumps(mods, ensure_ascii=False, default=str)[:4500]

    # repo_folder guidance
    repo_folder = (state.get("repo_folder") or "").strip()
    folder_exists: bool = rc.get("repo_folder_exists", False)
    folder_files: list[str] = rc.get("repo_folder_files") or []

    if repo_folder:
        if folder_exists:
            folder_file_list = "\n".join(f"  - {f}" for f in folder_files) or "  (empty folder)"
            folder_section = (
                f"\n## Target folder: '{repo_folder}' (EXISTS)\n"
                "Update or extend the files already present. "
                "Generate every file path starting with the full folder path.\n"
                f"Current files:\n{folder_file_list}\n"
            )
        else:
            folder_section = (
                f"\n## Target folder: '{repo_folder}' (DOES NOT EXIST — CREATE IT)\n"
                "Create this new folder. Base its internal structure on sibling folders "
                "visible in the repository layout above. "
                "Generate every file path starting with the full folder path.\n"
            )
    else:
        folder_section = (
            "\n## Target folder: not specified\n"
            "Choose a location that fits the existing repository layout.\n"
        )

    body = (
        "You are generating infrastructure changes for an EXISTING repository.\n"
        "Follow the repository layout and paths shown below; prefer updating listed files "
        "and reusing local Terraform/Kubernetes modules when applicable.\n\n"
        f"- Snapshot HEAD (short): {head}\n"
        f"- File count: {len(paths)}\n"
        f"- Top-level folders: {top_str}\n"
        f"- Reusable modules / patterns (JSON, truncated):\n{mods_json}\n"
        f"{folder_section}"
    )
    return [SystemMessage(content=body)]


def _codegen_system_messages(artifact: str) -> list[Any]:
    base_system_message = (
        "You have access to specialized MCP tools for getting knowledge on different platforms.\n"
        "For generating terraform code, you MUST make use of the appropriate Terraform MCP tools.\n"
        "Rules:\n"
        "- Call each tool at most once\n"
        "- After receiving tool results, you MUST produce final output\n"
        "- DO NOT call the same tool again\n"
        "- If tool results are already available, use them directly\n"
        "- Prefer completing the task over calling tools again"
        "When MCP tool results are available:\n"
        "- Use them to update the code\n"
        "- DO NOT call the same tool again\n"
        "- If sufficient information is available, produce final output WITHOUT tool calls"
    )
    
    if artifact.startswith("terraform_"):
        return [
            SystemMessage(
                content=(
                    base_system_message +
                    "You are a Terraform expert.\n\n"
                    "Use terraform-aws-modules/eks/aws best practices.\n"
                    "Follow this pattern:\n"
                    "- VPC module\n"
                    "- EKS module (no custom IAM)\n"
                    "- Outputs only from module\n\n"
                    "Follow these strict rules:\n"
                    "1. If using terraform-aws-modules/eks/aws:\n"
                    "   - DO NOT create IAM roles manually unless explicitly required\n"
                    "   - DO NOT create security groups unless needed\n"
                    "2. Separate infrastructure and Kubernetes resources:\n"
                    "   - No kubernetes_* resources in same apply as EKS creation\n"
                    "3. Ensure dependency correctness:\n"
                    "   - Providers must not depend on resources created in same plan\n"
                    "4. Prefer module defaults unless customization is required\n"
                    "5. Output only valid, deployable Terraform\n\n"
                    "6. Do not create duplicate terraform blocks across files\n"
                    "7. Every external module MUST include an explicit version.\n"
                    "8. Never generate module arguments not present in the module schema\n"
                    "9. If terraform version is present in existing repo README files use it. Or prefer existing module versions already used in repository\n"
                    "10. If module version unknown, inspect repository before generation\n"
                    "Before generating Terraform:\n"
                    "- Ensure Kubernetes version is currently supported by AWS EKS\n"
                    "- Use latest stable versions of Terraform modules unless specified\n"
                    "- Avoid deprecated provider/module versions\n"
                    "After generating Terraform:\n"
                    "1. Validate structure (files, blocks)\n"
                    "2. Validate AWS compatibility (versions, services)\n"
                    "3. Identify:\n"
                    "   - deprecated versions\n"
                    "   - duplicate blocks\n"
                    "   - missing best practices\n"
                    "4. Fix issues automatically\n"
                    "5. Output final corrected version\n"
                    "Before final output, verify:\n"
                    "- No hardcoded availability zones\n"
                    "- Cluster endpoint access is explicitly defined\n"
                    "- Cost-impacting resources (NAT Gateway) are intentional\n"
                    "- Tags are consistently applied\n"
                )
            )
        ]
    return [
        SystemMessage(
            content=(
                base_system_message +
                "You are a Kubernetes manifest expert.\n\n"
                "Generate valid Kubernetes YAML only.\n"
                "Do not include Terraform or EKS cluster creation resources.\n"
                "Use apiVersion/apps/v1 for Deployment and apiVersion/v1 for Service.\n"
                "Do not use :latest image tags.\n"
                "Keep resources focused on the application workload and common defaults.\n"
            )
        )
    ]


def _terraform_fmt(paths: list[Path]) -> None:
    tf_files = [p for p in paths if p.suffix == ".tf"]
    if not tf_files:
        return
    try:
        subprocess.run(
            ["terraform", "fmt", "-no-color", *[str(p) for p in tf_files]],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError:
        logger.info("terraform binary not found; skipping fmt")


def codegen_node(state: InfraGraphState) -> dict[str, Any]:
    logger.info("=== CODEGEN STAGE ===")
    item = state.get("current_config_item") or {}
    fields = state.get("config_fields_output") or {}
    artifact = item.get("type") or "k8s_deployment"
    logger.info("Generating code for artifact: %s", artifact)
    
    def _mock_codegen_text() -> str:
        if artifact == "k8s_deployment":
            return (
                "### k8s/deployment.yaml\n```yaml\n"
                "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n"
                "  name: app\nspec:\n  replicas: 1\n  selector: {}\n  template: {}\n```\n"
            )
        return (
            "### terraform/main.tf\n```hcl\n"
            'terraform {\n  required_version = ">= 1.5.0"\n}\n```\n'
        )

    rag_block = (state.get("repo_rag_context_text") or "").strip()
    plan_hint = ""
    cp = state.get("codegen_plan") or {}
    if cp:
        plan_hint = (
            "\n\n## codegen_plan (must align new/updated paths with this plan)\n"
            f"{json.dumps(cp, indent=2, default=str)[:6000]}\n"
        )

    base_user_prompt = (
        "Generate infrastructure files. Use markdown sections starting with "
        "'### <relative/path>' followed by a fenced code block.\n\n"
        f"Artifact type: {artifact}\n"
        f"Fields JSON:\n{json.dumps(fields, indent=2)}\n"
    )

    # repo_folder instruction in user prompt
    repo_folder = (state.get("repo_folder") or "").strip()
    rc = state.get("repo_context") or {}
    folder_exists: bool = rc.get("repo_folder_exists", False)
    if repo_folder:
        if folder_exists:
            base_user_prompt += (
                f"\nIMPORTANT: The target folder '{repo_folder}' already exists. "
                "All generated file paths MUST start with this folder path. "
                "Create new files or update existing ones within this folder.\n"
            )
        else:
            base_user_prompt += (
                f"\nIMPORTANT: The target folder '{repo_folder}' does NOT exist yet — create it. "
                "All generated file paths MUST start with this folder path. "
                "Model the folder structure on sibling folders in the repository.\n"
            )
    base_user_prompt += f"{plan_hint}"
    if rag_block:
        base_user_prompt += (
            "\n\n## Repository context (follow existing layout; reuse modules where noted)\n"
            f"{rag_block}\n"
        )

    if mock_llm_enabled():
        logger.debug("Using mock codegen")
        text = _mock_codegen_text()
        msg = AIMessage(content=text)
    else:
        try:
            logger.debug("Invoking LLM for code generation")
            llm = get_chat_model("codegen")

            tools = global_tools_loader.tools
            llm_with_tools = llm.bind_tools(tools)
            tool_names = [tool.name for tool in tools]
            if len(tool_names) > 0:
                mcp_tool_usage_prompt = (
                    "When using the `search_modules` tool:\n"
                    "- The `module_query` MUST be a simple keyword (e.g., \"eks\", \"vpc\", \"security-group\").\n"
                    "- DO NOT pass full module identifiers like \"namespace/name/provider\".\n"
                    "- If you need to use a full module identifier (e.g., \"terraform-aws-modules/eks/aws\"):\n"
                    "  → Extract only the module name (e.g., \"eks\") and use that as the query.\n"
                    "\n"
                    "After calling `search_modules`, select the best match where:"
                    "- namespace matches (if known)\n"
                    "- name matches\n"
                    "- provider matches\n\n"
                )
            else:
                mcp_tool_usage_prompt = ""

            tool_budget = max(0, 6 - int(state.get("tool_call_count") or 0))
            prompt = (
                base_user_prompt
                + f"\n\nAvailable tools: {', '.join(tool_names)}\n"
                + mcp_tool_usage_prompt
                + f"FOLLOW THIS STRICTLY: Number of MCP tool calls you can make is: {tool_budget}\n"
                "If number of tool calls you can make is zero or less, generate final output."
            )
            messages = (
                _codegen_system_messages(artifact)
                + _repo_aware_system_messages(state)
                + [HumanMessage(content=prompt)]
                + list(state.get("messages") or [])
            )
            msg = llm_with_tools.invoke(messages)
            logger.info("Code generation completed")
        except Exception as e:  # noqa: BLE001
            logger.exception("Codegen LLM failed for artifact: %s", artifact)
            error_payload = {
                "kind": "codegen_error",
                "message": f"Codegen failed for artifact '{artifact}': {e}",
                "error": str(e),
                "artifact": artifact,
                "fields": fields
            }
            edited = interrupt(error_payload)
            if edited and edited.get("retry"):
                logger.info("Retrying code generation after user review")
                msg = llm_with_tools.invoke(messages)
                logger.info("Code generation retry completed")
            else:
                raise RuntimeError(f"Codegen aborted: {e}")
    
    if hasattr(msg, "tool_calls"):
        for tc in msg.tool_calls:
            tool_call_log = {
                "tool": tc.get('name'),
                "args": tc.get('arguments') or tc.get('args')
            }
            if "tool_calls" not in state:
                state["tool_calls"] = []
            state["tool_calls"].append(tool_call_log)
            
    logger.info("==== TOOL CALLS ====")
    for c in state.get("tool_calls", []):
        logger.info(f"Tool: {c['tool']} | Args: {c['args']}")
    logger.info(f"Total tool calls: {len(state.get('tool_calls', []))}")

    return {
        "messages": [msg],
        "tool_calls": state.get("tool_calls", []),
        "tool_call_count": len(state.get("tool_calls", []))
    }

def git_push_node(state: InfraGraphState) -> dict[str, Any]:
    logger.info("=== CODE GEN OUTPUT PROCESSING ===")
    text = str(state.get("messages", [])[-1].content) if state.get("messages") else ""
    files = _parse_generated_files(text)

    if len(files) == 0:
        logger.error("No files generated from code generation agent")
        error_payload = {
            "kind": "Code generation output error",
            "message": f"Code generation output error: No files generated from code generation agent",
            "error": f"Code generation output error: No files generated from code generation agent",
            "generated_output": text,
            "tool_call_count": state.get("tool_call_count", 0),
            "tool_calls": state.get("tool_calls", []),
        }
        edited = interrupt(error_payload)
        if edited and edited.get("retry"):
            logger.info("Retrying code generation...")
            return {
                "generated_files": [],
            }
        else:
            logger.error("User did not retry code generation")
            raise RuntimeError(f"Code generation failed: No files generated")
    
    logger.info("Parsed %d files from codegen output", len(files))

    tmp = Path(tempfile.mkdtemp(prefix="infra-ai-codegen-"))
    written_fmt: list[tuple[str, str]] = []
    try:
        for rel, content in files:
            p = tmp / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            logger.debug("Generated file: %s", rel)
        
        tf_paths = [p for p in tmp.rglob("*.tf") if p.is_file()]
        if tf_paths:
            logger.info("Running terraform fmt on %d files", len(tf_paths))
            _terraform_fmt(tf_paths)
        
        for rel, _ in files:
            p = tmp / rel
            if p.is_file():
                written_fmt.append((rel, p.read_text(encoding="utf-8")))
        logger.info("Generated files ready for push: %d files", len(written_fmt))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    state_generated_files = [{"path": r, "content": c} for r, c in written_fmt]

    logger.info("=== GIT PUSH STAGE ===")
    item = state.get("current_config_item") or {}
    gfs = state_generated_files if state_generated_files is not None else []
    files = [(g["path"], g["content"]) for g in gfs]
    prefix = str(item.get("id") or "config").replace("/", "-")[:40]
    logger.info("Pushing %d files to git. Prefix: %s", len(files), prefix)
    
    svc = GitService(
        repo_url=state.get("repo_url"),
        default_branch=state.get("target_branch"),
        remote_name=state.get("git_remote_name"),
    )
    branch, messages = svc.push_files(files, branch_prefix=prefix)
    logger.info("Git push result. Branch: %s. Messages: %s", branch, messages)
    
    # Check if GitHub API push failed
    push_failed = any("GitHub API push failed" in msg for msg in messages)
    if push_failed:
        error_msg = next(msg for msg in messages if "GitHub API push failed" in msg)
        logger.error("GitHub API push failed: %s", error_msg)
        
        # Extract the actual error from the message
        error_payload = {
            "kind": "git_push_error",
            "message": f"GitHub API push failed. Local copy saved. You can retry after fixing the issue.",
            "error": error_msg,
            "repo_url": state.get("repo_url"),
            "target_branch": state.get("target_branch"),
            "branch": branch,
            "messages": messages,
        }
        edited = interrupt(error_payload)
        if edited and edited.get("retry"):
            logger.info("User requested retry for git push")
            # Retry push with same branch and files
            svc = GitService(
                repo_url=state.get("repo_url"),
                default_branch=state.get("target_branch"),
                remote_name=state.get("git_remote_name"),
            )
            branch, messages = svc.push_files(files, branch_prefix=prefix)
            logger.info("Git push retry completed. Branch: %s", branch)
            # Check again if retry succeeded
            if any("GitHub API push failed" in msg for msg in messages):
                logger.error("Git push retry still failed")
                raise RuntimeError("GitHub API push failed even after retry. Check credentials and network connectivity.")
        else:
            logger.error("User did not retry git push")
            raise RuntimeError(f"GitHub API push failed: {error_msg}")
    
    logger.info("Git push completed successfully")
    
    # Create PR if it's a remote GitHub repo
    pr_url = svc.create_pull_request(
        head_branch=branch,
        title=f"InfraAi: {prefix} configs",
        body=f"Generated infrastructure configs for {item.get('description', item.get('id', 'config'))}\n\nBranch: {branch}"
    )
    if pr_url:
        messages.append(f"PR created: {pr_url}")
        logger.info("PR created: %s", pr_url)
    
    return {
        "generated_files": [{"path": r, "content": c} for r, c in written_fmt],
        "last_git_branch": branch,
        "last_pr_url": pr_url,
        "events": [
            {"node": "codegen", "files": len(written_fmt)},
            {"node": "git_push", "messages": messages, "branch": branch, "pr_url": pr_url}
        ],
    }

def route_after_git_push(state: InfraGraphState) -> Literal["codegen", "human_continue"]:
    if len(state.get("generated_files", [])) > 0:
        return "human_continue"
    return "codegen"