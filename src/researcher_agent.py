import os
import json
import logging
import signal
import concurrent.futures
import base64
import requests as http_requests
from dotenv import load_dotenv
from daytona import Daytona, DaytonaConfig
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional

logger = logging.getLogger(__name__)

# --- Setup ---
load_dotenv()
GROQ_API_KEY    = os.environ.get("GROQ_API_KEY", "")
DAYTONA_API_KEY = os.environ.get("DAYTONA_API_KEY", "")

if not DAYTONA_API_KEY:
    raise RuntimeError("DAYTONA_API_KEY is not set.")

config  = DaytonaConfig(api_key=DAYTONA_API_KEY)
daytona = Daytona(config)
llm     = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile", temperature=0)

# --- Schemas & Prompts ---
class RepoExecutionPlan(BaseModel):
    is_runnable:       bool       = Field(description="False if repo is docs/list/book with no runnable app.")
    project_type:      str        = Field(description="Detected project type")
    install_commands:  List[str]  = Field(description="Ordered shell commands to install dependencies")
    entry_point:       str        = Field(description="Relative path of the file to execute")
    run_command:       str        = Field(description="Full shell command to run the project")
    working_directory: str        = Field(description="Directory from which to run the project")
    reasoning:         str        = Field(description="Brief explanation of choices")

parser = JsonOutputParser(pydantic_object=RepoExecutionPlan)
plan_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert software engineer. Analyze the repo and produce an execution plan.

CRITICAL INSTRUCTIONS:
1. If the repo is an Awesome List, book, tutorial collection, or pure documentation, set is_runnable=false.
2. For Python projects with a requirements.txt PRESENT IN THE FILE TREE:
   - Use 'pip install uv' first
   - Then 'uv pip install --system --no-build-isolation -r requirements.txt'
   - ONLY emit the uv install command if requirements.txt actually exists. NEVER emit a bare
     'uv pip install --system' with no arguments — this is an error.
3. For Python projects with pyproject.toml but NO requirements.txt, use 'pip install .' instead.
4. For Python projects with neither, use 'pip install <specific_package_name>' if needed.
5. Install only the bare minimum required to run the entry point.
6. Prefer the fewest install commands possible.

{format_instructions}"""),
    ("human", "FILE TREE:\n{file_tree}\n\nKEY FILE CONTENTS:\n{file_contents}\n\nProduce JSON plan.")
])
plan_chain = plan_prompt | llm | parser


# --------------------------------------------------------------------------- #
# GitHub API helpers                                                           #
# --------------------------------------------------------------------------- #

def _gh_headers() -> dict:
    headers = {"Accept": "application/vnd.github.v3+json"}
    gh_token = os.environ.get("GITHUB_TOKEN", "")
    if gh_token:
        headers["Authorization"] = f"Bearer {gh_token}"
    return headers


def _owner_repo_from_url(repo_url: str) -> Optional[str]:
    clean = repo_url.rstrip("/").removesuffix(".git")
    parts = clean.split("github.com/")
    return parts[1] if len(parts) >= 2 else None


def _is_docs_only(name: str, description: str, topics: list, files: list) -> bool:
    """Only skip sandbox for repos we are 100% certain are non-runnable."""
    if "awesome-list" in topics:
        return True
    name_lower = name.lower()
    if name_lower.startswith("awesome-") or name_lower == "awesome":
        return True
    non_readme =[f for f in files if not f.lower().startswith("readme")]
    if len(files) > 0 and len(non_readme) == 0:
        return True
    return False


def _fetch_file_via_api(owner_repo: str, path: str, headers: dict) -> Optional[str]:
    """Fetch a single file's decoded content from the GitHub contents API."""
    try:
        resp = http_requests.get(
            f"https://api.github.com/repos/{owner_repo}/contents/{path}",
            headers=headers, timeout=6
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        if data.get("encoding") == "base64":
            import base64
            return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        return None
    except Exception:
        return None


def clean_requirements(content: str, max_lines: int = 40) -> str:
    lines = content.splitlines()
    seen  = set()
    cleaned =[
        line for line in lines
        if not line.strip().startswith("#")
        and (pkg := line.split("==")[0].split(">=")[0].split("<=")[0].strip()) not in seen
        and not seen.add(pkg)
    ]
    return "\n".join(cleaned[:max_lines])


# --------------------------------------------------------------------------- #
# Pre-fetch metadata AND all key file contents via GitHub API                 #
# --------------------------------------------------------------------------- #

def _github_precheck(repo_url: str):
    """
    Fetches GitHub metadata + recursive file tree via API (all in parallel).
    """
    try:
        owner_repo = _owner_repo_from_url(repo_url)
        if not owner_repo:
            return None

        headers = _gh_headers()

        meta_resp = http_requests.get(
            f"https://api.github.com/repos/{owner_repo}",
            headers=headers, timeout=6
        )

        if meta_resp.status_code != 200:
            return None

        m = meta_resp.json()
        name   = m.get("name", "")
        desc   = m.get("description", "") or ""
        topics = m.get("topics",[])
        default_branch = m.get("default_branch", "main")

        # Fetch the FULL RECURSIVE file tree so subdirectories aren't ignored
        tree_resp = http_requests.get(
            f"https://api.github.com/repos/{owner_repo}/git/trees/{default_branch}?recursive=1",
            headers=headers, timeout=6
        )

        if tree_resp.status_code != 200:
            return None

        tree_data = tree_resp.json()
        root_items = tree_data.get("tree",[])

        # Filter to files only
        files = [f["path"] for f in root_items if f.get("type") == "blob"]

        # Short-circuit for obvious docs/list repos
        if _is_docs_only(name, desc, topics, files):
            logger.info(f"[{repo_url}] Pre-check: docs/list repo — skipping sandbox.")
            return {
                "repo_url":     repo_url,
                "success":      True,
                "project_type": "Documentation/List",
                "reasoning":    f"Detected as documentation or list repo (name={name}, topics={topics}).",
                "output":       "This repository is a documentation or list repository. No executable application found.",
            }

        key_filenames = {
            "requirements.txt", "package.json", "main.py",
            "app.py", "index.js", "README.md", "pyproject.toml"
        }
        
        ignore_dirs = {"node_modules", ".git", "__pycache__", "venv", ".venv"}
        ignore_exts = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".pdf", ".mp4"}

        # Extract only relevant key files while avoiding massive bloat directories
        files_to_fetch = [
            f["path"] for f in root_items
            if f.get("type") == "blob"
            and f["path"].split("/")[-1] in key_filenames
            and not any(ignored in f["path"].split("/") for ignored in ignore_dirs)
        ]

        files_to_fetch = files_to_fetch[:15]  # Safety cap

        logger.info(f"[{repo_url}] Pre-fetching {len(files_to_fetch)} key files in parallel via GitHub API...")

        contents_parts =[]
        if files_to_fetch:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(files_to_fetch)) as ex:
                fetch_futures = {
                    ex.submit(_fetch_file_via_api, owner_repo, path, headers): path
                    for path in files_to_fetch
                }
                for fut in concurrent.futures.as_completed(fetch_futures):
                    path    = fetch_futures[fut]
                    content = fut.result()
                    if content:
                        body = clean_requirements(content) if "requirements" in path else content[:1500]
                        contents_parts.append(f"### {path}\n{body}")

        # Construct file tree string properly nested up to 400 paths
        file_tree_lines = [
            f"/home/daytona/project/{f['path']}" + ("/" if f.get("type") == "tree" else "")
            for f in root_items
            if not any(ignored in f["path"].split("/") for ignored in ignore_dirs)
            and not any(f["path"].lower().endswith(ext) for ext in ignore_exts)
        ]
        
        file_tree = "\n".join(file_tree_lines[:400]) 

        logger.info(f"[{repo_url}] Pre-check complete — proceeding with sandbox.")
        return {
            "_prefetched":   True,
            "owner_repo":    owner_repo,
            "file_tree":     file_tree,
            "file_contents": "\n\n".join(contents_parts),
            "files":         files,
        }

    except Exception as e:
        logger.warning(f"[{repo_url}] Pre-check error ({e}) — falling through to sandbox scrape.")
        return None


# --------------------------------------------------------------------------- #
# Sandbox helpers                                                              #
# --------------------------------------------------------------------------- #

def _exec(sandbox, cmd: str, cwd=None) -> str:
    res    = sandbox.process.exec(cmd, cwd=cwd)
    output = getattr(res, "result", None) or getattr(res, "stdout", "") or ""
    return output.strip()


def scrape_repo_context(sandbox, workspace_root: str):
    file_tree = _exec(
        sandbox,
        f"find {workspace_root} -maxdepth 3 "
        f"-not -path '*/.git/*' "
        f"-not -path '*/node_modules/*' "
        f"-not -path '*/__pycache__/*'"
    )

    key_filenames = {
        "requirements.txt", "package.json", "main.py",
        "app.py", "index.js", "README.md", "pyproject.toml"
    }
    matched_files =[
        line.strip() for line in file_tree.split("\n")
        if any(line.strip().endswith(name) for name in key_filenames)
    ]

    if not matched_files:
        return file_tree, ""

    DELIM = "===CATFILE==="
    bash_parts =[
        f"echo '{DELIM}{path}' && cat '{path}' 2>/dev/null || true"
        for path in matched_files
    ]
    raw_output = _exec(sandbox, f"bash -c \"{' && '.join(bash_parts)}\"")

    contents_parts =[]
    current_path   = None
    current_lines  =[]

    for line in raw_output.split("\n"):
        if line.startswith(DELIM):
            if current_path and current_lines:
                content = "\n".join(current_lines).strip()
                rel     = current_path.replace(workspace_root + "/", "")
                body    = clean_requirements(content) if "requirements" in rel else content[:1500]
                contents_parts.append(f"### {rel}\n{body}")
            current_path  = line[len(DELIM):]
            current_lines =[]
        else:
            current_lines.append(line)

    if current_path and current_lines:
        content = "\n".join(current_lines).strip()
        rel     = current_path.replace(workspace_root + "/", "")
        body    = clean_requirements(content) if "requirements" in rel else content[:1500]
        contents_parts.append(f"### {rel}\n{body}")

    return file_tree, "\n\n".join(contents_parts)


def _validate_install_commands(commands: List[str], file_tree: str) -> List[str]:
    has_requirements = "requirements.txt" in file_tree
    cleaned =[]
    for cmd in commands:
        s = cmd.strip()
        if not s:
            continue
        if s == "uv pip install --system":
            continue
        if "uv pip install" in s and "-r requirements.txt" in s and not has_requirements:
            continue
        if s.startswith("uv pip install") and s.endswith(("--system", "--no-build-isolation", "--system --no-build-isolation")):
            continue

        cleaned.append(cmd)
    return cleaned


class TimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutException("Operation timed out")


# --------------------------------------------------------------------------- #
# Main entry point                                                              #
# --------------------------------------------------------------------------- #

def execute_repo(repo_link: str) -> dict:
    precheck = _github_precheck(repo_link)

    if precheck is not None and not precheck.get("_prefetched"):
        return precheck

    prefetched     = precheck is not None and precheck.get("_prefetched", False)
    api_file_tree  = precheck.get("file_tree", "")     if prefetched else ""
    api_file_conts = precheck.get("file_contents", "") if prefetched else ""

    has_sigalrm    = hasattr(signal, "SIGALRM")
    workspace_root = "/home/daytona/project"

    if prefetched and api_file_tree:
        logger.info(f"[{repo_link}] Starting sandbox + LLM plan in parallel...")

        plan_result    = [None]
        plan_exception =[None]

        def _run_plan():
            try:
                raw = plan_chain.invoke({
                    "file_tree":           api_file_tree,
                    "file_contents":       api_file_conts,
                    "format_instructions": parser.get_format_instructions(),
                })
                plan_result[0] = RepoExecutionPlan(**raw)
            except Exception as e:
                plan_exception[0] = e

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            sandbox_future = ex.submit(daytona.create)
            plan_future    = ex.submit(_run_plan)
            concurrent.futures.wait([sandbox_future, plan_future])

        sandbox = sandbox_future.result()

        if plan_exception[0]:
            logger.warning(f"[{repo_link}] Parallel LLM plan failed: {plan_exception[0]} — will re-run after clone.")
            plan_prefetched = None
        else:
            plan_prefetched = plan_result[0]
            logger.info(f"[{repo_link}] LLM plan ready (computed during sandbox boot).")
    else:
        logger.info(f"[{repo_link}] Creating Daytona sandbox...")
        sandbox         = daytona.create()
        plan_prefetched = None

    try:
        # Wrapped clone in explicit bash shell wrapper for robust command arguments parsing 
        logger.info(f"[{repo_link}] Cloning (shallow)...")
        _exec(sandbox, f"bash -c \"env GIT_TERMINAL_PROMPT=0 git clone -c core.autocrlf=false --depth 1 {repo_link} {workspace_root} < /dev/null\"")

        if plan_prefetched is not None:
            plan = plan_prefetched
            file_tree_for_validation = api_file_tree
            logger.info(f"[{repo_link}] Using pre-computed plan — skipping sandbox scrape.")
        else:
            logger.info(f"[{repo_link}] Scraping repo (batched single round-trip)...")
            file_tree, file_contents = scrape_repo_context(sandbox, workspace_root)
            file_tree_for_validation = file_tree

            logger.info(f"[{repo_link}] Generating execution plan via LLM...")
            if has_sigalrm:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(60)
            try:
                plan_raw = plan_chain.invoke({
                    "file_tree":           file_tree,
                    "file_contents":       file_contents,
                    "format_instructions": parser.get_format_instructions(),
                })
                plan = RepoExecutionPlan(**plan_raw)
            except TimeoutException as e:
                return {
                    "repo_url": repo_link, "success": False,
                    "project_type": "Unknown", "reasoning": "LLM planning timed out",
                    "output": str(e),
                }
            finally:
                if has_sigalrm:
                    signal.alarm(0)

        if not plan.is_runnable:
            logger.info(f"[{repo_link}] LLM says not runnable — skipping execution.")
            return {
                "repo_url":     repo_link,
                "success":      True,
                "project_type": plan.project_type,
                "reasoning":    plan.reasoning,
                "output":       "This repository is a documentation or list repository. No executable application found.",
            }

        run_cwd = (
            os.path.join(workspace_root, plan.working_directory)
            if plan.working_directory not in ("", ".")
            else workspace_root
        )

        install_cmds = _validate_install_commands(plan.install_commands, file_tree_for_validation)

        if install_cmds:
            logger.info(f"[{repo_link}] Installing ({len(install_cmds)} step(s) combined)...")
            combined = " && ".join(install_cmds)
            
            # Pipe base64 straight into bash; avoids quotes issues and forces EOF to kill interactive stdin prompts
            b64_cmd = base64.b64encode(combined.encode('utf-8')).decode('utf-8')
            
            out = _exec(
                sandbox,
                f"export DEBIAN_FRONTEND=noninteractive PIP_NO_INPUT=1; "
                f"echo {b64_cmd} | base64 -d | timeout 60s bash 2>&1",
                cwd=run_cwd,
            )
            logger.info(f"[{repo_link}] Install tail: {out[-300:]}")
        else:
            logger.info(f"[{repo_link}] No valid install commands — skipping install.")

        # Run (15s hard limit)
        if plan.run_command:
            logger.info(f"[{repo_link}] Running project (15s timeout)...")
            b64_run = base64.b64encode(plan.run_command.encode('utf-8')).decode('utf-8')
            
            final_output = _exec(
                sandbox,
                f"echo {b64_run} | base64 -d | timeout 15s bash 2>&1",
                cwd=run_cwd,
            )
        else:
            final_output = "(no run command generated)"

        error_markers =[
            "Traceback", "ModuleNotFoundError", "ImportError",
            "SyntaxError", "Error:", "Exception:", "FAILED", "fatal:",
        ]
        success = bool(final_output) and not any(e in final_output for e in error_markers)

        return {
            "repo_url":     repo_link,
            "success":      success,
            "project_type": plan.project_type,
            "reasoning":    plan.reasoning,
            "output":       final_output[-2000:] if final_output else "(no output captured)",
        }

    except Exception as e:
        logger.error(f"[{repo_link}] Failed: {e}")
        return {
            "repo_url": repo_link, "success": False,
            "project_type": "Unknown", "reasoning": "Exception occurred",
            "output": str(e),
        }

    finally:
        try:
            daytona.delete(sandbox)
            logger.info(f"[{repo_link}] Sandbox deleted.")
        except Exception as ex:
            logger.warning(f"[{repo_link}] Could not delete sandbox: {ex}")