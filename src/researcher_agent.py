import os
import json
import logging
import signal
import concurrent.futures
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
    non_readme = [f for f in files if not f.lower().startswith("readme")]
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
    cleaned = [
        line for line in lines
        if not line.strip().startswith("#")
        and (pkg := line.split("==")[0].split(">=")[0].split("<=")[0].strip()) not in seen
        and not seen.add(pkg)
    ]
    return "\n".join(cleaned[:max_lines])


# --------------------------------------------------------------------------- #
# FIX #3 + #1: Pre-fetch metadata AND all key file contents via GitHub API    #
# in parallel, before the sandbox even boots.                                  #
# --------------------------------------------------------------------------- #

def _github_precheck(repo_url: str):
    """
    Fetches GitHub metadata + key file contents via API (all in parallel).
    Returns one of:
      - A short-circuit result dict  (docs/list repos)
      - A prefetch dict with '_prefetched': True  (proceed to sandbox with data)
      - None  (API unavailable — fall through to sandbox-based scrape)
    """
    try:
        owner_repo = _owner_repo_from_url(repo_url)
        if not owner_repo:
            return None

        headers = _gh_headers()

        # FIX #1 (pre-sandbox): Fetch repo metadata and root file listing in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            meta_future = ex.submit(
                http_requests.get,
                f"https://api.github.com/repos/{owner_repo}",
                **{"headers": headers, "timeout": 6}
            )
            tree_future = ex.submit(
                http_requests.get,
                f"https://api.github.com/repos/{owner_repo}/contents",
                **{"headers": headers, "timeout": 6}
            )
            meta_resp = meta_future.result()
            tree_resp = tree_future.result()

        if meta_resp.status_code != 200 or tree_resp.status_code != 200:
            return None

        m      = meta_resp.json()
        name   = m.get("name", "")
        desc   = m.get("description", "") or ""
        topics = m.get("topics", [])

        root_items = tree_resp.json()
        if not isinstance(root_items, list):
            return None
        files = [f["name"] for f in root_items if isinstance(f, dict)]

        # Short-circuit for obvious docs/list repos — no sandbox needed
        if _is_docs_only(name, desc, topics, files):
            logger.info(f"[{repo_url}] Pre-check: docs/list repo — skipping sandbox.")
            return {
                "repo_url":     repo_url,
                "success":      True,
                "project_type": "Documentation/List",
                "reasoning":    f"Detected as documentation or list repo (name={name}, topics={topics}).",
                "output":       "This repository is a documentation or list repository. No executable application found.",
            }

        # FIX #1 + #3: Fetch all key files in parallel via GitHub API
        # This entirely replaces the serial per-file sandbox round-trips in scrape_repo_context
        key_filenames = {
            "requirements.txt", "package.json", "main.py",
            "app.py", "index.js", "README.md", "pyproject.toml"
        }
        files_to_fetch = [
            f["path"] for f in root_items
            if isinstance(f, dict)
            and f.get("name") in key_filenames
            and f.get("type") == "file"
        ]

        logger.info(f"[{repo_url}] Pre-fetching {len(files_to_fetch)} key files in parallel via GitHub API...")

        contents_parts = []
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

        # Build a simple file-tree string from the API root listing
        file_tree_lines = [
            f"/home/daytona/project/{f['name']}" + ("/" if f.get("type") == "dir" else "")
            for f in root_items if isinstance(f, dict)
        ]
        file_tree = "\n".join(file_tree_lines)

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
    """
    FIX #1: Reads ALL key files in a single bash round-trip using delimiters,
    replacing the original one-_exec-per-file loop.
    """
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
    matched_files = [
        line.strip() for line in file_tree.split("\n")
        if any(line.strip().endswith(name) for name in key_filenames)
    ]

    if not matched_files:
        return file_tree, ""

    # Single round-trip: print a unique delimiter before each file then cat it
    DELIM = "===CATFILE==="
    bash_parts = [
        f"echo '{DELIM}{path}' && cat '{path}' 2>/dev/null || true"
        for path in matched_files
    ]
    raw_output = _exec(sandbox, f"bash -c \"{' && '.join(bash_parts)}\"")

    # Parse the delimited output
    contents_parts = []
    current_path   = None
    current_lines  = []

    for line in raw_output.split("\n"):
        if line.startswith(DELIM):
            if current_path and current_lines:
                content = "\n".join(current_lines).strip()
                rel     = current_path.replace(workspace_root + "/", "")
                body    = clean_requirements(content) if "requirements" in rel else content[:1500]
                contents_parts.append(f"### {rel}\n{body}")
            current_path  = line[len(DELIM):]
            current_lines = []
        else:
            current_lines.append(line)

    # Flush last file
    if current_path and current_lines:
        content = "\n".join(current_lines).strip()
        rel     = current_path.replace(workspace_root + "/", "")
        body    = clean_requirements(content) if "requirements" in rel else content[:1500]
        contents_parts.append(f"### {rel}\n{body}")

    return file_tree, "\n\n".join(contents_parts)


def _validate_install_commands(commands: List[str], file_tree: str) -> List[str]:
    """
    FIX #2: Strip out broken uv install commands that carry no package argument.
    The LLM sometimes emits 'uv pip install --system' with nothing after it
    when there's no requirements.txt — this caused silent 90s timeouts.
    """
    has_requirements = "requirements.txt" in file_tree
    cleaned = []
    for cmd in commands:
        s = cmd.strip()
        if not s:
            continue

        # Bare 'uv pip install --system' with no target
        if s == "uv pip install --system":
            logger.warning("Dropping bare 'uv pip install --system' — no package argument.")
            continue

        # uv requirements install when file doesn't exist
        if "uv pip install" in s and "-r requirements.txt" in s and not has_requirements:
            logger.warning("Dropping uv -r requirements.txt — requirements.txt not in tree.")
            continue

        # uv install that ends with only flags (no package/file target)
        if s.startswith("uv pip install") and s.endswith(("--system", "--no-build-isolation", "--system --no-build-isolation")):
            logger.warning(f"Dropping incomplete uv command: {s}")
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
    """
    Performance improvements vs original:

    FIX #1 — Batch file reads: all key file cat calls combined into ONE
              bash round-trip (was N serial _exec calls).
    FIX #2 — Guard broken uv installs: strip empty 'uv pip install --system'
              commands that caused silent 90s timeouts.
    FIX #3 — Pre-fetch via GitHub API: key file contents fetched in parallel
              before sandbox boots, using GitHub REST API.
    FIX #4 — Parallel sandbox + LLM: LLM plan generation runs concurrently
              with daytona.create(), hiding sandbox boot latency.
    FIX #5 — Faster install: 60s timeout (was 90s) + --no-build-isolation.
    """

    # Pre-check: fetches metadata + key file contents via GitHub API (all parallel)
    precheck = _github_precheck(repo_link)

    # Short-circuit: docs/list repo detected
    if precheck is not None and not precheck.get("_prefetched"):
        return precheck

    prefetched     = precheck is not None and precheck.get("_prefetched", False)
    api_file_tree  = precheck.get("file_tree", "")     if prefetched else ""
    api_file_conts = precheck.get("file_contents", "") if prefetched else ""

    has_sigalrm    = hasattr(signal, "SIGALRM")
    workspace_root = "/home/daytona/project"

    # ── FIX #4: If we have pre-fetched context, run LLM plan WHILE sandbox boots ──
    if prefetched and api_file_tree:
        logger.info(f"[{repo_link}] Starting sandbox + LLM plan in parallel...")

        plan_result    = [None]
        plan_exception = [None]

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
        # Clone (shallow)
        logger.info(f"[{repo_link}] Cloning (shallow)...")
        _exec(sandbox, f"git clone --depth 1 {repo_link} {workspace_root}")

        # Use pre-computed plan if available, otherwise scrape + call LLM now
        if plan_prefetched is not None:
            plan = plan_prefetched
            file_tree_for_validation = api_file_tree
            logger.info(f"[{repo_link}] Using pre-computed plan — skipping sandbox scrape.")
        else:
            # FIX #1: single-round-trip batched scrape
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

        # FIX #2: Strip broken install commands before running
        install_cmds = _validate_install_commands(plan.install_commands, file_tree_for_validation)

        if install_cmds:
            logger.info(f"[{repo_link}] Installing ({len(install_cmds)} step(s) combined)...")
            combined = " && ".join(install_cmds)
            # FIX #5: 60s timeout (was 90s); --no-build-isolation already in LLM prompt
            out = _exec(
                sandbox,
                f"export DEBIAN_FRONTEND=noninteractive PIP_NO_INPUT=1; "
                f"timeout 60s bash -c '{combined}' 2>&1",
                cwd=run_cwd,
            )
            logger.info(f"[{repo_link}] Install tail: {out[-300:]}")
        else:
            logger.info(f"[{repo_link}] No valid install commands — skipping install.")

        # Run (15s hard limit)
        logger.info(f"[{repo_link}] Running project (15s timeout)...")
        final_output = _exec(
            sandbox,
            f"timeout 15s bash -c '{plan.run_command} 2>&1'",
            cwd=run_cwd,
        )

        error_markers = [
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