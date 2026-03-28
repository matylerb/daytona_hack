import os
import logging
from daytona import Daytona, DaytonaConfig
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

logger = logging.getLogger(__name__)

# --- Setup LLM & Daytona ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
DAYTONA_API_KEY = os.environ.get("DAYTONA_API_KEY", "")

config = DaytonaConfig(api_key=DAYTONA_API_KEY)
daytona = Daytona(config)
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile", temperature=0)

# --- Schemas & Prompts ---
class RepoExecutionPlan(BaseModel):
    is_runnable: bool = Field(description="Set to false if the repo is mostly markdown lists, books, or docs without a main app to run.")
    project_type: str = Field(description="The detected project type")
    install_commands: List[str] = Field(description="Ordered list of shell commands to install dependencies")
    entry_point: str = Field(description="The exact relative path of the file to execute")
    run_command: str = Field(description="The full shell command to run the project")
    working_directory: str = Field(description="The directory from which to run the project")
    reasoning: str = Field(description="Brief explanation of choices")

parser = JsonOutputParser(pydantic_object=RepoExecutionPlan)
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert software engineer. Analyze the repo and produce an execution plan.

CRITICAL INSTRUCTIONS:
1. IF the repository is an "Awesome List", a book, a tutorial collection, or just documentation, set `is_runnable` to false, leave commands empty, and explain in reasoning.
2. For Python projects, ALWAYS use 'pip install uv' as your first command, then use 'uv pip install --system' instead of 'pip install' for 10x faster installations. 
3. Only install the bare minimum requirements needed to run the entry point.
    
{format_instructions}"""),
    ("human", "FILE TREE:\n{file_tree}\n\nKEY FILE CONTENTS:\n{file_contents}\n\nProduce JSON plan.")
])
chain = prompt | llm | parser

def clean_requirements(content: str, max_lines: int = 40) -> str:
    lines = content.splitlines()
    seen = set()
    cleaned =[line for line in lines if not line.strip().startswith("#") and (pkg := line.split("==")[0]) not in seen and not seen.add(pkg)]
    return "\n".join(cleaned[:max_lines])

def scrape_repo_context(sandbox, workspace_root: str) -> tuple[str, str]:
    def execute(cmd):
        res = sandbox.process.exec(cmd)
        return getattr(res, "result", getattr(res, "stdout", "")).strip()

    file_tree = execute(f"find {workspace_root} -maxdepth 3 -not -path '*/.git/*' -not -path '*/node_modules/*' -not -path '*/__pycache__/*'")
    key_filenames =["requirements.txt", "package.json", "main.py", "app.py", "index.js", "README.md"]
    file_contents_parts =[]
    
    for line in file_tree.split("\n"):
        if any(line.strip().endswith(name) for name in key_filenames):
            content = execute(f"cat '{line.strip()}'")
            if content:
                rel_path = line.replace(workspace_root + "/", "")
                content = clean_requirements(content) if "requirements" in rel_path else content[:1500]
                file_contents_parts.append(f"### {rel_path}\n{content}")

    return file_tree, "\n\n".join(file_contents_parts)

def execute_repo(repo_link: str) -> dict:
    """Clones, plans, installs, and runs a repo in Daytona. Returns the results."""
    logger.info(f"[{repo_link}] Spinning up Daytona Sandbox...")
    sandbox = daytona.create()
    workspace_root = "/home/daytona/project"

    def execute(cmd, cwd=None):
        res = sandbox.process.exec(cmd, cwd=cwd)
        return getattr(res, "result", getattr(res, "stdout", "")).strip()

    try:
        logger.info(f"[{repo_link}] Cloning repository (shallow)...")
        execute(f"git clone --depth 1 {repo_link} {workspace_root}")
        
        file_tree, file_contents = scrape_repo_context(sandbox, workspace_root)
        
        logger.info(f"[{repo_link}] Generating execution plan...")
        plan_raw = chain.invoke({
            "file_tree": file_tree, 
            "file_contents": file_contents, 
            "format_instructions": parser.get_format_instructions()
        })
        plan = RepoExecutionPlan(**plan_raw)

        # SPEEDUP/FIX: Skip if it's just a markdown document/list
        if not plan.is_runnable:
            logger.info(f"[{repo_link}] AI determined repo is not runnable (e.g., Markdown list). Skipping execution.")
            return {
                "repo_url": repo_link,
                "success": True, 
                "project_type": plan.project_type,
                "reasoning": plan.reasoning,
                "output": "This repository is a documentation or list repository. No executable application found."
            }
        
        run_cwd = os.path.join(workspace_root, plan.working_directory) if plan.working_directory not in (".", "") else workspace_root
        
        # FIX: Added strict 60s timeout and non-interactive flag to prevent ANY install commands from hanging!
        logger.info(f"[{repo_link}] Installing dependencies...")
        for cmd in plan.install_commands:
            if not cmd.strip(): continue
            safe_install_cmd = f"export DEBIAN_FRONTEND=noninteractive; timeout 60s bash -c '{cmd}'"
            execute(safe_install_cmd, cwd=run_cwd)

        # Run project
        logger.info(f"[{repo_link}] Running project (with 15s timeout)...")
        final_output = execute(f"timeout 15s bash -c '{plan.run_command} 2>&1'", cwd=run_cwd)
        
        success = not any(err in final_output for err in["Traceback", "Error:", "Exception:"])
        
        return {
            "repo_url": repo_link,
            "success": success,
            "project_type": plan.project_type,
            "reasoning": plan.reasoning,
            "output": final_output[-1500:] 
        }
        
    except Exception as e:
        logger.error(f"[{repo_link}] Failed: {e}")
        return {"repo_url": repo_link, "success": False, "project_type": "Unknown", "reasoning": "Exception occurred", "output": str(e)}