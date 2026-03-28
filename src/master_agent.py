import os
import logging
import signal
import sys
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

def rank_and_summarize(topic: str, execution_results: list[dict]) -> str:
    """
    Acts as the Master Agent. Takes the execution outputs of the 3 repos 
    and ranks their relevancy and success for the user.
    """
    llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY", ""),
        model="llama-3.3-70b-versatile",
        temperature=0.2,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Master Developer Agent. The user wants to learn about: "{topic}".
        Three sub-agents cloned, installed, and ran 3 top GitHub repositories on this topic.

        Be concise and direct. Your response should be under 600 words total.

        Structure your Markdown response as:
        1. **Quick Summary** — one sentence per repo describing what it does.
        2. **Ranked List** — rank repos #1 to #3 with a 1-2 sentence reason each.
        3. **Execution Highlights** — only the most relevant lines of output per repo (skip noise/errors).
        4. **Recommendation** — one short paragraph on which to use and why.

        Do not repeat repo URLs more than once. Skip any repo that produced no useful output."""),
        ("human", "Here are the execution results from the researchers:\n\n{results}")
    ])

    # Format the results into a readable string for the LLM
    formatted_results = ""
    for i, res in enumerate(execution_results, 1):
        formatted_results += f"### Repository {i}: {res['repo_url']}\n"
        formatted_results += f"- Project Type: {res['project_type']}\n"
        formatted_results += f"- Execution Successful: {res['success']}\n"
        formatted_results += f"- Execution Output (Last 2000 chars):\n```\n{res['output']}\n```\n\n"

    chain = prompt | llm

    class TimeoutException(Exception):
        pass

    def timeout_handler(signum, frame):
        raise TimeoutException("Master agent report generation timed out")


    # Set a 120 second timeout for master agent (Unix only)
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)  # 60s is enough for a concise report

    try:
        response = chain.invoke({"topic": topic, "results": formatted_results})
        return response.content
    except TimeoutException as e:
        logger.error(f"Master agent timeout: {e}")
        return f"# Research Report on '{topic}'\n\nMaster agent timed out while generating report.\n\n## Execution Results Summary:\n\n{formatted_results}"
    finally:
        # Cancel the alarm if on Unix
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)


def create_markdown(topic: str, execution_results: list[dict]) -> str:
    """
    Creates a markdown report summarizing the execution results of the repositories.
    """
    markdown = f"# Research Report on '{topic}'\n\n"
    
    for i, res in enumerate(execution_results, 1):
        markdown += f"## Repository {i}: {res['repo_url']}\n"
        markdown += f"- **Project Type**: {res['project_type']}\n"
        markdown += f"- **Execution Successful**: {'Yes' if res['success'] else 'No'}\n"
        markdown += f"- **Execution Output (Last 2000 chars)**:\n```\n{res['output']}\n```\n\n"

    return markdown