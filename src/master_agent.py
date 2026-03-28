import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

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
        Three sub-agents have cloned 3 top GitHub repositories related to this topic, analyzed them, 
        installed their dependencies, and executed them. 
        
        Your job is to:
        1. Review the output from each repository.
        2. Explain briefly what each repository does based on its execution output.
        3. Rank the 3 repositories from most to least relevant/successful for the user's learning goal.
        4. Provide an output of the code execution results that is concise and focused on the most relevant information for the user, removing any irrelevant logs or errors.
        5. Provide a final recommendation.
        
        Format your response nicely in Markdown."""),
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
    response = chain.invoke({"topic": topic, "results": formatted_results})



    
    return response.content


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
