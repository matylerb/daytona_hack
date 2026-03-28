import os
import logging
import concurrent.futures
from dotenv import load_dotenv

from github_tools import search_github_repos
from researcher_agent import execute_repo
from master_agent import rank_and_summarize

# Load environment variables (.env file containing GROQ_API_KEY and DAYTONA_API_KEY)
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    print("="*60)
    print("Welcome to the Distributed GitHub AI Researcher")
    print("="*60)
    
    topic = input("\nWhat topic would you like to research? (e.g. 'machine learning with fastapi'): ").strip()
    if not topic:
        print("Topic cannot be empty. Exiting.")
        return

    # Step 1: Find Repositories
    print(f"\n[Master Agent] Searching GitHub for the top 3 repositories on '{topic}'...")
    repo_urls = search_github_repos(topic, limit=3)
    
    if not repo_urls:
        print("No repositories found. Exiting.")
        return

    # Step 2: Spin up 3 parallel Researcher Agents
    print("\n[Master Agent] Dispatching 3 Researcher Agents to clone and execute these repos. This may take a minute...")
    execution_results =[]
    
    # Run the 3 Daytona environments in parallel to save time
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Map the function to the URLs
        futures = {executor.submit(execute_repo, url): url for url in repo_urls}
        
        for future in concurrent.futures.as_completed(futures):
            url = futures[future]
            try:
                result = future.result()
                execution_results.append(result)
                success_text = "✅ Success" if result["success"] else "❌ Error/Failed"
                print(f"  -> Agent finished analyzing {url} [{success_text}]")
            except Exception as exc:
                print(f"  -> Agent crashed while analyzing {url}: {exc}")

    # Step 3: Master Agent evaluates the findings
    print("\n[Master Agent] All execution outputs received. Evaluating and ranking results...\n")
    final_report = rank_and_summarize(topic, execution_results)
    
    print("="*60)
    print("🏆 FINAL RESEARCH REPORT 🏆")
    print("="*60)
    print(final_report)

if __name__ == "__main__":
    main()