import requests
import logging

logger = logging.getLogger(__name__)

def search_github_repos(topic: str, limit: int = 3) -> list[str]:
    """
    Searches GitHub for the top repositories related to a topic, 
    excluding markdown lists and books.
    """
    logger.info(f"Searching GitHub for topic: {topic}")
    
    # Appending NOT awesome NOT primer to filter out giant markdown lists
    query = f"{topic} NOT awesome NOT primer"
    url = f"https://api.github.com/search/repositories?q={requests.utils.quote(query)}&sort=stars&order=desc"
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    data = response.json()
    items = data.get("items",[])
    
    repo_urls = []
    for item in items[:limit]:
        repo_urls.append(item["clone_url"])
        
    logger.info(f"Found repos: {repo_urls}")
    return repo_urls