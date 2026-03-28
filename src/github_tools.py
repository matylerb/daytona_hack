import os
import requests
import logging

logger = logging.getLogger(__name__)

def search_github_repos(topic: str, limit: int = 3) -> list[str]:
    """
    Searches GitHub for the top repositories related to a topic,
    excluding markdown lists and books. Uses GITHUB_TOKEN if available
    for higher rate limits (5000/hr vs 60/hr unauthenticated).
    """
    logger.info(f"Searching GitHub for topic: {topic}")

    # Filter out giant markdown lists and low-star noise
    query = f"{topic} NOT awesome NOT primer NOT book stars:>50"
    url = (
        f"https://api.github.com/search/repositories"
        f"?q={requests.utils.quote(query)}&sort=stars&order=desc&per_page={limit * 2}"
    )
    headers = {"Accept": "application/vnd.github.v3+json"}
    gh_token = os.environ.get("GITHUB_TOKEN", "")
    if gh_token:
        headers["Authorization"] = f"Bearer {gh_token}"

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    data  = response.json()
    items = data.get("items", [])

    repo_urls = []
    for item in items[:limit]:
        repo_urls.append(item["clone_url"])

    logger.info(f"Found repos: {repo_urls}")
    return repo_urls