import os
import logging
import concurrent.futures
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

from github_tools import search_github_repos
from researcher_agent import execute_repo
from master_agent import rank_and_summarize

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Serve index.html from the same directory as this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")
CORS(app)


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/research", methods=["POST"])
def research():
    """Research Mode: find top repos on a topic, execute them, rank results."""
    data = request.get_json()
    topic = (data or {}).get("topic", "").strip()

    if not topic:
        return jsonify({"error": "Topic is required."}), 400

    logger.info(f"[Research] Topic: {topic}")

    # Step 1: Find repos
    repo_urls = search_github_repos(topic, limit=3)
    if not repo_urls:
        return jsonify({"error": "No repositories found for that topic."}), 404

    # Step 2: Execute repos in parallel
    execution_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(execute_repo, url): url for url in repo_urls}
        for future in concurrent.futures.as_completed(futures):
            url = futures[future]
            try:
                result = future.result()
                execution_results.append(result)
            except Exception as exc:
                execution_results.append({
                    "repo_url": url,
                    "success": False,
                    "project_type": "Unknown",
                    "reasoning": "Agent crashed",
                    "output": str(exc)
                })

    # Step 3: Master agent ranks
    final_report = rank_and_summarize(topic, execution_results)

    return jsonify({
        "topic": topic,
        "repos": execution_results,
        "report": final_report
    })


@app.route("/api/execute", methods=["POST"])
def execute():
    """Execute Mode: clone and run a single GitHub repo URL."""
    data = request.get_json()
    repo_url = (data or {}).get("repo_url", "").strip()

    if not repo_url:
        return jsonify({"error": "repo_url is required."}), 400

    if not repo_url.startswith("http"):
        return jsonify({"error": "Please provide a full GitHub URL (https://...)."}), 400

    logger.info(f"[Execute] Repo: {repo_url}")
    result = execute_repo(repo_url)

    return jsonify(result)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)