import os
import sys
import json
import logging
import queue
import concurrent.futures
import signal
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv

from github_tools import search_github_repos
from researcher_agent import execute_repo
from master_agent import rank_and_summarize

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == 'win32'

RESEARCH_TIMEOUT = 300  # 5 minutes
EXECUTE_TIMEOUT  = 300  # 5 minutes

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")
CORS(app)


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Request timed out")


# ── Original blocking endpoint (kept for backward compat) ──────────────────
@app.route("/api/research", methods=["POST"])
def research():
    data = request.get_json()
    topic = (data or {}).get("topic", "").strip()
    if not topic:
        return jsonify({"error": "Topic is required."}), 400

    logger.info(f"[Research] Topic: {topic}")

    if not IS_WINDOWS:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(RESEARCH_TIMEOUT)

    try:
        repo_urls = search_github_repos(topic, limit=3)
        if not repo_urls:
            return jsonify({"error": "No repositories found for that topic."}), 404

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

        try:
            final_report = rank_and_summarize(topic, execution_results)
        except Exception as e:
            logger.error(f"Master agent error: {e}")
            final_report = "Error generating report."

        return jsonify({"topic": topic, "repos": execution_results, "report": final_report})

    except TimeoutException:
        return jsonify({"error": "Research timed out."}), 500
    finally:
        if not IS_WINDOWS:
            signal.alarm(0)


# ── NEW: Streaming endpoint — sends each repo result as it arrives ──────────
@app.route("/api/research/stream", methods=["POST"])
def research_stream():
    """
    Server-Sent Events endpoint.
    The client receives:
      - event: repo      — one per agent as it completes (JSON)
      - event: report    — the master agent summary at the end
      - event: error     — on failure
      - event: done      — signals the stream is finished
    """
    data = request.get_json()
    topic = (data or {}).get("topic", "").strip()
    if not topic:
        return jsonify({"error": "Topic is required."}), 400

    result_queue: queue.Queue = queue.Queue()

    def agent_worker(url: str):
        try:
            result = execute_repo(url)
        except Exception as exc:
            result = {
                "repo_url": url,
                "success": False,
                "project_type": "Unknown",
                "reasoning": "Agent crashed",
                "output": str(exc)
            }
        result_queue.put(result)

    def generate():
        # Step 1: find repos
        try:
            repo_urls = search_github_repos(topic, limit=3)
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
            return

        if not repo_urls:
            yield f"event: error\ndata: {json.dumps({'error': 'No repositories found.'})}\n\n"
            return

        # Step 2: fire off all 3 agents, stream results as they finish
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(agent_worker, url) for url in repo_urls]

            execution_results = []
            received = 0
            while received < len(repo_urls):
                try:
                    result = result_queue.get(timeout=RESEARCH_TIMEOUT)
                    execution_results.append(result)
                    received += 1
                    # Send this repo's result immediately to the client
                    yield f"event: repo\ndata: {json.dumps(result)}\n\n"
                except queue.Empty:
                    yield f"event: error\ndata: {json.dumps({'error': 'Timed out waiting for agents.'})}\n\n"
                    return

        # Step 3: master agent summary
        try:
            final_report = rank_and_summarize(topic, execution_results)
        except Exception as e:
            logger.error(f"Master agent error: {e}")
            final_report = "Error generating report."

        yield f"event: report\ndata: {json.dumps({'report': final_report, 'topic': topic})}\n\n"
        yield f"event: done\ndata: {{}}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Disable nginx buffering if proxied
        }
    )


# ── Execute single repo ─────────────────────────────────────────────────────
@app.route("/api/execute", methods=["POST"])
def execute():
    data = request.get_json()
    repo_url = (data or {}).get("repo_url", "").strip()

    if not repo_url:
        return jsonify({"error": "repo_url is required."}), 400
    if not repo_url.startswith("http"):
        return jsonify({"error": "Please provide a full GitHub URL (https://...)."}), 400

    logger.info(f"[Execute] Repo: {repo_url}")

    if not IS_WINDOWS:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(EXECUTE_TIMEOUT)

    try:
        result = execute_repo(repo_url)
        return jsonify(result)
    except TimeoutException:
        return jsonify({"error": "Execute timed out."}), 500
    except Exception as e:
        logger.error(f"Execute error: {e}")
        return jsonify({"repo_url": repo_url, "success": False, "project_type": "Unknown", "reasoning": "Exception occurred", "output": str(e)})
    finally:
        if not IS_WINDOWS:
            signal.alarm(0)


# ── Health ──────────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}", exc_info=True)
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(404)
def handle_404(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(400)
def handle_400(e):
    return jsonify({"error": "Bad request"}), 400


if __name__ == "__main__":
    # threaded=True is required for SSE to work correctly with Flask dev server
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)