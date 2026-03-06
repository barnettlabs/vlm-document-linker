"""
app.py -- Web dashboard for OCR benchmark stack.
Reads all state from Redis and serves a live dashboard + review UI.
"""

import os
import json
import io
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_file, abort
import redis
import docker
import fitz  # pymupdf for PDF rendering

app = Flask(__name__)

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
INPUT_DIR = os.environ.get("INPUT_DIR", "/input")
OLLAMA_ENDPOINTS = [
    url.strip() for url in
    os.environ.get("OLLAMA_ENDPOINTS", "").split(",") if url.strip()
]

r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
docker_client = docker.DockerClient(base_url="unix:///var/run/docker.sock")

MANAGED_CONTAINERS = [
    "vdl-ollama-gpu0",
    "vdl-ollama-gpu1",
    "vdl-ollama-pull-gpu0",
    "vdl-ollama-pull-gpu1",
    "vdl-worker",
    "vdl-enqueuer",
    "vdl-exporter",
]

KNOWN_KEYS = [
    "queue:pending",
    "files",
    "failures",
    "config:pipeline",
]


def get_all_files() -> list[dict]:
    """Fetch all file records from Redis."""
    data = r.hgetall("files")
    records = []
    for key, val in data.items():
        try:
            rec = json.loads(val)
            rec.setdefault("_key", key)
            records.append(rec)
        except json.JSONDecodeError:
            pass
    return records


# -- Pages -------------------------------------------------------------------

@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/review")
def review_page():
    return render_template("review.html")


# -- Status API --------------------------------------------------------------

@app.route("/api/status")
def api_status():
    """Main API endpoint -- returns all dashboard data."""
    queue_pending = r.llen("queue:pending")
    queue_items = r.lrange("queue:pending", 0, 49)
    files = get_all_files()

    auto_accepted = [f for f in files if f.get("status") == "auto_accepted"]
    needs_review = [f for f in files if f.get("status") == "needs_review"]
    reviewed = [f for f in files if f.get("status") == "reviewed"]
    errored = [f for f in files if f.get("status") == "error"]

    # Get in-progress files
    processing_raw = r.hgetall("processing")
    processing = []
    for path_key, val in processing_raw.items():
        try:
            processing.append(json.loads(val))
        except Exception:
            pass

    # Compute per-model stats from passes
    model_stats = {}
    for rec in files:
        for p in rec.get("passes", []):
            model = p.get("model", "unknown")
            if model not in model_stats:
                model_stats[model] = {
                    "total": 0,
                    "succeeded": 0,
                    "failed": 0,
                    "times": [],
                    "confidences": [],
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_tokens": 0,
                }
            stats = model_stats[model]
            stats["total"] += 1
            stats["total_prompt_tokens"] += p.get("prompt_tokens", 0) or 0
            stats["total_completion_tokens"] += p.get("completion_tokens", 0) or 0
            stats["total_tokens"] += p.get("total_tokens", 0) or 0
            if p.get("error"):
                stats["failed"] += 1
            else:
                stats["succeeded"] += 1
                if p.get("inference_seconds"):
                    stats["times"].append(p["inference_seconds"])
                if p.get("confidence") is not None:
                    stats["confidences"].append(p["confidence"])

    for model, stats in model_stats.items():
        times = stats.pop("times")
        confidences = stats.pop("confidences")
        stats["avg_time"] = round(sum(times) / len(times), 2) if times else 0
        stats["avg_confidence"] = round(sum(confidences) / len(confidences), 2) if confidences else 0

    return jsonify({
        "queue_pending": queue_pending,
        "queue_items": queue_items,
        "processing": processing,
        "processing_count": len(processing),
        "total_files": len(files),
        "auto_accepted": len(auto_accepted),
        "needs_review": len(needs_review),
        "reviewed": len(reviewed),
        "errored": len(errored),
        "model_stats": model_stats,
        "files": sorted(files, key=lambda x: x.get("original_filename", "")),
    })


# -- Pipeline Config ---------------------------------------------------------

@app.route("/api/pipeline")
def get_pipeline():
    raw = r.get("config:pipeline")
    pipeline = json.loads(raw) if raw else []
    return jsonify({"pipeline": pipeline})


@app.route("/api/pipeline", methods=["POST"])
def set_pipeline():
    pipeline = request.json.get("pipeline", [])
    r.set("config:pipeline", json.dumps(pipeline))
    return jsonify({"ok": True})


@app.route("/api/pipeline/models")
def get_available_models():
    """Query all Ollama instances from pipeline config for available models."""
    import urllib.request

    # Use configured endpoints, plus any additional ones from pipeline config
    seen_urls = set(OLLAMA_ENDPOINTS)
    raw = r.get("config:pipeline")
    pipeline = json.loads(raw) if raw else []
    for entry in pipeline:
        base_url = entry.get("base_url", "")
        ollama_root = base_url.replace("/v1", "")
        if ollama_root:
            seen_urls.add(ollama_root)

    models = []
    for ollama_root in seen_urls:
        try:
            req = urllib.request.Request(f"{ollama_root}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                for model in data.get("models", []):
                    models.append({
                        "name": model["name"],
                        "base_url": f"{ollama_root}/v1",
                        "label": model["name"],
                        "size": model.get("size", 0),
                    })
        except Exception:
            pass

    return jsonify({"models": models})


# -- File Serving ------------------------------------------------------------

@app.route("/api/file-image")
def serve_file_image():
    """Serve a file as an image (renders PDFs to JPEG)."""
    file_path = request.args.get("path")
    if not file_path:
        abort(400)

    path = Path(file_path)
    if not path.exists():
        abort(404)

    # Security: ensure file is within input directory
    try:
        path.resolve().relative_to(Path(INPUT_DIR).resolve())
    except ValueError:
        abort(403)

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        doc = fitz.open(str(path))
        mat = fitz.Matrix(200 / 72, 200 / 72)  # 200 DPI for review
        pix = doc[0].get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("jpeg")
        return send_file(io.BytesIO(img_bytes), mimetype="image/jpeg")

    elif suffix in {".jpg", ".jpeg"}:
        return send_file(path, mimetype="image/jpeg")

    elif suffix == ".png":
        return send_file(path, mimetype="image/png")

    else:
        abort(415)


@app.route("/api/file-detail")
def file_detail():
    """Get all stored data for a specific file."""
    file_path = request.args.get("path")
    if not file_path:
        abort(400)

    raw = r.hget("files", file_path)
    if not raw:
        abort(404)

    return jsonify(json.loads(raw))


# -- Review ------------------------------------------------------------------

@app.route("/api/review-queue")
def review_queue():
    """Get all files needing human review."""
    files = get_all_files()
    needs_review = [f for f in files if f.get("status") == "needs_review"]
    return jsonify({
        "files": sorted(needs_review, key=lambda x: x.get("original_filename", "")),
    })


@app.route("/api/review", methods=["POST"])
def submit_review():
    """Submit a human review decision."""
    file_path = request.json.get("path")
    final_value = request.json.get("final_value")

    if not file_path or final_value is None:
        return jsonify({"error": "path and final_value required"}), 400

    raw = r.hget("files", file_path)
    if not raw:
        return jsonify({"error": "file not found"}), 404

    record = json.loads(raw)
    record["status"] = "reviewed"
    record["final_value"] = final_value
    record["reviewed_by"] = "human"
    record["reviewed_at"] = datetime.now(timezone.utc).isoformat()

    r.hset("files", file_path, json.dumps(record))
    return jsonify({"ok": True})


# -- Queue Management --------------------------------------------------------

@app.route("/api/queue/clear", methods=["POST"])
def clear_queue():
    r.delete("queue:pending")
    return jsonify({"ok": True})


@app.route("/api/queue/remove", methods=["POST"])
def remove_from_queue():
    item = request.json.get("item")
    if item:
        r.lrem("queue:pending", 0, item)
    return jsonify({"ok": True})


@app.route("/api/requeue-failed", methods=["POST"])
def requeue_failed():
    """Re-enqueue all errored files."""
    files = get_all_files()
    errored = [f for f in files if f.get("status") == "error"]
    count = 0
    for rec in errored:
        path = rec.get("path") or rec.get("_key")
        if path:
            r.hdel("files", path)
            r.hdel("failures", path)
            r.lpush("queue:pending", path)
            count += 1
    return jsonify({"ok": True, "requeued": count})


@app.route("/api/requeue-file", methods=["POST"])
def requeue_file():
    """Re-run a specific file through the pipeline."""
    file_path = request.json.get("path")
    if not file_path:
        return jsonify({"error": "path required"}), 400
    r.hdel("files", file_path)
    r.hdel("failures", file_path)
    r.lpush("queue:pending", file_path)
    return jsonify({"ok": True})


# -- Redis Keys --------------------------------------------------------------

@app.route("/api/keys")
def api_keys():
    keys = []
    for key in KNOWN_KEYS:
        key_type = r.type(key)
        if key_type == "hash":
            size = r.hlen(key)
        elif key_type == "list":
            size = r.llen(key)
        elif key_type == "string":
            size = 1
        elif key_type == "none":
            size = 0
        else:
            size = 0
        keys.append({"key": key, "type": key_type, "size": size})
    return jsonify({"keys": keys})


@app.route("/api/keys/delete", methods=["POST"])
def delete_key():
    key = request.json.get("key")
    if key not in KNOWN_KEYS:
        return jsonify({"error": "unknown key"}), 400
    r.delete(key)
    return jsonify({"ok": True, "deleted": key})


@app.route("/api/keys/delete-all", methods=["POST"])
def delete_all_keys():
    for key in KNOWN_KEYS:
        r.delete(key)
    return jsonify({"ok": True, "deleted": KNOWN_KEYS})


# -- Docker Containers -------------------------------------------------------

@app.route("/api/containers")
def api_containers():
    containers = []
    for name in MANAGED_CONTAINERS:
        try:
            c = docker_client.containers.get(name)
            containers.append({
                "name": name,
                "status": c.status,
                "running": c.status == "running",
            })
        except docker.errors.NotFound:
            containers.append({
                "name": name,
                "status": "not created",
                "running": False,
            })
    return jsonify({"containers": containers})


@app.route("/api/containers/start", methods=["POST"])
def start_container():
    name = request.json.get("name")
    if name not in MANAGED_CONTAINERS:
        return jsonify({"error": "unknown container"}), 400
    try:
        c = docker_client.containers.get(name)
        if c.status != "running":
            c.start()
        return jsonify({"ok": True, "status": c.status})
    except docker.errors.NotFound:
        return jsonify({"error": f"{name} has not been created yet. Run it once from the CLI first."}), 404


@app.route("/api/containers/stop", methods=["POST"])
def stop_container():
    name = request.json.get("name")
    if name not in MANAGED_CONTAINERS:
        return jsonify({"error": "unknown container"}), 400
    try:
        c = docker_client.containers.get(name)
        if c.status == "running":
            c.stop()
        return jsonify({"ok": True, "status": "stopped"})
    except docker.errors.NotFound:
        return jsonify({"error": f"{name} not found"}), 404


@app.route("/api/containers/restart", methods=["POST"])
def restart_container():
    name = request.json.get("name")
    if name not in MANAGED_CONTAINERS:
        return jsonify({"error": "unknown container"}), 400
    try:
        c = docker_client.containers.get(name)
        c.restart()
        return jsonify({"ok": True, "status": "restarting"})
    except docker.errors.NotFound:
        return jsonify({"error": f"{name} not found"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
