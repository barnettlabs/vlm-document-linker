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
import base64
import time
import openai
import anthropic
from PIL import Image
import pytesseract

app = Flask(__name__)

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
INPUT_DIR = os.environ.get("INPUT_DIR", "/input")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OLLAMA_ENDPOINTS = [
    url.strip() for url in
    os.environ.get("OLLAMA_ENDPOINTS", "").split(",") if url.strip()
]

# Claude pricing per million tokens
CLAUDE_PRICING = {
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
}

r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
docker_client = docker.DockerClient(base_url="unix:///var/run/docker.sock")

MANAGED_CONTAINERS = [
    "vdl-ollama-gpu0",
    "vdl-ollama-gpu1",
    "vdl-ollama-multi",
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


@app.route("/test")
def test_page():
    return render_template("test.html")


# -- Image orientation -------------------------------------------------------

def auto_rotate(img: Image.Image) -> tuple[Image.Image, int]:
    """Detect and correct image rotation via Tesseract OSD.

    Tesseract's 'Rotate:' reports how many degrees the text is currently rotated.
    To correct it, we rotate by (360 - angle) since Pillow rotates counter-clockwise.
    """
    try:
        osd = pytesseract.image_to_osd(img)
        for line in osd.splitlines():
            if line.startswith("Rotate:"):
                angle = int(line.split(":")[1].strip())
                if angle != 0:
                    correction = (360 - angle) % 360
                    if correction != 0:
                        return img.rotate(correction, expand=True), correction
    except Exception:
        pass
    return img, 0


# -- Prompt Presets ----------------------------------------------------------

# Mirrors the prompts from workers/worker.py so the test page can use them
PROMPT_PRESETS = {
    "lead_paint": {
        "label": "Lead Paint Certificate",
        "prompt": """/nothink
You are extracting a certificate number from a lead inspection certificate.

Find the value of the field labeled "INSPECTION CERTIFICATE NO" in this document.
This field should always be present on the document. It is typically a numeric or
alphanumeric identifier.

Rules:
- Return the exact value shown in the field, do not modify or reformat it.
- confidence should reflect how clearly you can read the value (0.0 to 1.0).
- If you cannot find a field labeled "INSPECTION CERTIFICATE NO", set found to false
  and leave certificate_number as null.

Return ONLY valid JSON, no markdown, no explanation:
{
  "certificate_number": "the extracted value or null",
  "found": true,
  "confidence": 0.95
}""",
    },
    "rental_license": {
        "label": "Rental License",
        "prompt": """/nothink
You are extracting a license number from a rental license document.

Look for the license number in this document. It may appear as:
- A field labeled "Number" (typically in the top-right area of the document)
- A field labeled "License #" (sometimes at the bottom of the document)

The same license number may appear in multiple places. Extract the value you find.

Rules:
- Return the exact value shown in the field, do not modify or reformat it.
- confidence should reflect how clearly you can read the value (0.0 to 1.0).
- If you cannot find a license number, set found to false and leave certificate_number as null.

Return ONLY valid JSON, no markdown, no explanation:
{
  "certificate_number": "the extracted value or null",
  "found": true,
  "confidence": 0.95
}""",
    },
}


@app.route("/api/prompts")
def get_prompts():
    """Return available prompt presets."""
    return jsonify({
        "presets": {k: {"label": v["label"], "prompt": v["prompt"]} for k, v in PROMPT_PRESETS.items()}
    })


# -- Test API ----------------------------------------------------------------

def file_to_base64(file_path: str) -> tuple[str, str]:
    """Convert a file to base64 JPEG for vision model ingestion.
    Applies auto-rotation correction via Tesseract OSD."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        doc = fitz.open(str(path))
        mat = fitz.Matrix(150 / 72, 150 / 72)
        pix = doc[0].get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        doc.close()
    elif suffix in {".jpg", ".jpeg", ".png"}:
        img = Image.open(path).convert("RGB")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    img, _ = auto_rotate(img)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"


@app.route("/api/test/files")
def test_list_files():
    """List available files in the input directory."""
    input_path = Path(INPUT_DIR)
    files = []
    for f in sorted(input_path.rglob("*")):
        if f.suffix.lower() in {".pdf", ".jpg", ".jpeg", ".png"}:
            files.append(str(f))
    return jsonify({"files": files})


@app.route("/api/test/run", methods=["POST"])
def test_run_model():
    """Run a single model against a file and return the raw response."""
    file_path = request.json.get("file")
    model_name = request.json.get("model")
    base_url = request.json.get("base_url")
    provider = request.json.get("provider", "ollama")
    prompt = request.json.get("prompt", "OCR this document")

    if not file_path or not model_name:
        return jsonify({"error": "file and model required"}), 400

    if provider == "ollama" and not base_url:
        return jsonify({"error": "base_url required for Ollama models"}), 400

    path = Path(file_path)
    if not path.exists():
        return jsonify({"error": f"File not found: {file_path}"}), 404

    try:
        path.resolve().relative_to(Path(INPUT_DIR).resolve())
    except ValueError:
        return jsonify({"error": "File must be within input directory"}), 403

    try:
        image_data, media_type = file_to_base64(file_path)

        if provider == "claude":
            return _run_claude(model_name, image_data, media_type, prompt)
        else:
            return _run_ollama(model_name, base_url, image_data, media_type, prompt)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _run_ollama(model_name, base_url, image_data, media_type, prompt):
    client = openai.OpenAI(base_url=base_url, api_key="not-needed")
    t0 = time.time()
    response = client.chat.completions.create(
        model=model_name,
        max_tokens=2048,
        temperature=0.0,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{image_data}"},
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )
    elapsed = round(time.time() - t0, 2)

    usage = getattr(response, "usage", None)
    raw_content = response.choices[0].message.content or ""
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0

    return jsonify({
        "raw_response": raw_content,
        "provider": "ollama",
        "model": model_name,
        "elapsed_seconds": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    })


def _run_claude(model_name, image_data, media_type, prompt):
    if not ANTHROPIC_API_KEY:
        return jsonify({"error": "ANTHROPIC_API_KEY not configured"}), 400

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    t0 = time.time()
    response = client.messages.create(
        model=model_name,
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )
    elapsed = round(time.time() - t0, 2)

    raw_content = response.content[0].text if response.content else ""
    prompt_tokens = response.usage.input_tokens
    completion_tokens = response.usage.output_tokens

    # Calculate cost
    pricing = CLAUDE_PRICING.get(model_name, {"input": 0, "output": 0})
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost

    return jsonify({
        "raw_response": raw_content,
        "provider": "claude",
        "model": model_name,
        "elapsed_seconds": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost": {
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6),
            "input_rate": pricing["input"],
            "output_rate": pricing["output"],
        },
    })


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

    # Grand total token counts across all models
    grand_prompt_tokens = sum(s["total_prompt_tokens"] for s in model_stats.values())
    grand_completion_tokens = sum(s["total_completion_tokens"] for s in model_stats.values())
    grand_total_tokens = sum(s["total_tokens"] for s in model_stats.values())
    files_with_passes = sum(1 for f in files if f.get("passes"))

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
        "token_totals": {
            "prompt_tokens": grand_prompt_tokens,
            "completion_tokens": grand_completion_tokens,
            "total_tokens": grand_total_tokens,
            "avg_tokens_per_file": round(grand_total_tokens / files_with_passes) if files_with_passes else 0,
        },
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
    seen_models = set()
    for ollama_root in seen_urls:
        try:
            req = urllib.request.Request(f"{ollama_root}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                for model in data.get("models", []):
                    if model["name"] not in seen_models:
                        seen_models.add(model["name"])
                        models.append({
                            "name": model["name"],
                            "label": model["name"],
                            "size": model.get("size", 0),
                        })
        except Exception:
            pass

    # Build friendly endpoint labels from hostnames
    endpoints = []
    for url in OLLAMA_ENDPOINTS:
        host = url.replace("http://", "").replace("https://", "").split(":")[0]
        endpoints.append({"label": host, "base_url": f"{url}/v1"})

    return jsonify({"models": models, "endpoints": endpoints})


# -- File Serving ------------------------------------------------------------

@app.route("/api/file-info")
def file_info():
    """Return metadata about a file (page count, type)."""
    file_path = request.args.get("path")
    if not file_path:
        abort(400)

    path = Path(file_path)
    if not path.exists():
        abort(404)

    try:
        path.resolve().relative_to(Path(INPUT_DIR).resolve())
    except ValueError:
        abort(403)

    suffix = path.suffix.lower()
    pages = 1
    if suffix == ".pdf":
        doc = fitz.open(str(path))
        pages = len(doc)
        doc.close()

    return jsonify({"pages": pages, "type": suffix})


@app.route("/api/file-image")
def serve_file_image():
    """Serve a file as an image (renders PDFs to JPEG). Accepts ?page=N for PDFs."""
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
    page = int(request.args.get("page", 0))

    if suffix == ".pdf":
        doc = fitz.open(str(path))
        if page >= len(doc):
            doc.close()
            abort(404)
        mat = fitz.Matrix(200 / 72, 200 / 72)  # 200 DPI for review
        pix = doc[page].get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        doc.close()
    elif suffix in {".jpg", ".jpeg", ".png"}:
        img = Image.open(path).convert("RGB")
    else:
        abort(415)

    img, _ = auto_rotate(img)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


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
