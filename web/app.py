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
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
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

# Groq pricing per million tokens
GROQ_PRICING = {
    "meta-llama/llama-4-scout-17b-16e-instruct": {"input": 0.11, "output": 0.34},
    "meta-llama/llama-4-maverick-17b-128e-instruct": {"input": 0.50, "output": 0.77},
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


# -- Run helpers -------------------------------------------------------------

def run_keys(run_name: str) -> dict:
    """Return namespaced Redis key names for a given run."""
    return {
        "files": f"run:{run_name}:files",
        "processing": f"run:{run_name}:processing",
        "failures": f"run:{run_name}:failures",
        "queue": f"run:{run_name}:queue",
        "pipeline": f"run:{run_name}:pipeline",
    }


def get_active_run() -> str:
    name = r.get("config:active_run")
    if not name:
        r.set("config:active_run", "default")
        return "default"
    return name


def migrate_legacy_keys():
    """One-time migration: move old flat keys into run:default:* namespace."""
    if r.exists("files") and not r.exists("runs"):
        keys = run_keys("default")
        r.rename("files", keys["files"])
        if r.exists("processing"):
            r.rename("processing", keys["processing"])
        if r.exists("failures"):
            r.rename("failures", keys["failures"])
        if r.exists("queue:pending"):
            r.rename("queue:pending", keys["queue"])
        pipeline_raw = r.get("config:pipeline")
        if pipeline_raw:
            r.set(keys["pipeline"], pipeline_raw)
        r.hset("runs", "default", json.dumps({
            "name": "default",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }))
        r.set("config:active_run", "default")
    elif not r.exists("runs"):
        r.hset("runs", "default", json.dumps({
            "name": "default",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }))
        r.set("config:active_run", "default")


migrate_legacy_keys()


def get_all_files(run_name: str = None) -> list[dict]:
    """Fetch all file records from Redis for a given run."""
    if run_name is None:
        run_name = get_active_run()
    keys = run_keys(run_name)
    data = r.hgetall(keys["files"])
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

def file_to_base64(file_path: str, page: int = 0) -> tuple[str, str]:
    """Convert a single page of a file to base64 JPEG for vision model ingestion.
    Applies auto-rotation correction via Tesseract OSD."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        doc = fitz.open(str(path))
        mat = fitz.Matrix(150 / 72, 150 / 72)
        pix = doc[page].get_pixmap(matrix=mat)
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


def file_all_pages_to_base64(file_path: str) -> list[tuple[str, str]]:
    """Convert all pages of a file to a list of (base64_data, media_type) tuples."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        doc = fitz.open(str(path))
        pages = []
        mat = fitz.Matrix(150 / 72, 150 / 72)
        for i in range(len(doc)):
            pix = doc[i].get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            img, _ = auto_rotate(img)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            pages.append((base64.b64encode(buf.getvalue()).decode(), "image/jpeg"))
        doc.close()
        return pages
    else:
        return [file_to_base64(file_path, page=0)]


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
    """Run a single model against a file and return the raw response.

    Accepts optional 'page' param: integer for a specific page, or "all" to
    send every page as separate images in one request. Defaults to 0.
    """
    file_path = request.json.get("file")
    model_name = request.json.get("model")
    base_url = request.json.get("base_url")
    provider = request.json.get("provider", "ollama")
    prompt = request.json.get("prompt", "OCR this document")
    page_param = request.json.get("page", 0)

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
        if page_param == "all":
            images = file_all_pages_to_base64(file_path)
        else:
            images = [file_to_base64(file_path, page=int(page_param))]

        if provider == "claude":
            return _run_claude(model_name, images, prompt)
        elif provider == "groq":
            return _run_groq(model_name, images, prompt)
        else:
            return _run_ollama(model_name, base_url, images, prompt)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _build_openai_content(images, prompt):
    """Build OpenAI-compatible content array with one or more images + text."""
    content = []
    for image_data, media_type in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{image_data}"},
        })
    content.append({"type": "text", "text": prompt})
    return content


def _run_ollama(model_name, base_url, images, prompt):
    client = openai.OpenAI(base_url=base_url, api_key="not-needed")
    t0 = time.time()
    response = client.chat.completions.create(
        model=model_name,
        max_tokens=2048,
        temperature=0.0,
        messages=[{"role": "user", "content": _build_openai_content(images, prompt)}],
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
        "pages_sent": len(images),
    })


def _run_claude(model_name, images, prompt):
    if not ANTHROPIC_API_KEY:
        return jsonify({"error": "ANTHROPIC_API_KEY not configured"}), 400

    content = []
    for image_data, media_type in images:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_data,
            },
        })
    content.append({"type": "text", "text": prompt})

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    t0 = time.time()
    response = client.messages.create(
        model=model_name,
        max_tokens=2048,
        messages=[{"role": "user", "content": content}],
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
        "pages_sent": len(images),
        "cost": {
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6),
            "input_rate": pricing["input"],
            "output_rate": pricing["output"],
        },
    })


def _run_groq(model_name, images, prompt):
    if not GROQ_API_KEY:
        return jsonify({"error": "GROQ_API_KEY not configured"}), 400

    client = openai.OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
    t0 = time.time()
    response = client.chat.completions.create(
        model=model_name,
        max_tokens=2048,
        temperature=0.0,
        messages=[{"role": "user", "content": _build_openai_content(images, prompt)}],
    )
    elapsed = round(time.time() - t0, 2)

    usage = getattr(response, "usage", None)
    raw_content = response.choices[0].message.content or ""
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0

    # Calculate cost
    pricing = GROQ_PRICING.get(model_name, {"input": 0, "output": 0})
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost

    return jsonify({
        "raw_response": raw_content,
        "provider": "groq",
        "model": model_name,
        "elapsed_seconds": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "pages_sent": len(images),
        "cost": {
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6),
            "input_rate": pricing["input"],
            "output_rate": pricing["output"],
        },
    })


# -- Cost Estimation ---------------------------------------------------------

# Hosted vision LLM pricing: (input $/M tokens, output $/M tokens)
HOSTED_PRICING = {
    "GPT-4o": {"input": 2.50, "output": 10.00},
    "GPT-4o mini": {"input": 0.15, "output": 0.60},
    "GPT-4.1": {"input": 2.00, "output": 8.00},
    "GPT-4.1 mini": {"input": 0.40, "output": 1.60},
    "GPT-4.1 nano": {"input": 0.10, "output": 0.40},
    "Claude Sonnet 4": {"input": 3.00, "output": 15.00},
    "Claude Haiku 4.5": {"input": 0.80, "output": 4.00},
    "Claude Opus 4": {"input": 15.00, "output": 75.00},
    "Gemini 2.5 Flash": {"input": 0.15, "output": 0.60},
    "Gemini 2.5 Pro": {"input": 1.25, "output": 10.00},
    "Groq Llama 4 Scout": {"input": 0.11, "output": 0.34},
    "Groq Llama 4 Maverick": {"input": 0.50, "output": 0.77},
}


def _estimate_costs(prompt_tokens: int, completion_tokens: int) -> list[dict]:
    """Estimate what the current token usage would cost on hosted APIs."""
    estimates = []
    for name, pricing in HOSTED_PRICING.items():
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        estimates.append({
            "provider": name,
            "input_rate": pricing["input"],
            "output_rate": pricing["output"],
            "input_cost": round(input_cost, 4),
            "output_cost": round(output_cost, 4),
            "total_cost": round(input_cost + output_cost, 4),
        })
    estimates.sort(key=lambda x: x["total_cost"])
    return estimates


# -- Status API --------------------------------------------------------------

@app.route("/api/status")
def api_status():
    """Main API endpoint -- returns all dashboard data."""
    run_name = request.args.get("run") or get_active_run()
    keys = run_keys(run_name)

    queue_pending = r.llen(keys["queue"])
    queue_items = r.lrange(keys["queue"], 0, 49)
    files = get_all_files(run_name)

    auto_accepted = [f for f in files if f.get("status") == "auto_accepted"]
    needs_review = [f for f in files if f.get("status") == "needs_review"]
    reviewed = [f for f in files if f.get("status") == "reviewed"]
    errored = [f for f in files if f.get("status") == "error"]

    # Get in-progress files
    processing_raw = r.hgetall(keys["processing"])
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

    # Grand totals across all models
    grand_prompt_tokens = sum(s["total_prompt_tokens"] for s in model_stats.values())
    grand_completion_tokens = sum(s["total_completion_tokens"] for s in model_stats.values())
    grand_total_tokens = sum(s["total_tokens"] for s in model_stats.values())
    grand_succeeded = sum(s["succeeded"] for s in model_stats.values())
    grand_failed = sum(s["failed"] for s in model_stats.values())
    files_with_passes = sum(1 for f in files if f.get("passes"))
    avg_prompt = round(grand_prompt_tokens / files_with_passes) if files_with_passes else 0
    avg_completion = round(grand_completion_tokens / files_with_passes) if files_with_passes else 0

    # Aggregate timing and confidence across all passes
    all_times = []
    all_confidences = []
    for rec in files:
        for p in rec.get("passes", []):
            if not p.get("error"):
                if p.get("inference_seconds"):
                    all_times.append(p["inference_seconds"])
                if p.get("confidence") is not None:
                    all_confidences.append(p["confidence"])
    avg_time = round(sum(all_times) / len(all_times), 2) if all_times else 0
    total_time = round(sum(all_times), 2)
    avg_confidence = round(sum(all_confidences) / len(all_confidences), 2) if all_confidences else 0

    # Cost estimates: what this workload would cost on hosted vision APIs
    # Prices per million tokens (input / output)
    cost_estimates = _estimate_costs(grand_prompt_tokens, grand_completion_tokens)

    # Collect run list for the selector
    runs_raw = r.hgetall("runs")
    runs_list = []
    for rname, rval in runs_raw.items():
        try:
            meta = json.loads(rval)
            meta["file_count"] = r.hlen(f"run:{rname}:files")
            meta["queue_depth"] = r.llen(f"run:{rname}:queue")
            runs_list.append(meta)
        except Exception:
            pass
    runs_list.sort(key=lambda x: x.get("created_at", ""))

    return jsonify({
        "active_run": get_active_run(),
        "viewing_run": run_name,
        "runs": runs_list,
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
        "pass_totals": {
            "succeeded": grand_succeeded,
            "failed": grand_failed,
            "avg_time": avg_time,
            "total_time": total_time,
            "avg_confidence": avg_confidence,
        },
        "token_totals": {
            "prompt_tokens": grand_prompt_tokens,
            "completion_tokens": grand_completion_tokens,
            "total_tokens": grand_total_tokens,
            "avg_prompt_per_file": avg_prompt,
            "avg_completion_per_file": avg_completion,
            "avg_tokens_per_file": round(grand_total_tokens / files_with_passes) if files_with_passes else 0,
        },
        "cost_estimates": cost_estimates,
        "files": sorted(files, key=lambda x: x.get("original_filename", "")),
    })


# -- Pipeline Config ---------------------------------------------------------

@app.route("/api/pipeline")
def get_pipeline():
    run_name = request.args.get("run")
    if run_name:
        raw = r.get(f"run:{run_name}:pipeline")
        if raw:
            return jsonify({"pipeline": json.loads(raw)})
    raw = r.get("config:pipeline")
    pipeline = json.loads(raw) if raw else []
    return jsonify({"pipeline": pipeline})


@app.route("/api/pipeline", methods=["POST"])
def set_pipeline():
    pipeline = request.json.get("pipeline", [])
    run_name = request.json.get("run")
    # Always update global config
    r.set("config:pipeline", json.dumps(pipeline))
    # Also update per-run snapshot if a run is specified
    if run_name and r.hexists("runs", run_name):
        r.set(f"run:{run_name}:pipeline", json.dumps(pipeline))
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


# -- Runs --------------------------------------------------------------------

import re as _re

@app.route("/api/runs")
def list_runs():
    """List all runs with metadata."""
    runs_raw = r.hgetall("runs")
    active = get_active_run()
    runs_list = []
    for rname, rval in runs_raw.items():
        try:
            meta = json.loads(rval)
            meta["file_count"] = r.hlen(f"run:{rname}:files")
            meta["queue_depth"] = r.llen(f"run:{rname}:queue")
            meta["active"] = (rname == active)
            runs_list.append(meta)
        except Exception:
            pass
    runs_list.sort(key=lambda x: x.get("created_at", ""))
    return jsonify({"runs": runs_list, "active_run": active})


@app.route("/api/runs", methods=["POST"])
def create_run():
    """Create a new named run. Snapshots current pipeline config."""
    name = request.json.get("name", "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400
    if not _re.match(r'^[a-zA-Z0-9_-]+$', name):
        return jsonify({"error": "name must be alphanumeric, hyphens, underscores only"}), 400
    if len(name) > 64:
        return jsonify({"error": "name too long (max 64 chars)"}), 400
    if r.hexists("runs", name):
        return jsonify({"error": f"run '{name}' already exists"}), 409

    # Snapshot the current global pipeline for this run
    keys = run_keys(name)
    pipeline_raw = r.get("config:pipeline") or "[]"
    r.set(keys["pipeline"], pipeline_raw)

    r.hset("runs", name, json.dumps({
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }))
    return jsonify({"ok": True, "name": name})


@app.route("/api/runs/activate", methods=["POST"])
def activate_run():
    """Set the active run."""
    name = request.json.get("name", "").strip()
    if not name or not r.hexists("runs", name):
        return jsonify({"error": "unknown run"}), 404
    r.set("config:active_run", name)
    return jsonify({"ok": True, "active_run": name})


@app.route("/api/runs/<name>", methods=["DELETE"])
def delete_run(name):
    """Delete a run and all its data."""
    if not r.hexists("runs", name):
        return jsonify({"error": "unknown run"}), 404
    if name == get_active_run():
        return jsonify({"error": "cannot delete the active run — switch to another first"}), 400

    keys = run_keys(name)
    if r.hlen(keys["processing"]) > 0:
        return jsonify({"error": "run has files currently being processed"}), 400

    r.delete(keys["files"], keys["processing"], keys["failures"], keys["queue"], keys["pipeline"])
    r.hdel("runs", name)
    return jsonify({"ok": True})


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

    run_name = request.args.get("run") or get_active_run()
    keys = run_keys(run_name)
    raw = r.hget(keys["files"], file_path)
    if not raw:
        abort(404)

    return jsonify(json.loads(raw))


# -- Review ------------------------------------------------------------------

@app.route("/api/review-queue")
def review_queue():
    """Get all files needing human review."""
    run_name = request.args.get("run") or get_active_run()
    files = get_all_files(run_name)
    needs_review = [f for f in files if f.get("status") == "needs_review"]
    return jsonify({
        "files": sorted(needs_review, key=lambda x: x.get("original_filename", "")),
    })


@app.route("/api/review", methods=["POST"])
def submit_review():
    """Submit a human review decision."""
    file_path = request.json.get("path")
    document_id = request.json.get("document_id")

    if not file_path or document_id is None:
        return jsonify({"error": "path and document_id required"}), 400

    run_name = get_active_run()
    keys = run_keys(run_name)
    raw = r.hget(keys["files"], file_path)
    if not raw:
        return jsonify({"error": "file not found"}), 404

    record = json.loads(raw)
    record["status"] = "reviewed"
    record["document_id"] = document_id
    record["reviewed_by"] = "human"
    record["reviewed_at"] = datetime.now(timezone.utc).isoformat()

    r.hset(keys["files"], file_path, json.dumps(record))
    return jsonify({"ok": True})


# -- Queue Management --------------------------------------------------------

@app.route("/api/queue/clear", methods=["POST"])
def clear_queue():
    run_name = get_active_run()
    keys = run_keys(run_name)
    r.delete(keys["queue"])
    return jsonify({"ok": True})


@app.route("/api/queue/remove", methods=["POST"])
def remove_from_queue():
    run_name = get_active_run()
    keys = run_keys(run_name)
    item = request.json.get("item")
    if item:
        r.lrem(keys["queue"], 0, item)
    return jsonify({"ok": True})


@app.route("/api/requeue-failed", methods=["POST"])
def requeue_failed():
    """Re-enqueue all errored files."""
    run_name = get_active_run()
    keys = run_keys(run_name)
    files = get_all_files(run_name)
    errored = [f for f in files if f.get("status") == "error"]
    count = 0
    for rec in errored:
        path = rec.get("path") or rec.get("_key")
        if path:
            r.hdel(keys["files"], path)
            r.hdel(keys["failures"], path)
            r.lpush(keys["queue"], path)
            count += 1
    return jsonify({"ok": True, "requeued": count})


@app.route("/api/requeue-file", methods=["POST"])
def requeue_file():
    """Re-run a specific file through the pipeline."""
    run_name = get_active_run()
    keys = run_keys(run_name)
    file_path = request.json.get("path")
    if not file_path:
        return jsonify({"error": "path required"}), 400
    r.hdel(keys["files"], file_path)
    r.hdel(keys["failures"], file_path)
    r.lpush(keys["queue"], file_path)
    return jsonify({"ok": True})


# -- Redis Keys --------------------------------------------------------------

@app.route("/api/keys")
def api_keys():
    """Dynamically discover all run-related and config keys."""
    discovered = set()
    # Config keys
    for key in ["config:pipeline", "config:active_run", "runs"]:
        discovered.add(key)
    # Per-run keys
    runs_raw = r.hgetall("runs")
    for rname in runs_raw:
        for suffix in ["files", "processing", "failures", "queue", "pipeline"]:
            discovered.add(f"run:{rname}:{suffix}")

    result = []
    for key in sorted(discovered):
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
        if size > 0 or key_type != "none":
            result.append({"key": key, "type": key_type, "size": size})
    return jsonify({"keys": result})


@app.route("/api/keys/view")
def view_key():
    """Return the contents of a Redis key."""
    key = request.args.get("key")
    if not key:
        return jsonify({"error": "key required"}), 400
    key_type = r.type(key)
    if key_type == "string":
        val = r.get(key)
        # Try to parse as JSON for pretty display
        try:
            val = json.loads(val)
        except (json.JSONDecodeError, TypeError):
            pass
        return jsonify({"key": key, "type": key_type, "value": val})
    elif key_type == "hash":
        raw = r.hgetall(key)
        parsed = {}
        for k, v in raw.items():
            try:
                parsed[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                parsed[k] = v
        return jsonify({"key": key, "type": key_type, "value": parsed})
    elif key_type == "list":
        raw = r.lrange(key, 0, 199)  # Cap at 200 items
        total = r.llen(key)
        parsed = []
        for v in raw:
            try:
                parsed.append(json.loads(v))
            except (json.JSONDecodeError, TypeError):
                parsed.append(v)
        return jsonify({"key": key, "type": key_type, "value": parsed, "total": total})
    elif key_type == "none":
        return jsonify({"error": "key does not exist"}), 404
    else:
        return jsonify({"key": key, "type": key_type, "value": f"(unsupported type: {key_type})"})


@app.route("/api/keys/delete", methods=["POST"])
def delete_key():
    key = request.json.get("key")
    if not key:
        return jsonify({"error": "key required"}), 400
    r.delete(key)
    return jsonify({"ok": True, "deleted": key})


@app.route("/api/keys/delete-all", methods=["POST"])
def delete_all_keys():
    """Delete all known keys (full reset)."""
    runs_raw = r.hgetall("runs")
    for rname in runs_raw:
        for suffix in ["files", "processing", "failures", "queue", "pipeline"]:
            r.delete(f"run:{rname}:{suffix}")
    r.delete("runs", "config:pipeline", "config:active_run")
    return jsonify({"ok": True})


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
