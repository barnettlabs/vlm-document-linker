"""
worker.py — Multi-model pipeline worker.

Pulls files from Redis queue, runs them through all configured models
in the pipeline, performs triage, and stores unified results.

Configured via environment variables:
  REDIS_HOST              Redis hostname (default: localhost)
  QUEUE_NAME              Redis list to pull from
  WORKER_ID               Label for logging
  OUTPUT_DIR              Directory to write per-file JSON results
  AUTO_ACCEPT_THRESHOLD   Confidence threshold for auto-accept (default: 0.85)
"""

import os
import json
import base64
import re
import time
from datetime import datetime, timezone
import redis
import openai
import fitz  # pymupdf
from PIL import Image
from pathlib import Path
import io

# -- Config ------------------------------------------------------------------
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
QUEUE_NAME = os.environ.get("QUEUE_NAME", "queue:pending")
WORKER_ID = os.environ.get("WORKER_ID", "worker")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))
MAX_RETRIES = 3
AUTO_ACCEPT_THRESHOLD = float(os.environ.get("AUTO_ACCEPT_THRESHOLD", "0.85"))

PROMPT = """/nothink
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
}"""

DEFAULT_PIPELINE = [
    {"name": "qwen3-vl:2b", "base_url": "http://ollama-gpu0:11434/v1", "label": "Qwen3-VL 2B"},
    {"name": "glm-ocr", "base_url": "http://ollama-gpu1:11434/v1", "label": "GLM-OCR 0.9B"},
]

# -- Redis -------------------------------------------------------------------
r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_pipeline():
    """Get pipeline config from Redis, or set and return default."""
    raw = r.get("config:pipeline")
    if raw:
        return json.loads(raw)
    r.set("config:pipeline", json.dumps(DEFAULT_PIPELINE))
    return DEFAULT_PIPELINE


# -- File handling -----------------------------------------------------------
def file_to_base64(path: Path) -> tuple[str, str]:
    """Convert any supported file to base64 JPEG for vision model ingestion."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        doc = fitz.open(str(path))
        mat = fitz.Matrix(150 / 72, 150 / 72)
        pix = doc[0].get_pixmap(matrix=mat)
        data = pix.tobytes("jpeg")
        return base64.b64encode(data).decode(), "image/jpeg"

    elif suffix in {".jpg", ".jpeg"}:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode(), "image/jpeg"

    elif suffix == ".png":
        img = Image.open(path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"

    else:
        raise ValueError(f"Unsupported file type: {suffix}")


# -- Model calling -----------------------------------------------------------
def call_model(path: Path, model_config: dict) -> dict:
    """Send image to a model endpoint and parse the JSON result."""
    client = openai.OpenAI(base_url=model_config["base_url"], api_key="not-needed")
    image_data, media_type = file_to_base64(path)

    t0 = time.time()
    response = client.chat.completions.create(
        model=model_config["name"],
        max_tokens=2048,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_data}"
                        },
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
    )
    elapsed = round(time.time() - t0, 2)

    raw = response.choices[0].message.content or ""
    raw = raw.strip()
    print(f"[{WORKER_ID}]   Raw response: {raw[:500]}", flush=True)

    # Capture token usage
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0

    # Strip <think>...</think> reasoning blocks (Qwen 3.5 thinking mode)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    result = json.loads(raw)
    result["model"] = model_config["name"]
    result["inference_seconds"] = elapsed
    result["prompt_tokens"] = prompt_tokens
    result["completion_tokens"] = completion_tokens
    result["total_tokens"] = prompt_tokens + completion_tokens
    result["error"] = None
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    return result


# -- Triage ------------------------------------------------------------------
def run_triage(passes: list[dict]) -> tuple[str, str | None]:
    """Determine if results should be auto-accepted or need human review."""
    successful = [p for p in passes if not p.get("error")]
    if not successful:
        return "needs_review", None

    numbers = [p.get("certificate_number") for p in successful if p.get("certificate_number")]
    confidences = [p.get("confidence", 0) for p in successful]

    # If the field wasn't found by any model, flag for review
    not_found = all(p.get("found") is False for p in successful)
    if not numbers or not_found:
        return "needs_review", None

    all_agree = len(set(numbers)) == 1
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    if all_agree and avg_confidence >= AUTO_ACCEPT_THRESHOLD:
        return "auto_accepted", numbers[0]
    else:
        return "needs_review", None


# -- Processing --------------------------------------------------------------
def process_file(path_str: str) -> dict:
    """Run a file through all pipeline models and triage the results."""
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path_str}")

    pipeline = get_pipeline()
    passes = []

    for model_config in pipeline:
        label = model_config.get("label", model_config["name"])
        print(f"[{WORKER_ID}]   Running {label}...", flush=True)

        try:
            result = call_model(path, model_config)
            passes.append(result)
            print(
                f"[{WORKER_ID}]   {label}: "
                f"{result.get('certificate_number')} "
                f"(found: {result.get('found')}, "
                f"confidence: {result.get('confidence')}, "
                f"{result.get('inference_seconds')}s)",
                flush=True,
            )
        except Exception as e:
            passes.append({
                "model": model_config["name"],
                "license_number": None,
                "field_label": None,
                "confidence": 0.0,
                "inference_seconds": None,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            print(f"[{WORKER_ID}]   {label}: FAILED - {e}", flush=True)

    status, final_value = run_triage(passes)

    return {
        "original_filename": path.name,
        "path": path_str,
        "passes": passes,
        "status": status,
        "final_value": final_value,
        "reviewed_by": None,
        "reviewed_at": None,
    }


# -- Main loop ---------------------------------------------------------------
print(f"[{WORKER_ID}] Worker started. Listening on queue: {QUEUE_NAME}", flush=True)
print(f"[{WORKER_ID}] Pipeline: {[m.get('label', m['name']) for m in get_pipeline()]}", flush=True)

while True:
    item = r.brpop(QUEUE_NAME, timeout=30)

    if item is None:
        depth = r.llen(QUEUE_NAME)
        if depth == 0:
            print(f"[{WORKER_ID}] Queue empty. Waiting...", flush=True)
        continue

    _, path_str = item
    filename = Path(path_str).name
    print(f"[{WORKER_ID}] Processing {filename}...", flush=True)

    # Track in-progress
    r.hset("processing", path_str, json.dumps({
        "worker": WORKER_ID,
        "filename": filename,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }))

    try:
        record = process_file(path_str)

        # Save to Redis
        r.hdel("processing", path_str)
        r.hset("files", path_str, json.dumps(record))

        # Save individual JSON file
        out_file = OUTPUT_DIR / f"{Path(path_str).stem}_result.json"
        out_file.write_text(json.dumps(record, indent=2))

        print(
            f"[{WORKER_ID}] -> {filename} -> {record['status']} "
            f"(final: {record['final_value']}, {len(record['passes'])} passes)",
            flush=True,
        )

    except Exception as e:
        r.hdel("processing", path_str)
        failure_count = r.hincrby("failures", path_str, 1)

        if failure_count < MAX_RETRIES:
            r.lpush(QUEUE_NAME, path_str)
            print(
                f"[{WORKER_ID}] x {filename} failed "
                f"(attempt {failure_count}/{MAX_RETRIES}): {e} -- requeued",
                flush=True,
            )
        else:
            error_record = {
                "original_filename": filename,
                "path": path_str,
                "passes": [],
                "status": "error",
                "final_value": None,
                "reviewed_by": None,
                "reviewed_at": None,
                "error": str(e),
            }
            r.hset("files", path_str, json.dumps(error_record))

            out_file = OUTPUT_DIR / f"{Path(path_str).stem}_result.json"
            out_file.write_text(json.dumps(error_record, indent=2))

            print(
                f"[{WORKER_ID}] x {filename} permanently failed "
                f"after {MAX_RETRIES} attempts: {e}",
                flush=True,
            )
