"""
export.py — Exports Redis file results to CSV.
Run anytime to get a current snapshot: make export
"""

import os
import json
import csv
import redis
from pathlib import Path

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))

r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Determine which run to export
run_name = r.get("config:active_run") or "default"
files_key = f"run:{run_name}:files"

# Fetch all file records
data = r.hgetall(files_key)

if not data:
    print(f"No results yet for run '{run_name}'.")
    exit(0)

records = []
for path_key, val in data.items():
    try:
        rec = json.loads(val)
    except json.JSONDecodeError:
        continue

    passes = rec.get("passes", [])
    successful = [p for p in passes if not p.get("error")]
    total_time = sum(p.get("inference_seconds", 0) or 0 for p in passes)
    total_tokens = sum(p.get("total_tokens", 0) or 0 for p in passes)

    # Collect all found values across passes
    found_values = [
        p.get("certificate_number")
        for p in successful
        if p.get("found") and p.get("certificate_number")
    ]
    all_values = ", ".join(dict.fromkeys(found_values)) if found_values else ""

    # Best confidence from found passes
    confidences = [p.get("confidence", 0) for p in successful if p.get("found")]
    best_confidence = max(confidences) if confidences else None

    # Models used
    models_used = ", ".join(dict.fromkeys(p.get("model", "") for p in passes if p.get("model")))

    records.append({
        "filename": rec.get("original_filename", ""),
        "path": rec.get("path", path_key),
        "doc_type": rec.get("doc_type", ""),
        "status": rec.get("status", ""),
        "document_id": rec.get("document_id", ""),
        "all_values": all_values,
        "best_confidence": round(best_confidence, 3) if best_confidence is not None else "",
        "passes": len(passes),
        "errors": sum(1 for p in passes if p.get("error")),
        "total_time_s": round(total_time, 2) if total_time else "",
        "total_tokens": total_tokens or "",
        "models": models_used,
        "reviewed_by": rec.get("reviewed_by", ""),
        "reviewed_at": rec.get("reviewed_at", ""),
        "error": rec.get("error", ""),
    })

# Sort by filename
records.sort(key=lambda r: r["filename"])

FIELDNAMES = [
    "filename",
    "path",
    "doc_type",
    "status",
    "document_id",
    "all_values",
    "best_confidence",
    "passes",
    "errors",
    "total_time_s",
    "total_tokens",
    "models",
    "reviewed_by",
    "reviewed_at",
    "error",
]

out_path = OUTPUT_DIR / f"export_{run_name}.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(records)

# Summary
statuses = {}
for rec in records:
    s = rec["status"]
    statuses[s] = statuses.get(s, 0) + 1

print(f"Exported {len(records)} files from run '{run_name}' to {out_path}")
for status, count in sorted(statuses.items()):
    print(f"  {status}: {count}")

queue_key = f"run:{run_name}:queue"
pending = r.llen(queue_key)
if pending:
    print(f"\nQueue still pending: {pending}")
