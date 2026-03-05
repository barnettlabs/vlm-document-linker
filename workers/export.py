"""
export.py — Exports Redis results hashes to CSV files.
Run anytime to get current progress snapshot.
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

FIELDNAMES = [
    "original_filename",
    "license_number",
    "field_label",
    "confidence",
    "model",
    "inference_seconds",
    "path",
    "error",
]

for model in ["qwen", "chandra"]:
    key = f"results:{model}"
    data = r.hgetall(key)

    if not data:
        print(f"No results for {model} yet.")
        continue

    records = [json.loads(v) for v in data.values()]
    out_path = OUTPUT_DIR / f"mapping_{model}.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    success = sum(1 for r_ in records if not r_.get("error"))
    failed = len(records) - success

    print(f"{model}: {len(records)} total | {success} success | {failed} failed → {out_path}")

# Queue stats
pending = r.llen("queue:pending")
print(f"\nQueue still pending: {pending}")
