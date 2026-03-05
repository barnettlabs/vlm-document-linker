"""
enqueue.py -- Scans /input for supported files and pushes them into Redis.
Safe to re-run: skips files already in the files hash.
"""

import os
import redis
from pathlib import Path

SUPPORTED = {".pdf", ".jpg", ".jpeg", ".png"}
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
INPUT_DIR = os.environ.get("INPUT_DIR", "/input")

r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)

input_path = Path(INPUT_DIR)
files = sorted([f for f in input_path.rglob("*") if f.suffix.lower() in SUPPORTED])

print(f"Found {len(files)} files in {INPUT_DIR}")

enqueued = 0
skipped = 0

for f in files:
    path_str = str(f)

    # Skip if already processed
    if r.hexists("files", path_str):
        skipped += 1
        continue

    # Only add if not already in queue
    if not r.lpos("queue:pending", path_str):
        r.lpush("queue:pending", path_str)
        enqueued += 1

print(f"Enqueued: {enqueued} | Skipped (already done): {skipped}")
print(f"Queue depth: {r.llen('queue:pending')}")
