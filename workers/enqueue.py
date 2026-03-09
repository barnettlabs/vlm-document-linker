"""
enqueue.py -- Scans /input for supported files and pushes them into Redis.
Safe to re-run: skips files already processed in the active run.
"""

import os
import json
import redis
from pathlib import Path
from datetime import datetime, timezone

SUPPORTED = {".pdf", ".jpg", ".jpeg", ".png"}
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
INPUT_DIR = os.environ.get("INPUT_DIR", "/input")

r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)


def run_keys(run_name: str) -> dict:
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
        r.hset("runs", "default", json.dumps({
            "name": "default",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }))
        r.set("config:active_run", "default")
        print("Migrated legacy keys to run:default:*")
    elif not r.exists("runs"):
        r.hset("runs", "default", json.dumps({
            "name": "default",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }))
        r.set("config:active_run", "default")


migrate_legacy_keys()

active_run = get_active_run()
keys = run_keys(active_run)

input_path = Path(INPUT_DIR)
files = sorted([f for f in input_path.rglob("*") if f.suffix.lower() in SUPPORTED])

print(f"Found {len(files)} files in {INPUT_DIR}")
print(f"Active run: {active_run}")

enqueued = 0
skipped = 0

for f in files:
    path_str = str(f)

    # Skip if already processed in this run
    if r.hexists(keys["files"], path_str):
        skipped += 1
        continue

    # Only add if not already in queue
    if not r.lpos(keys["queue"], path_str):
        r.lpush(keys["queue"], path_str)
        enqueued += 1

print(f"Enqueued: {enqueued} | Skipped (already done): {skipped}")
print(f"Queue depth: {r.llen(keys['queue'])}")
