# OCR Benchmark Stack

Test Qwen2.5-VL-7B and Chandra OCR on your 3060 Ti GPUs, one at a time.

## Prerequisites

- Docker + Docker Compose with NVIDIA runtime
- `nvidia-container-toolkit` installed
- HuggingFace account (for model downloads — set `HF_TOKEN` env var if needed)

## Directory Structure

```
vlm-document-linker/
├── docker-compose.yml
├── input/              ← Drop your test PDFs/JPGs/PNGs here
├── output/
│   ├── qwen/           ← Per-file JSON results from Qwen
│   ├── chandra/        ← Per-file JSON results from Chandra
│   ├── benchmark_report.json
│   └── benchmark_report.csv
└── workers/
    ├── Dockerfile
    ├── requirements.txt
    ├── enqueue.py
    ├── worker.py
    ├── benchmark.py
    └── export.py
```

## Usage

### 1. Drop test files into /input

```bash
cp /your/test/docs/*.pdf ./input/
cp /your/test/docs/*.jpg ./input/
```

### 2. Test Qwen (one 3060 Ti)

```bash
# Start Qwen vLLM + worker
docker compose --profile qwen up

# In another terminal, enqueue your files
docker compose --profile workers run enqueuer
```

Watch the worker terminal for results. First run downloads the model (~8GB).

### 3. Test Chandra (same card, swap profiles)

```bash
# Stop Qwen first to free the GPU
docker compose --profile qwen down

# Start Chandra vLLM + worker
docker compose --profile chandra up

# Enqueue again (skips already-processed files)
docker compose --profile workers run enqueuer
```

### 4. Run benchmark (both models, sequential)

Requires BOTH vLLM servers running simultaneously.
Only use this if you have 2 GPUs available (or enough VRAM for both).

```bash
docker compose --profile qwen --profile chandra up -d
docker compose --profile benchmark run benchmark
```

### 5. Export results to CSV

```bash
docker compose --profile export run exporter
# Output: output/mapping_qwen.csv, output/mapping_chandra.csv
```

## Monitoring

```bash
# Live queue depth
watch -n 2 'docker exec vdl-redis redis-cli llen queue:pending'

# Results count
docker exec vdl-redis redis-cli hlen results:qwen
docker exec vdl-redis redis-cli hlen results:chandra

# See failures
docker exec vdl-redis redis-cli hgetall failures:qwen

# Worker logs
docker compose logs -f worker-qwen
docker compose logs -f worker-chandra
```

## Web Dashboard

The Flask dashboard (`web/app.py`) auto-refreshes every 5 seconds from a single
`GET /api/status` endpoint that reads all state from Redis.

### Stat Cards (top row)

| Card | How it's computed |
|------|-------------------|
| **Queue Pending** | `LLEN queue:pending` — direct count of items waiting to be processed |
| **Total Completed** | Sum of non-error records from `results:qwen` + `results:chandra` |
| **Total Failed** | Sum of records with an `error` field from both models |
| **Qwen Avg Time** | Mean of `inference_seconds` across successful Qwen results |
| **Chandra Avg Time** | Mean of `inference_seconds` across successful Chandra results |
| **Qwen Avg Confidence** | Mean of `confidence` across successful Qwen results |
| **Chandra Avg Confidence** | Mean of `confidence` across successful Chandra results |

### Inference Timing Comparison (bar chart)

- Collects `inference_seconds` from successful results for both models, keyed by `original_filename`
- Shows side-by-side horizontal bars per file (blue = Qwen, purple = Chandra)
- Bar width is proportional to the max time across both models
- Hidden when no results exist

### Pending Queue

- Shows up to the first 50 items from `queue:pending` (Redis list)
- Each item has a **Remove** button (`LREM`)
- **Clear Queue** button deletes the entire list

### Model Results (two side-by-side panels)

Each panel (Qwen 2.5-VL-7B / Chandra 9B) shows:
- Header counts: succeeded / failed / total
- **Retry Failed** button (appears when failures > 0) — deletes the failed record
  from `results:<model>` and `failures:<model>`, then `LPUSH`es the path back
  into `queue:pending`
- Scrollable results table: File, License #, Confidence, Time, Status (OK/FAIL)
- Results are sorted alphabetically by `original_filename`

### Redis Keys (reset panel)

At the bottom of the dashboard, the **Redis Keys** panel lists all known keys:
- `queue:pending` — the pending work queue
- `results:qwen` / `results:chandra` — result hashes per model
- `failures:qwen` / `failures:chandra` — failure retry counters

Each key shows its type, entry count, and a **Delete** button to clear it individually.
The **Delete All (Full Reset)** button wipes all five keys at once.

After deleting results, re-run `enqueue.py` to re-populate the queue since it
skips files that already have results.

### Queue behavior

- **Removing an item** from the queue (`LREM`) only removes it from the pending
  list. There is no automatic re-scan that would re-add it.
- **Re-enqueueing** happens only via `enqueue.py` (skips files already in
  `results:qwen` or `results:chandra`) or the dashboard's "Retry Failed" button.
- A removed file will **not** be picked up again unless you manually re-run
  `enqueue.py` (and its result records don't exist) or use "Retry Failed."

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/status` | Returns all dashboard data (queue depth, queue items, per-model stats + results) |
| POST | `/api/queue/clear` | Deletes the entire `queue:pending` list |
| POST | `/api/queue/remove` | Removes a specific item from the queue (body: `{"item": "<path>"}`) |
| POST | `/api/requeue-failed` | Moves all failed results for a model back to the queue (body: `{"model": "qwen\|chandra"}`) |
| GET | `/api/keys` | Lists all known Redis keys with type and entry count |
| POST | `/api/keys/delete` | Deletes a specific Redis key (body: `{"key": "<key>"}`) |
| POST | `/api/keys/delete-all` | Deletes all known Redis keys (full reset) |

## Tuning Notes

### Chandra on 3060 Ti (8GB)
Chandra's base model is BF16 (~18GB). The docker-compose uses bitsandbytes
4-bit quantization. If it OOMs, try reducing `--max-model-len` to 2048
or `--max-num-seqs` to 1.

Alternatively, use a pre-quantized GGUF via llama.cpp instead of vLLM —
see: https://huggingface.co/noctrex/Chandra-OCR-GGUF

### Qwen on 3060 Ti (8GB)
AWQ 4-bit fits cleanly. If you have issues, reduce `--gpu-memory-utilization`
from 0.90 to 0.85.

### DPI vs Speed tradeoff
In worker.py, `150 / 72` is the DPI multiplier for PDF rasterization.
- Lower (100 DPI): faster, may miss fine text
- Higher (200 DPI): slower, better for dense forms
- 150 DPI: good default for clean government forms like MDE 330

## Output Format

Each file produces a JSON result:
```json
{
  "license_number": "667299",
  "field_label": "Inspection Certificate No",
  "confidence": 0.95,
  "original_filename": "A_G-_Colonial_Square-_LP.pdf",
  "path": "/input/A_G-_Colonial_Square-_LP.pdf",
  "model": "qwen-vl",
  "inference_seconds": 4.2,
  "error": null
}
```

The final mapping CSVs contain one row per file across all processed documents.
