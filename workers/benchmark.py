"""
benchmark.py — Runs a sample of documents through both models sequentially,
measures timing, and writes a side-by-side comparison report.

Run with:
  docker compose --profile benchmark up benchmark

Note: Start ollama-qwen or chandra separately first, then point this
script at whichever is running via QWEN_URL / CHANDRA_URL env vars.
If only one model is running, the other column will show "unavailable".
"""

import os
import json
import base64
import time
import random
from pathlib import Path

import openai
import fitz
from PIL import Image
import io
from tabulate import tabulate

# ── Config ────────────────────────────────────────────────────────────────────
QWEN_URL    = os.environ.get("QWEN_URL", "http://ollama-qwen:8000/v1")
CHANDRA_URL = os.environ.get("CHANDRA_URL", "http://chandra:8001/v1")
INPUT_DIR   = Path(os.environ.get("INPUT_DIR", "/input"))
OUTPUT_DIR  = Path(os.environ.get("OUTPUT_DIR", "/output"))
SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", "10"))

SUPPORTED = {".pdf", ".jpg", ".jpeg", ".png"}

PROMPT = """Extract the primary license, certificate, or tracking number from this document.
Return ONLY valid JSON:
{
  "license_number": "the extracted number or null",
  "field_label": "what label was on that field",
  "confidence": 0.0
}"""

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def file_to_base64(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        doc = fitz.open(str(path))
        mat = fitz.Matrix(150 / 72, 150 / 72)
        pix = doc[0].get_pixmap(matrix=mat)
        return base64.b64encode(pix.tobytes("jpeg")).decode(), "image/jpeg"
    elif suffix in {".jpg", ".jpeg"}:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode(), "image/jpeg"
    elif suffix == ".png":
        img = Image.open(path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"


def run_model(client: openai.OpenAI, model_name: str, path: Path) -> dict:
    """Run a single file through a model, return result + timing."""
    image_data, media_type = file_to_base64(path)

    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=model_name,
            max_tokens=256,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{image_data}"},
                        },
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
        )
        elapsed = round(time.time() - t0, 2)
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        result = json.loads(raw)
        result["seconds"] = elapsed
        result["error"] = None
        return result

    except Exception as e:
        return {
            "license_number": None,
            "field_label": None,
            "confidence": 0.0,
            "seconds": round(time.time() - t0, 2),
            "error": str(e),
        }


def check_model_available(url: str) -> bool:
    try:
        client = openai.OpenAI(base_url=url, api_key="x", timeout=5.0)
        client.models.list()
        return True
    except Exception:
        return False


def main():
    print("=" * 60)
    print("OCR BENCHMARK RUNNER")
    print("=" * 60)

    # Discover files
    all_files = [f for f in INPUT_DIR.rglob("*") if f.suffix.lower() in SUPPORTED]
    if not all_files:
        print(f"No files found in {INPUT_DIR}. Mount your test docs there.")
        return

    # Sample
    sample = random.sample(all_files, min(SAMPLE_SIZE, len(all_files)))
    print(f"Sampling {len(sample)} of {len(all_files)} files\n")

    # Check which models are up
    qwen_available = check_model_available(QWEN_URL)
    chandra_available = check_model_available(CHANDRA_URL)

    print(f"Qwen    ({QWEN_URL}): {'✓ online' if qwen_available else '✗ offline'}")
    print(f"Chandra ({CHANDRA_URL}): {'✓ online' if chandra_available else '✗ offline'}")
    print()

    qwen_client = openai.OpenAI(base_url=QWEN_URL, api_key="x") if qwen_available else None
    chandra_client = openai.OpenAI(base_url=CHANDRA_URL, api_key="x") if chandra_available else None

    results = []
    table_rows = []

    for i, path in enumerate(sample, 1):
        print(f"[{i}/{len(sample)}] {path.name}")
        row = {"file": path.name, "path": str(path)}

        if qwen_client:
            print(f"  → Qwen...", end="", flush=True)
            q = run_model(qwen_client, "qwen-vl", path)
            row["qwen_license"] = q.get("license_number")
            row["qwen_confidence"] = q.get("confidence")
            row["qwen_seconds"] = q.get("seconds")
            row["qwen_error"] = q.get("error")
            print(f" {q.get('license_number')} ({q.get('seconds')}s)")
        else:
            row.update({"qwen_license": "N/A", "qwen_confidence": None, "qwen_seconds": None, "qwen_error": "offline"})

        if chandra_client:
            print(f"  → Chandra...", end="", flush=True)
            c = run_model(chandra_client, "chandra", path)
            row["chandra_license"] = c.get("license_number")
            row["chandra_confidence"] = c.get("confidence")
            row["chandra_seconds"] = c.get("seconds")
            row["chandra_error"] = c.get("error")
            print(f" {c.get('license_number')} ({c.get('seconds')}s)")
        else:
            row.update({"chandra_license": "N/A", "chandra_confidence": None, "chandra_seconds": None, "chandra_error": "offline"})

        # Agreement check
        row["agree"] = (
            row.get("qwen_license") == row.get("chandra_license")
            and row.get("qwen_license") is not None
        )

        results.append(row)
        table_rows.append([
            path.name[:40],
            row.get("qwen_license", "—"),
            f"{row.get('qwen_seconds', '—')}s",
            row.get("chandra_license", "—"),
            f"{row.get('chandra_seconds', '—')}s",
            "✓" if row["agree"] else "✗",
        ])

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(tabulate(
        table_rows,
        headers=["File", "Qwen Result", "Qwen Time", "Chandra Result", "Chandra Time", "Agree"],
        tablefmt="rounded_outline",
    ))

    # ── Stats ──────────────────────────────────────────────────────────────────
    qwen_times = [r["qwen_seconds"] for r in results if r.get("qwen_seconds")]
    chandra_times = [r["chandra_seconds"] for r in results if r.get("chandra_seconds")]
    agreements = sum(1 for r in results if r.get("agree"))

    print("\nSUMMARY")
    print(f"  Files tested:       {len(results)}")
    print(f"  Agreements:         {agreements}/{len(results)}")

    if qwen_times:
        print(f"  Qwen avg time:      {round(sum(qwen_times)/len(qwen_times), 2)}s")
        print(f"  Qwen total time:    {round(sum(qwen_times), 2)}s")

    if chandra_times:
        print(f"  Chandra avg time:   {round(sum(chandra_times)/len(chandra_times), 2)}s")
        print(f"  Chandra total time: {round(sum(chandra_times), 2)}s")

    # ── Save outputs ───────────────────────────────────────────────────────────
    report_path = OUTPUT_DIR / "benchmark_report.json"
    report_path.write_text(json.dumps(results, indent=2))
    print(f"\nFull report saved to: {report_path}")

    # CSV version
    csv_path = OUTPUT_DIR / "benchmark_report.csv"
    import csv
    fieldnames = ["file", "qwen_license", "qwen_confidence", "qwen_seconds",
                  "chandra_license", "chandra_confidence", "chandra_seconds", "agree", "path"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV saved to:        {csv_path}")


if __name__ == "__main__":
    main()
