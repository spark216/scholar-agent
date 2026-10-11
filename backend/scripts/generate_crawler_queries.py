"""Pre-pass: generate PaSa Crawler search queries for the benchmark queries.

Produces two sets per query (cached for the end-to-end A/B):
  - "initial":  official generate_query prompt
  - "adaptive": coverage-gap-targeted follow-up (direction-1: generates queries
    for aspects the initial set missed)

Output: .modeling-audit/crawler_queries_cache.json
  { "<qid>": {"query": ..., "initial": [...], "adaptive": [...]} }

Run: python -m backend.scripts.generate_crawler_queries
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GENERATE_QUERY = (
    "Please generate some mutually exclusive queries in a list to search the "
    "relevant papers according to the User Query. Searching for survey papers "
    "would be better.\nUser Query: {user_query}"
)
ADAPTIVE_QUERY = (
    "You are an elite researcher searching for academic papers on the following topic.\n"
    "User Query: {user_query}\n\n"
    "The following search queries have already been used and returned papers, but "
    "relevant papers are still missing. Generate additional mutually exclusive "
    "search queries that target aspects of the user query NOT yet covered by the "
    "previous queries. Searching for survey papers would be better.\n"
    "Previous queries:\n{queries}\n\n"
    "Output format:\n[Search]query1[\n[Search]query2[\n..."
)
SEARCH_TEMPLATE = re.compile(r"Search\](.*?)\[", flags=re.DOTALL)
CRAWLER_DIR = ROOT / "pasa-main" / "pasa-main" / "checkpoints" / "pasa-7b-crawler"
OUT = ROOT / ".modeling-audit" / "crawler_queries_cache.json"


def _safe_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="ascii", errors="replace")
    except Exception:  # noqa: BLE001
        pass


def main() -> None:
    _safe_stdout()
    if not (CRAWLER_DIR / "model.safetensors.index.json").exists():
        raise SystemExit(f"crawler model missing: {CRAWLER_DIR}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"loading crawler {CRAWLER_DIR} (4-bit)...", flush=True)
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(CRAWLER_DIR), local_files_only=True, padding_side="left"
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(CRAWLER_DIR),
        local_files_only=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        ),
        device_map="auto",
        low_cpu_mem_usage=True,
        max_memory={"cpu": "6GiB", 0: "11GiB"},
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    print(f"crawler loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    def generate(prompt: str) -> str:
        text = tokenizer.apply_chat_template(
            [{"content": prompt.strip(), "role": "user"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated = model.generate(**inputs, max_new_tokens=512)
        return tokenizer.batch_decode(
            generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )[0]

    test_path = ROOT / "backend" / "paper-test" / "datasets" / "RealScholarQuery" / "test.jsonl"
    samples = [json.loads(l) for l in test_path.open(encoding="utf-8") if l.strip()]
    indices = sorted(random.Random(42).sample(range(len(samples)), 5))
    samples = [samples[i] for i in indices]

    cache = {}
    with torch.inference_mode():
        for row in samples:
            qid = row["qid"]
            user_query = row["question"]

            t0 = time.perf_counter()
            initial = [
                q.strip()
                for q in SEARCH_TEMPLATE.findall(generate(GENERATE_QUERY.format(user_query=user_query)))
            ][:5]
            t1 = time.perf_counter()

            adaptive = [
                q.strip()
                for q in SEARCH_TEMPLATE.findall(
                    generate(
                        ADAPTIVE_QUERY.format(
                            user_query=user_query,
                            queries="\n".join(f"- {q}" for q in initial),
                        )
                    )
                )
            ][:5]
            t2 = time.perf_counter()

            cache[qid] = {
                "query": user_query,
                "initial": initial,
                "adaptive": adaptive,
            }
            print(f"=== {qid} (initial {t1 - t0:.1f}s, adaptive {t2 - t1:.1f}s) ===", flush=True)
            print("  initial:", flush=True)
            for q in initial:
                print(f"    - {q[:90]}", flush=True)
            print("  adaptive:", flush=True)
            for q in adaptive:
                print(f"    - {q[:90]}", flush=True)

    OUT.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
