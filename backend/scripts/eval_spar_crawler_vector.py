"""SPARBench full crawler-vector candidate-recall test.

Extends backend/scripts/test_crawler_queries.py from a 5-sample
RealScholarQuery subset to the full 50-query SPAR benchmark
(dataset/SPAR_Benchmark.jsonl).

For each SPAR query:
  * generate initial crawler search queries (pasa-7b-crawler, official
    generate_query prompt; initial-only) -- checkpointed per query to a cache
    file so interrupted generation chunks resume cleanly
  * measure gold-in-pool through the fused local FTS + faiss-vector retrieval:
      baseline (lexical-fallback understanding) vs crawler-only vs FUSED

Gold identity: normalized title-key OR arXiv-ID extracted from paperID
(bare arXiv id / arxiv.org URL).  Unlike test_crawler_queries.py, pool IDs are
taken from external_ids.arxiv / arxiv_id (the "id" field is a stable sha256,
never a raw arXiv id).

Modes (chunk-friendly, resumes via cache):
  python -m backend.scripts.eval_spar_crawler_vector --mode generate --limit 12
  python -m backend.scripts.eval_spar_crawler_vector --mode recall
  python -m backend.scripts.eval_spar_crawler_vector              # both
"""

from __future__ import annotations

import argparse
import json
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
SEARCH_TEMPLATE = re.compile(r"Search\](.*?)\[", flags=re.DOTALL)
CRAWLER_DIR = ROOT / "pasa-main" / "pasa-main" / "checkpoints" / "pasa-7b-crawler"
SPAR_PATH = ROOT / "dataset" / "SPAR_Benchmark.jsonl"
CACHE = ROOT / ".modeling-audit" / "spar_crawler_queries_cache.json"
OUT = ROOT / ".modeling-audit" / "spar_crawler_vector_test.json"
_ARXIV_RE = re.compile(r"^\d{4}\.\d{4,5}$")


def _safe_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="ascii", errors="replace")
    except Exception:  # noqa: BLE001
        pass


def _key(value: str) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFKD", str(value or "")).casefold()
    return "".join(c for c in normalized if c.isalnum())


def _norm_arxiv(pid: str) -> str:
    p = str(pid or "").strip().casefold()
    p = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", p)
    p = p.removesuffix(".pdf")
    return re.sub(r"v\d+$", "", p)


def load_spar(limit: int) -> list[dict]:
    """Load SPARBench rows; assign sequential qids, extract gold titles and
    arXiv ids from source_meta.answers[].paperID."""
    samples: list[dict] = []
    with SPAR_PATH.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            qid = f"SPARBench_{line_number}"
            gold_answers: list[dict] = []
            for answer in (value.get("source_meta") or {}).get("answers") or []:
                title = str(answer.get("title") or "")
                title_key = _key(title)
                pid = str(answer.get("paperID") or "")
                arxiv_id = ""
                if "arxiv" in pid.lower() or _ARXIV_RE.match(pid):
                    arxiv_id = _norm_arxiv(pid)
                gold_answers.append(
                    {"title": title, "title_key": title_key, "arxiv_id": arxiv_id}
                )
            samples.append(
                {
                    "qid": qid,
                    "question": str(value.get("question") or ""),
                    "gold_answers": gold_answers,
                }
            )
    if limit and limit < len(samples):
        samples = samples[:limit]
    return samples


def _load_crawler_cache() -> dict:
    if CACHE.is_file():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    return {}


def _save_crawler_cache(cache: dict) -> None:
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_queries(samples: list[dict], cache: dict) -> dict:
    """Run pasa-7b-crawler generate_query for samples missing from the cache;
    checkpoint after every query so a killed chunk resumes without losing work."""
    missing = [s for s in samples if s["qid"] not in cache]
    if not missing:
        print("no missing queries in cache", flush=True)
        return cache
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

    with torch.inference_mode():
        for index, sample in enumerate(missing, start=1):
            qid = sample["qid"]
            user_query = sample["question"]
            t0 = time.perf_counter()
            queries = [
                q.strip()
                for q in SEARCH_TEMPLATE.findall(
                    generate(GENERATE_QUERY.format(user_query=user_query))
                )
            ][:5]
            elapsed = time.perf_counter() - t0
            cache[qid] = {
                "query": user_query,
                "initial": queries,
            }
            _save_crawler_cache(cache)
            print(
                f"[{index}/{len(missing)}] {qid} ({elapsed:.1f}s): "
                + "; ".join(q[:60] for q in queries),
                flush=True,
            )
    return cache


def _pool_from_understanding(user_query: str, understanding) -> tuple[set, set]:
    from backend.app.services.local_pasa_corpus import search_local_pasa_corpus
    from backend.app.services.local_vector_corpus import search_local_vector_corpus

    fts_papers, _ = search_local_pasa_corpus(
        user_query, understanding, limit=200, offset=0
    )
    vec_papers, _ = search_local_vector_corpus(
        user_query, understanding, limit=200, offset=0
    )

    def arxiv_id(paper: dict) -> str:
        return str(
            paper.get("arxiv_id")
            or (paper.get("external_ids") or {}).get("arxiv")
            or ""
        ).casefold()

    titles = {_key(p.get("title")) for p in fts_papers + vec_papers}
    ids = {arxiv_id(p) for p in fts_papers + vec_papers if arxiv_id(p)}
    return titles, ids


def _existing_report() -> dict:
    if OUT.is_file():
        try:
            return json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def run_recall(samples: list[dict], cache: dict, *, start: int = 0) -> dict:
    from backend.app.models.schemas import QueryUnderstanding

    # Resume: keep per-query rows already computed in a previous chunk.
    previous = _existing_report()
    results = list(previous.get("per_query") or [])
    done = {row["qid"] for row in results}
    gold_total = previous.get("gold_total", 0)
    found_total = dict(previous.get("found") or {"baseline": 0, "crawler_only": 0, "fused": 0})
    t0 = time.perf_counter()
    for index, sample in enumerate(samples, start=1):
        if sample["qid"] in done:
            continue
        qid = sample["qid"]
        user_query = sample["question"]
        gold_answers = sample["gold_answers"]
        gold_ids = {a["arxiv_id"].casefold() for a in gold_answers if a["arxiv_id"]}
        queries = (cache.get(qid) or {}).get("initial") or []
        gold_total += len(gold_answers)

        def count(titles, ids) -> int:
            # Distinct gold answers found: title-key OR arxiv-id match, so a
            # paper that hits both namespaces is never double counted.
            return sum(
                1
                for answer in gold_answers
                if answer["title_key"] in titles
                or (answer["arxiv_id"] and answer["arxiv_id"] in ids)
            )

        base_understanding = QueryUnderstanding(
            intent=f"检索与 {user_query} 相关的学术论文",
            is_paper_search=True,
            language="zh+en",
            topics=[user_query],
            keywords=[user_query],
            sub_queries=[user_query],
            rewritten_queries=[user_query, f"{user_query} academic papers"],
        )
        bt, bi = _pool_from_understanding(user_query, base_understanding)
        base_found = count(bt, bi)

        if queries:
            crawler_understanding = QueryUnderstanding(
                intent=f"检索与 {user_query} 相关的学术论文",
                is_paper_search=True,
                language="zh+en",
                topics=[user_query],
                keywords=list(queries) or [user_query],
                sub_queries=list(queries) or [user_query],
                rewritten_queries=list(queries) or [user_query],
            )
            ct, ci = _pool_from_understanding(user_query, crawler_understanding)
            crawler_only = count(ct, ci)
            fused_found = count(bt | ct, bi | ci)
        else:
            ct, ci = set(), set()
            crawler_only = 0
            fused_found = base_found

        found_total["baseline"] += base_found
        found_total["crawler_only"] += crawler_only
        found_total["fused"] += fused_found

        fused_gold_titles = sorted(
            str(a.get("title"))
            for a in gold_answers
            if a["title_key"] in (bt | ct)
            or (a["arxiv_id"] and a["arxiv_id"] in (bi | ci))
        )
        results.append(
            {
                "qid": qid,
                "question": user_query[:200],
                "gold": len(gold_answers),
                "gold_arxiv": len(gold_ids),
                "queries": queries,
                "baseline": base_found,
                "crawler_only": crawler_only,
                "fused": fused_found,
                "fused_gold_titles": fused_gold_titles,
            }
        )
        print(
            f"[{start + index}/{len(samples) + start}] {qid}: base={base_found} "
            f"crawler={crawler_only} FUSED={fused_found} "
            f"(gold {len(gold_answers)})",
            flush=True,
        )
    elapsed = time.perf_counter() - t0

    report = {
        "experiment": "SPARBench full crawler-vector candidate-recall",
        "date": "2026-08-26",
        "dataset": str(SPAR_PATH),
        "queries": len(samples),
        "gold_total": gold_total,
        "found": found_total,
        "recall_pct": {
            k: (v / gold_total * 100 if gold_total else 0)
            for k, v in found_total.items()
        },
        "elapsed_s": round(elapsed, 1),
        "per_query": results,
        "note": (
            "gold_total = distinct gold answers (title-key OR arxiv-id match). "
            "Corpus ceiling: only 201/317 arXiv-identified gold papers are in "
            "the local 568k corpus, so absolute recall is capped well below 100%."
        ),
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved -> {OUT}", flush=True)
    print(f"gold_total={gold_total} elapsed={elapsed:.1f}s", flush=True)
    for k, v in found_total.items():
        print(f"  {k}: {v} ({v / gold_total * 100:.1f}%)", flush=True)
    return report


def main() -> None:
    _safe_stdout()
    parser = argparse.ArgumentParser(description="SPARBench crawler-vector candidate recall")
    parser.add_argument("--mode", choices=["generate", "recall", "both"], default="both")
    parser.add_argument("--limit", type=int, default=0, help="first-N SPAR queries (0=all 50)")
    parser.add_argument("--start", type=int, default=0, help="skip the first N SPAR queries (recall chunking)")
    args = parser.parse_args()

    samples = load_spar(args.limit)
    if args.start:
        samples = samples[args.start:]
    print(f"loaded {len(samples)} SPAR queries (start={args.start})", flush=True)

    cache = _load_crawler_cache()
    if args.mode in ("generate", "both"):
        cache = generate_queries(samples, cache)
        _save_crawler_cache(cache)
    if args.mode in ("recall", "both"):
        run_recall(samples, cache, start=args.start)


if __name__ == "__main__":
    main()
