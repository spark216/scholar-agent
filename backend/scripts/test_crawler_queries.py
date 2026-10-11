"""Minimal test of the official PaSa Crawler query-generation function.

Faithful to bytedance/pasa run_paper_agent.py / paper_agent.py:
  - prompt  = agent_prompt.json "generate_query" template
  - infer() = chat-template user turn, generate(max_new_tokens=512)
  - parse   = re.findall(r"Search\\](.*?)\\[", response)

For each benchmark query we then measure candidate-recall (gold-in-pool) using
the generated queries through the same local FTS + faiss-vector retrieval the
candidate-recall diagnostic uses, and compare against the known baseline
(29/82 gold in pool with the lexical-fallback understanding).

Run: python -m backend.scripts.test_crawler_queries
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
SEARCH_TEMPLATE = re.compile(r"Search\](.*?)\[", flags=re.DOTALL)
CRAWLER_DIR = (
    ROOT / "pasa-main" / "pasa-main" / "checkpoints" / "pasa-7b-crawler"
)


def _safe_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="ascii", errors="replace")
    except Exception:  # noqa: BLE001
        pass


def _key(value: str) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFKD", str(value or "")).casefold()
    return "".join(c for c in normalized if c.isalnum())


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

    test_path = ROOT / "backend" / "paper-test" / "datasets" / "RealScholarQuery" / "test.jsonl"
    samples = [json.loads(l) for l in test_path.open(encoding="utf-8") if l.strip()]
    indices = sorted(random.Random(42).sample(range(len(samples)), 5))
    samples = [samples[i] for i in indices]

    results = []
    gold_total = 0
    crawler_gold = 0
    with torch.inference_mode():
        for row in samples:
            qid = row["qid"]
            user_query = row["question"]
            gold = [g for g in (row.get("answer") or []) if g]
            gold_ids = [g for g in (row.get("answer_arxiv_id") or []) if g]
            gold_total += len(gold)
            gold_title_keys = {_key(g) for g in gold}
            gold_id_keys = {(str(a or "").casefold()) for a in gold_ids}

            prompt = GENERATE_QUERY.format(user_query=user_query).strip()
            text = tokenizer.apply_chat_template(
                [{"content": prompt, "role": "user"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            t0 = time.perf_counter()
            generated = model.generate(**inputs, max_new_tokens=512)
            gen = generated[:, inputs.input_ids.shape[1]:]
            response = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
            gen_s = time.perf_counter() - t0
            queries = [
                q.strip()
                for q in SEARCH_TEMPLATE.findall(response)
            ][:5]
            print(f"\n=== {qid} (生成耗时 {gen_s:.1f}s) ===", flush=True)
            print("crawler queries:", flush=True)
            for q in queries:
                print(f"  - {q[:100]}", flush=True)

            # ---- candidate recall: baseline vs crawler vs fused (union) ---
            # The crawler query set is AUXILIARY: it is fused into the existing
            # query method (LLM-intent variants), never a replacement.
            from backend.app.models.schemas import QueryUnderstanding
            from backend.app.services.local_pasa_corpus import search_local_pasa_corpus
            from backend.app.services.local_vector_corpus import search_local_vector_corpus

            def pool_from_understanding(understanding) -> tuple[set, set]:
                fts_papers, _ = search_local_pasa_corpus(
                    user_query, understanding, limit=200, offset=0
                )
                vec_papers, _ = search_local_vector_corpus(
                    user_query, understanding, limit=200, offset=0
                )
                titles = {_key(p.get("title")) for p in fts_papers + vec_papers}
                ids = {
                    str(p.get("id") or p.get("arxiv_id") or "").casefold()
                    for p in fts_papers + vec_papers
                }
                return titles, ids

            def count_gold(titles, ids) -> int:
                return len(gold_title_keys & titles) + len(gold_id_keys & ids)

            # baseline: the lexical-fallback understanding the candidate-recall
            # diagnostic uses (same as eval_pasa_real_candidate_recall.py).
            base_understanding = QueryUnderstanding(
                intent=f"检索与 {user_query} 相关的学术论文",
                is_paper_search=True,
                language="zh+en",
                topics=[user_query],
                keywords=[user_query],
                sub_queries=[user_query],
                rewritten_queries=[user_query, f"{user_query} academic papers"],
            )
            bt, bi = pool_from_understanding(base_understanding)
            base_found = count_gold(bt, bi)

            # crawler-only: the generated queries as the sole variant source.
            crawler_understanding = QueryUnderstanding(
                intent=f"检索与 {user_query} 相关的学术论文",
                is_paper_search=True,
                language="zh+en",
                topics=[user_query],
                keywords=list(queries) or [user_query],
                sub_queries=list(queries) or [user_query],
                rewritten_queries=list(queries) or [user_query],
            )
            ct, ci = pool_from_understanding(crawler_understanding)
            crawler_only = count_gold(ct, ci)

            # fused: union of baseline pool and crawler pool (the actual
            # production shape — crawler queries ADD variants on top).
            fused_found = count_gold(bt | ct, bi | ci)
            crawler_gold += fused_found
            print(
                f"  baseline pool: gold_in_pool={base_found}/{len(gold)}  "
                f"crawler-only={crawler_only}/{len(gold)}  "
                f"FUSED={fused_found}/{len(gold)} (+{fused_found - base_found})",
                flush=True,
            )
            results.append(
                {
                    "qid": qid,
                    "queries": queries,
                    "gold": len(gold),
                    "baseline": base_found,
                    "crawler_only": crawler_only,
                    "fused": fused_found,
                }
            )

    print("\n=== 汇总（融合 = 基线 ∪ crawler 查询）===", flush=True)
    print(f"FUSED gold-in-pool: {crawler_gold}/{gold_total} = {crawler_gold / gold_total * 100:.1f}%", flush=True)
    print("baseline (fallback understanding, FTS+vec): 29/82 = 35.4%", flush=True)
    print("per-query baseline: Q1=10 Q7=8 Q17=5 Q40=3 Q47=3", flush=True)
    for r in results:
        print(
            f"  {r['qid']}: baseline={r['baseline']} crawler_only={r['crawler_only']} "
            f"FUSED={r['fused']} (+{r['fused'] - r['baseline']})",
            flush=True,
        )

    out = ROOT / ".modeling-audit" / "crawler_queries_test.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
