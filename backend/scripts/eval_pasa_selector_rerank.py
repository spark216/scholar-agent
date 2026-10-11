"""Standalone test: does the official PaSa 7B Selector improve ranking /
recall of the host project's candidate pool, compared to the current
bge-reranker signal and to the historical best full-pipeline run?

Method (faithful to bytedance/pasa):
  - selector prompt = agent_prompt.json "get_selected" template
    (Title / Abstract / User Query / "Decision: True/False ... Decision:")
  - score = softmax P("True") of the single greedy next token
    (infer_score), NOT a chat-template call.
  - > 0.5 is the official "selected" threshold.

Rankings evaluated on the SAME pool (built by build_selector_pool.py):
  selector | reranker | rrf-fusion | sel-filter->rerank | sel-topN->rerank
Metrics (mirror host paper_test.py): P/R/F1/nDCG/MRR @5/10/20, macro avg.

Run:
  python -m backend.scripts.eval_pasa_selector_rerank
  python -m backend.scripts.eval_pasa_selector_rerank --skip-selector   # reranker only
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GET_SELECTED = (
    "You are an elite researcher in the field of AI, conducting research on {user_query}. "
    "Evaluate whether the following paper fully satisfies the detailed requirements of the "
    "user query and provide your reasoning. Ensure that your decision and reasoning are consistent.\n\n"
    "Searched Paper:\nTitle: {title}\nAbstract: {abstract}\n\n"
    "User Query: {user_query}\n\n"
    "Output format: Decision: True/False\nReason:... \nDecision:"
)


def _safe_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="ascii", errors="replace")
    except Exception:  # noqa: BLE001
        pass


# --- identity matching + metrics (mirror host paper_test.py) --------------
def normalize_arxiv_id(value) -> str:
    normalized = str(value or "").strip().casefold()
    normalized = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", normalized)
    normalized = re.sub(r"^arxiv:\s*", "", normalized)
    normalized = normalized.removesuffix(".pdf")
    return re.sub(r"v\d+$", "", normalized)


def normalize_title(value) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def gold_records(gold_arxiv_ids, gold_titles):
    records = []
    for index in range(max(len(gold_arxiv_ids), len(gold_titles))):
        raw_id = gold_arxiv_ids[index] if index < len(gold_arxiv_ids) else ""
        raw_title = gold_titles[index] if index < len(gold_titles) else ""
        arxiv_id = normalize_arxiv_id(raw_id)
        title = normalize_title(raw_title)
        if arxiv_id or title:
            records.append((arxiv_id, title, str(raw_title or raw_id)))
    return records


def relevance_flags(papers, gold_arxiv_ids, gold_titles):
    gold = gold_records(gold_arxiv_ids, gold_titles)
    unmatched = set(range(len(gold)))
    flags, matches = [], []
    for paper in papers:
        arxiv_id = normalize_arxiv_id(paper.get("arxiv_id"))
        title = normalize_title(paper.get("title"))
        matched_index = next(
            (i for i in unmatched if arxiv_id and gold[i][0] and arxiv_id == gold[i][0]),
            None,
        )
        if matched_index is None:
            matched_index = next(
                (i for i in unmatched if title and gold[i][1] and title == gold[i][1]),
                None,
            )
        flags.append(1 if matched_index is not None else 0)
        if matched_index is not None:
            unmatched.remove(matched_index)
            matches.append(gold[matched_index][2])
    return flags, matches


def metric_at_k(flags, relevant_count, k):
    selected = flags[:k]
    hits = sum(selected)
    precision = hits / k
    recall = hits / relevant_count if relevant_count else 0.0
    denom = precision + recall
    f1 = 2 * precision * recall / denom if denom else 0.0
    dcg = sum((1 / math.log2(rank + 1)) for rank, flag in enumerate(selected, start=1) if flag)
    ideal_hits = min(relevant_count, k)
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
    first_hit = next((rank for rank, flag in enumerate(selected, start=1) if flag), None)
    mrr = (1 / first_hit) if first_hit else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "ndcg": ndcg, "mrr": mrr, "hits": hits}


def rrf_rank(lists_of_papers, k=60):
    """lists_of_papers: list of ranked paper lists (dicts with arxiv_id)."""
    scores: dict[str, float] = {}
    for ranked in lists_of_papers:
        for rank, paper in enumerate(ranked, start=1):
            key = paper["arxiv_id"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    order = sorted(scores, key=lambda key: -scores[key])
    by_id = {p["arxiv_id"]: p for ranked in lists_of_papers for p in ranked}
    return [by_id[key] for key in order]


# --- ranking variants ------------------------------------------------------
def rank_selector(papers):
    return sorted(papers, key=lambda p: -(float(p.get("sel_score") or 0.0)))


def rank_reranker(papers):
    return sorted(papers, key=lambda p: -(float(p.get("cross_logit") or 0.0)))


def rank_sel_filter_rerank(papers):
    """Official PaSa flow: >0.5 selected, then ordered by relevance score."""
    kept = [p for p in papers if float(p.get("sel_score") or 0.0) > 0.5]
    return rank_selector(kept if kept else papers)


def rank_sel_filter_rerank_signal(papers):
    """Filter >0.5, then order by the host bge-reranker (hybrid)."""
    kept = [p for p in papers if float(p.get("sel_score") or 0.0) > 0.5]
    return rank_reranker(kept if kept else papers)


def rank_sel_topn_rerank(papers, top_n=50):
    top = rank_selector(papers)[:top_n]
    return rank_reranker(top)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=str, default=str(ROOT / ".modeling-audit" / "selector_test_pool.json"))
    parser.add_argument("--out", type=str, default=str(ROOT / ".modeling-audit" / "selector_rerank_results.json"))
    parser.add_argument("--model", type=str, default=str(ROOT / "models" / "pasa-7b-selector-sharded"))
    parser.add_argument("--skip-selector", action="store_true")
    parser.add_argument("--selector-batch", type=int, default=8)
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    args = parser.parse_args()
    _safe_stdout()

    data = json.loads(Path(args.pool).read_text(encoding="utf-8"))
    queries = data["queries"]
    print(f"loaded pool: {len(queries)} queries, pool={data['pool']}", flush=True)

    already_scored = all(
        "sel_score" in p for q in queries for p in q["pool"]
    ) and len(queries) > 0

    # --- score with pasa-7b-selector ------------------------------------
    if args.skip_selector:
        print("skipping selector scoring (--skip-selector)", flush=True)
    elif already_scored:
        print("sel_scores already present in pool, skipping selector scoring", flush=True)
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"loading selector {args.model} (4-bit)...", flush=True)
        t0 = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, local_files_only=True, padding_side="left"
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            ),
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model.eval()
        print(f"selector loaded in {time.perf_counter() - t0:.1f}s", flush=True)

        true_token_id = tokenizer.convert_tokens_to_ids("True")
        prompts_by_q = []
        for q in queries:
            prompts = [
                GET_SELECTED.format(
                    user_query=q["query"],
                    title=p.get("title", ""),
                    abstract=p.get("abstract", ""),
                )
                for p in q["pool"]
            ]
            prompts_by_q.append(prompts)

        with torch.inference_mode():
            for qi, (q, prompts) in enumerate(zip(queries, prompts_by_q, strict=True)):
                t0 = time.perf_counter()
                scores: list[float] = []
                for start in range(0, len(prompts), args.selector_batch):
                    batch = prompts[start:start + args.selector_batch]
                    encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
                    encoded = {k: v.to(model.device) for k, v in encoded.items()}
                    outputs = model.generate(
                        **encoded,
                        max_new_tokens=1,
                        output_scores=True,
                        return_dict_in_generate=True,
                        do_sample=False,
                    )
                    probs = outputs.scores[0].softmax(dim=-1)[:, true_token_id].float().cpu().tolist()
                    scores.extend(float(v) for v in probs)
                for p, s in zip(q["pool"], scores, strict=True):
                    p["sel_score"] = s
                print(
                    f"  {q['qid']}: scored {len(scores)} papers in {time.perf_counter() - t0:.1f}s, "
                    f"mean P(True)={sum(scores) / len(scores):.3f}",
                    flush=True,
                )
        del model, tokenizer
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        Path(args.pool).write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )
        print("sel_scores saved back to pool file", flush=True)

    # --- score with the host bge-reranker (current ranking signal) --------
    from backend.app.services.reranker import get_bge_reranker

    reranker = get_bge_reranker()
    print(f"reranker provider: {reranker.provider}", flush=True)
    for q in queries:
        t0 = time.perf_counter()
        pairs = [[q["query"], f"{p.get('title','')}\n{p.get('abstract','')}"] for p in q["pool"]]
        logits, sigmoids = reranker._score_pairs(pairs)
        for p, logit, sigmoid in zip(q["pool"], logits, sigmoids, strict=True):
            p["cross_logit"] = logit
            p["cross_sigmoid"] = sigmoid
        print(f"  {q['qid']}: reranked {len(q['pool'])} papers in {time.perf_counter() - t0:.1f}s", flush=True)

    # --- evaluate ranking variants ----------------------------------------
    variants = {
        "selector": rank_selector,
        "reranker": rank_reranker,
        "sel_filter_rerank": rank_sel_filter_rerank,
        "sel_filter_rerank_signal": rank_sel_filter_rerank_signal,
        "sel_topN_rerank": lambda papers: rank_sel_topn_rerank(papers, 50),
    }
    pool_gold = 0
    total_gold = 0
    for q in queries:
        total_gold += len(q["gold_titles"])
        pool_gold += sum(
            relevance_flags(q["pool"], q["gold_arxiv_ids"], q["gold_titles"])[0]
        )

    print(f"\n=== pool recall ceiling: {pool_gold}/{total_gold} "
          f"({pool_gold / total_gold * 100:.1f}%) ===", flush=True)

    # RRF fusion (uses selector + reranker ranks)
    rrf_results = {}
    for q in queries:
        sel_ranked = rank_selector(q["pool"])
        rr_ranked = rank_reranker(q["pool"])
        rrf_results[q["qid"]] = rrf_rank([sel_ranked, rr_ranked])

    summary = {}
    for name, fn in variants.items():
        per_query = []
        for q in queries:
            ranked = fn(q["pool"])
            flags, matches = relevance_flags(ranked, q["gold_arxiv_ids"], q["gold_titles"])
            entry = {
                "qid": q["qid"],
                "hits_total": sum(flags),
                "matched": matches,
                "metrics": {str(k): metric_at_k(flags, len(gold_records(q["gold_arxiv_ids"], q["gold_titles"])), k) for k in args.ks},
            }
            per_query.append(entry)
            print(
                f"  [{name}] {q['qid']}: hits={sum(flags)} "
                f"MRR@5={entry['metrics']['5']['mrr']:.3f} R@10={entry['metrics']['10']['recall']:.3f}",
                flush=True,
            )
        agg = {}
        for k in args.ks:
            agg[str(k)] = {m: sum(q["metrics"][str(k)][m] for q in per_query) / len(per_query)
                           for m in ["precision", "recall", "f1", "ndcg", "mrr"]}
            agg[str(k)]["hits"] = sum(q["metrics"][str(k)]["hits"] for q in per_query)
        summary[name] = {"per_query": per_query, "agg": agg}
        print(
            f"  [{name}] @5: P={agg['5']['precision']:.3f} R={agg['5']['recall']:.3f} "
            f"F1={agg['5']['f1']:.3f} nDCG={agg['5']['ndcg']:.3f} MRR={agg['5']['mrr']:.3f} hits={agg['5']['hits']}",
            flush=True,
        )
        print(
            f"  [{name}] @20: P={agg['20']['precision']:.3f} R={agg['20']['recall']:.3f} "
            f"F1={agg['20']['f1']:.3f} hits={agg['20']['hits']}",
            flush=True,
        )

    # RRF fusion eval (needs pool papers keyed by qid)
    rrf_per_query = []
    for q in queries:
        ranked = rrf_results[q["qid"]]
        flags, matches = relevance_flags(ranked, q["gold_arxiv_ids"], q["gold_titles"])
        rrf_per_query.append({
            "qid": q["qid"],
            "hits_total": sum(flags),
            "matched": matches,
            "metrics": {str(k): metric_at_k(flags, len(gold_records(q["gold_arxiv_ids"], q["gold_titles"])), k) for k in args.ks},
        })
    agg = {}
    for k in args.ks:
        agg[str(k)] = {m: sum(q["metrics"][str(k)][m] for q in rrf_per_query) / len(rrf_per_query)
                       for m in ["precision", "recall", "f1", "ndcg", "mrr"]}
        agg[str(k)]["hits"] = sum(q["metrics"][str(k)]["hits"] for q in rrf_per_query)
    summary["rrf_fusion"] = {"per_query": rrf_per_query, "agg": agg}
    print(
        f"  [rrf_fusion] @5: P={agg['5']['precision']:.3f} R={agg['5']['recall']:.3f} "
        f"F1={agg['5']['f1']:.3f} nDCG={agg['5']['ndcg']:.3f} MRR={agg['5']['mrr']:.3f} hits={agg['5']['hits']}",
        flush=True,
    )

    # selector filter diagnostics (official >0.5 threshold)
    print("\n=== selector threshold diagnostics (P(True)>0.5) ===", flush=True)
    for q in queries:
        sel = [p for p in q["pool"] if float(p.get("sel_score") or 0.0) > 0.5]
        flags, matches = relevance_flags(sel, q["gold_arxiv_ids"], q["gold_titles"])
        print(f"  {q['qid']}: selected {len(sel)}/{len(q['pool'])} papers, "
              f"{sum(flags)} gold among selected, precision={sum(flags) / len(sel) if sel else 0:.3f}", flush=True)

    out = Path(args.out)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved: {out}", flush=True)


if __name__ == "__main__":
    main()
