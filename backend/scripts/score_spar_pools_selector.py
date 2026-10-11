"""Score SPAR candidate pools with the official pasa-7b-selector.

Faithful to eval_pasa_selector_rerank.py: GET_SELECTED template, greedy
single-token generation, softmax P("True").  Differences:
  * per-query checkpointing (pool file rewritten after every query, so a
    killed chunk resumes without re-scoring)
  * larger GPU batch (default 16; --batch to tune) so the RTX 3060 is fed
    more tokens per forward pass
  * --start/--limit chunking for long pool files

Run (agent env, chunks so each foreground call stays bounded):
  python -m backend.scripts.score_spar_pools_selector --pool .modeling-audit/spar_baseline_pool.json --start 0 --limit 12
  python -m backend.scripts.score_spar_pools_selector --pool .modeling-audit/spar_baseline_pool.json --start 12 --limit 24
  ... then --start 24/36/48 ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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
# Sharded layout: single-file 15.2GB safetensors crashes on this machine
# (torch.storage access violation during meta-model load); 4 x 3.8GB shards
# load reliably. Built by shard_selector_model.py.
MODEL_DIR = ROOT / "models" / "pasa-7b-selector-sharded"


def _safe_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="ascii", errors="replace")
    except Exception:  # noqa: BLE001
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=str, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="process queries [start, start+limit)")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=512, help="truncate prompt tokens (abstract tail adds prefill, little signal)")
    parser.add_argument("--model", type=str, default=str(MODEL_DIR))
    args = parser.parse_args()
    _safe_stdout()

    pool_path = Path(args.pool)
    data = json.loads(pool_path.read_text(encoding="utf-8"))
    queries = data["queries"]
    if args.limit:
        queries = queries[args.start:args.start + args.limit]
    else:
        queries = queries[args.start:]
    print(
        f"pool={pool_path.name} scoring {len(queries)} queries "
        f"(start={args.start}, batch={args.batch})",
        flush=True,
    )

    to_score = [
        q for q in queries
        if not q["pool"] or not all("sel_score" in p for p in q["pool"])
    ]
    if not to_score:
        print("all queries already scored, nothing to do", flush=True)
        return

    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"loading selector {args.model} (4-bit, GPU)...", flush=True)
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, padding_side="left"
    )
    # 4-bit load of the 15.2GB single-file safetensors is FLAKY on this machine
    # (occasional Windows access violation in torch.storage while materializing
    # the meta model). Retry across the two historically-working parameter
    # patterns instead of dying on the first attempt.
    model = None
    for attempt in range(1, 4):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
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
            break
        except Exception as exc:  # noqa: BLE001
            print(f"  load attempt {attempt} failed: {type(exc).__name__}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            if attempt == 3:
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

    with torch.inference_mode():
        for qi, q in enumerate(to_score, start=1):
            qid = q["qid"]
            prompts = [
                GET_SELECTED.format(
                    user_query=q["query"],
                    title=p.get("title", ""),
                    abstract=p.get("abstract", ""),
                )
                for p in q["pool"]
            ]
            t0 = time.perf_counter()
            scores: list[float] = []
            for start in range(0, len(prompts), args.batch):
                batch = prompts[start:start + args.batch]
                encoded = tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=args.max_len,
                )
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
            pool_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            print(
                f"[{args.start + qi}/{len(data['queries'])}] {qid}: scored "
                f"{len(scores)} papers in {time.perf_counter() - t0:.1f}s "
                f"(mean P(True)={sum(scores) / len(scores):.3f})",
                flush=True,
            )

    del model, tokenizer
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    print(f"done -> {pool_path}", flush=True)


if __name__ == "__main__":
    main()
