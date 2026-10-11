"""Sanity check for the downloaded pasa-7b-selector before the real eval.

Uses the official example from bytedance/pasa models.py __main__:
  query: smaller dataset in LLM pre-training -> better models
  paper: "Specialized Language Models with Cheap Inference from Limited
         Domain Data" (this paper IS a gold answer for that query)
Expectation: a sensible P(True) > 0.5 for the gold paper, and much lower for
an obviously irrelevant paper.

Run: python -m backend.scripts.sanity_pasa_selector
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

GET_SELECTED = (
    "You are an elite researcher in the field of AI, conducting research on {user_query}. "
    "Evaluate whether the following paper fully satisfies the detailed requirements of the "
    "user query and provide your reasoning. Ensure that your decision and reasoning are consistent.\n\n"
    "Searched Paper:\nTitle: {title}\nAbstract: {abstract}\n\n"
    "User Query: {user_query}\n\n"
    "Output format: Decision: True/False\nReason:... \nDecision:"
)

QUERY = (
    "Give me papers which shows that using a smaller dataset in large language model "
    "pre-training can result in better models than using bigger datasets."
)

PAPERS = [
    # gold (used in official example)
    (
        "Specialized Language Models with Cheap Inference from Limited Domain Data",
        "Large language models have emerged as a versatile tool but are challenging to apply to "
        "tasks lacking large inference budgets and large in-domain training sets. This work "
        "formalizes these constraints and distinguishes four important variables: the pretraining "
        "budget, the specialization budget, the inference budget, and the in-domain training set "
        "size. Across these settings, we compare different approaches from the machine learning "
        "literature. Limited by inference cost, we find better alternatives to the standard "
        "practice of training very large vanilla transformer models. In particular, we show that "
        "hyper-networks and mixture of experts have better perplexity for large pretraining "
        "budgets, while small models trained on importance sampled datasets are attractive for "
        "large specialization budgets.",
    ),
    # clearly irrelevant
    (
        "A Survey of Solar Panel Efficiency in Arid Climates",
        "We survey photovoltaic panel efficiency across arid regions, comparing silicon and "
        "perovskite technologies under high irradiance and dust accumulation conditions.",
    ),
]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="ascii", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    model_dir = ROOT / "models" / "pasa-7b-selector"
    weights = model_dir / "model.safetensors"
    if not weights.exists():
        raise SystemExit(f"model weights missing: {weights}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"loading {model_dir} (4-bit)...", flush=True)
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
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
    print(f"loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    true_token_id = tokenizer.convert_tokens_to_ids("True")
    prompts = [
        GET_SELECTED.format(user_query=QUERY, title=title, abstract=abstract)
        for title, abstract in PAPERS
    ]
    with torch.inference_mode():
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        outputs = model.generate(
            **encoded,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False,
        )
        probs = outputs.scores[0].softmax(dim=-1)
        for i, (title, _) in enumerate(PAPERS):
            p = probs[i, true_token_id].item()
            print(f"  P(True)={p:.4f}  {title[:70]}", flush=True)

    del model, tokenizer
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    print("sanity done", flush=True)


if __name__ == "__main__":
    main()
