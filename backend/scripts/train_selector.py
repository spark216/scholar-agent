"""Fine-tune bge-reranker as an academic Selector (PaSA-style relevance model).

Trains a LoRA adapter on top of the pinned bge-reranker-v2-m3 checkpoint so
the model becomes specialised at "is this paper relevant to this academic
query". Training data comes from build_selector_training_data.py (query,
gold paper -> 1; retrieved-but-not-gold paper -> 0).

Output: a PEFT adapter saved to --output (e.g. models/selector_lora) plus a
metrics.json with train/eval loss and accuracy.

Run: python -m backend.scripts.train_selector [--data ...] [--epochs 1] [--max-len 384]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _safe_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="ascii", errors="replace")
    except Exception:  # noqa: BLE001
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "backend/paper-test/datasets/selector_train.jsonl")
    parser.add_argument("--model", type=Path, default=ROOT / "models/bge-reranker-v2-m3-953dc6f6")
    parser.add_argument("--output", type=Path, default=ROOT / "models/selector_lora")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-len", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--limit", type=int, default=0, help="0 = all pairs")
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--resume", action="store_true", help="resume from the latest checkpoint")
    parser.add_argument("--save-steps", type=int, default=0, help="checkpoint every N steps (default: 0.25 epoch)")
    args = parser.parse_args()
    _safe_stdout()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    if not args.model.exists():
        raise SystemExit(f"model path not found: {args.model}")
    print(f"model: {args.model}", flush=True)

    pairs = []
    with args.data.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"pairs = {len(pairs)}", flush=True)
    random_state = __import__("random").Random(42)
    random_state.shuffle(pairs)
    val_count = max(1, int(len(pairs) * args.val_fraction))
    train_pairs = pairs[val_count:]
    val_pairs = pairs[:val_count]

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    if getattr(tokenizer, "pad_token", None) is None:
        # Qwen and similar tokenizers have no pad token; reuse eos so the
        # Trainer can pad batches.
        tokenizer.pad_token = tokenizer.eos_token

    def make_dataset(rows: list[dict]) -> Dataset:
        texts = [
            f"{row['query']} [SEP] {row['title']} {row['abstract']}"
            for row in rows
        ]
        labels = [float(row["label"]) for row in rows]
        enc = tokenizer(
            texts, padding=True, truncation=True, max_length=args.max_len,
            return_tensors=None,
        )
        return Dataset.from_dict({**{k: v for k, v in enc.items()}, "labels": labels})

    train_ds = make_dataset(train_pairs)
    val_ds = make_dataset(val_pairs)
    print(f"train = {len(train_ds)} val = {len(val_ds)}", flush=True)

    # Load in fp32 and let the Trainer's fp16 mixed precision handle
    # gradient scaling; pre-casting to fp16 breaks the AMP grad scaler.
    # attn_implementation="sdpa" avoids the very slow eager attention on
    # decoder-only models (Qwen) without flash-attn installed.
    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.model),
        local_files_only=True,
        num_labels=1,
        attn_implementation="sdpa",
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    class SelectorTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels").float()
            outputs = model(**inputs)
            logits = outputs.logits.view(-1).float()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            return (loss, outputs) if return_outputs else loss

    args_output = Path("backend/paper-test/selector_checkpoints")
    save_steps = (
        args.save_steps
        if args.save_steps > 0
        else max(1, int(len(train_ds) / (args.batch_size * 4)))
    )
    training_args = TrainingArguments(
        output_dir=str(args_output),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        fp16=True,
        logging_steps=200,
        eval_strategy="steps",
        eval_steps=save_steps,
        # Periodic checkpoints so an interrupted long run can resume.
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        report_to=[],
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    trainer = SelectorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )
    started = time.perf_counter()
    resume_checkpoint = None
    if args.resume:
        candidates = sorted(
            args_output.glob("checkpoint-*"),
            key=lambda path: int(path.name.split("-")[-1]),
        )
        if candidates:
            resume_checkpoint = str(candidates[-1])
            print(f"resuming from {resume_checkpoint}", flush=True)
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    train_seconds = time.perf_counter() - started

    args.output.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    metrics = trainer.evaluate()
    metrics.update(
        {
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "train_seconds": round(train_seconds, 1),
            "epochs": args.epochs,
            "lora_r": 16,
            "base_model": str(args.model),
        }
    )
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"trained in {train_seconds:.0f}s -> {args.output}")
    print(f"metrics: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
