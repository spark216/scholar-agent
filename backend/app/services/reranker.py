"""Auditable BGE Cross-Encoder reranker.

Raw logits and sigmoid-transformed values are stored separately.  The sigmoid
value is not a calibrated relevance probability and is never used as such.
"""

from __future__ import annotations

import logging
import math
import time
import hashlib
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any

from backend.app.config import get_settings

logger = logging.getLogger(__name__)


def _resolve_device(preferred: str, torch: Any) -> str:
    value = (preferred or "auto").lower()
    if value == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "mps":
        return "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    if value == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BGEReranker:
    def __init__(self) -> None:
        settings = get_settings()
        self.enabled = bool(settings.bge_reranker_enabled)
        self.required = bool(settings.bge_reranker_required)
        self.model_path = settings.bge_reranker_model_path
        self.model_revision = settings.bge_reranker_model_revision or None
        self.cache_dir = settings.bge_model_cache_dir or None
        self.local_files_only = bool(settings.bge_local_files_only)
        self.device_preference = settings.bge_reranker_device
        self.use_fp16 = bool(settings.bge_reranker_use_fp16)
        self.batch_size = settings.bge_reranker_batch_size
        self.max_length = settings.bge_reranker_max_length
        self.top_n = settings.bge_reranker_top_n
        self.cache_size = max(0, settings.bge_reranker_cache_size)
        self._tokenizer = None
        self._model = None
        self._torch = None
        self.device = "unresolved"
        self.resolved_model_path: str | None = None
        self.provider = "disabled" if not self.enabled else "unavailable"
        self._load_error: str | None = None
        self._score_cache: OrderedDict[str, float] = OrderedDict()
        self._inference_lock = RLock()
        if self.enabled:
            self._load_model()

    def _cache_key(self, query: str, document: str) -> str:
        model_identity = getattr(self, "model_revision", None) or self.model_path
        payload = f"{model_identity}\x1f{self.max_length}\x1f{query}\x1f{document}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load_model(self) -> None:
        try:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model_path = self.model_path
            if not Path(model_path).exists():
                model_path = snapshot_download(
                    repo_id=self.model_path,
                    revision=self.model_revision,
                    cache_dir=self.cache_dir,
                    local_files_only=self.local_files_only,
                )
            self.resolved_model_path = str(model_path)
            self.device = _resolve_device(self.device_preference, torch)
            dtype = torch.float16 if self.use_fp16 and self.device == "cuda" else torch.float32
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                local_files_only=True,
                torch_dtype=dtype,
            )
            # Optional fine-tuned Selector adapter (academic relevance LoRA).
            selector_path = str(get_settings().bge_reranker_selector_path or "").strip()
            if selector_path and Path(selector_path).exists():
                from peft import PeftModel

                self._model = PeftModel.from_pretrained(self._model, selector_path)
                self.provider = "bge-reranker+selector"
                logger.info("Loaded Selector LoRA adapter from %s", selector_path)
            else:
                self.provider = "bge-reranker"
            self._model.to(self.device)
            self._model.eval()
            self._torch = torch
            logger.info(
                "Loaded BGE reranker from %s (revision=%s) on %s",
                model_path,
                self.model_revision or "unresolved",
                self.device,
            )
        except Exception as exc:  # noqa: BLE001
            self._tokenizer = None
            self._model = None
            self.provider = "unavailable"
            self._load_error = str(exc)
            logger.exception("Failed to load BGE reranker: %s", exc)

    def _score_pairs(self, pairs: list[list[str]]) -> tuple[list[float], list[float]]:
        assert self._tokenizer is not None and self._model is not None and self._torch is not None
        logits_out: list[float] = []
        torch = self._torch
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start:start + self.batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                logits = self._model(**inputs, return_dict=True).logits.view(-1).float().cpu().tolist()
            if isinstance(logits, float):
                logits = [logits]
            logits_out.extend(float(value) for value in logits)
        sigmoid = [
            1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value))))
            for value in logits_out
        ]
        return logits_out, sigmoid

    @staticmethod
    def _fallback_order(papers: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        return sorted(
            (dict(paper) for paper in papers),
            key=lambda value: (
                -float(value.get("rrf_score") or 0.0),
                str(value.get("id") or ""),
            ),
        )[:top_k]

    def rerank(
        self,
        query: str,
        papers: list[dict[str, Any]],
        top_k: int,
        *,
        candidate_limit: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not papers:
            return [], {
                "reranker": self.provider,
                "model_revision": self.model_revision or "unresolved",
                "applied": False,
                "cross_input_count": 0,
                "returned_count": 0,
            }

        if self._model is None or not self.enabled:
            if self.enabled and self.required:
                raise RuntimeError(
                    "BGE reranker is required but unavailable: "
                    f"{self._load_error or 'unknown load error'}"
                )
            fallback = self._fallback_order(papers, top_k)
            return fallback, {
                "reranker": self.provider,
                "model_revision": self.model_revision or "unresolved",
                "applied": False,
                "error": self._load_error,
                "cross_input_count": 0,
                "returned_count": len(fallback),
                "batch_size": self.batch_size,
                "device": self.device,
            }

        candidate_n = min(
            len(papers),
            max(1, candidate_limit) if candidate_limit is not None else self.top_n,
        )
        subset = papers[:candidate_n]
        pairs = [[query, f"{paper.get('title', '')}\n{paper.get('abstract', '')}"] for paper in subset]
        started = time.perf_counter()
        try:
            cache_keys = [self._cache_key(pair[0], pair[1]) for pair in pairs]
            lock = getattr(self, "_inference_lock", RLock())
            with lock:
                missing: dict[str, list[str]] = {}
                for key, pair in zip(cache_keys, pairs, strict=True):
                    if key not in self._score_cache and key not in missing:
                        missing[key] = pair
                call_results: dict[str, float] = {}
                if missing:
                    missing_keys = list(missing)
                    missing_logits, _ = self._score_pairs(
                        [missing[key] for key in missing_keys]
                    )
                    for key, logit in zip(missing_keys, missing_logits, strict=True):
                        call_results[key] = logit
                        if self.cache_size:
                            self._score_cache[key] = logit
                            self._score_cache.move_to_end(key)
                            while len(self._score_cache) > self.cache_size:
                                self._score_cache.popitem(last=False)
                logits: list[float] = []
                for key in cache_keys:
                    if key in self._score_cache:
                        logits.append(self._score_cache[key])
                        self._score_cache.move_to_end(key)
                    else:
                        logits.append(call_results[key])
            sigmoid = [
                1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value))))
                for value in logits
            ]
            scored: list[dict[str, Any]] = []
            for paper, logit, display_value in zip(subset, logits, sigmoid, strict=True):
                item = dict(paper)
                item["cross_logit"] = logit
                item["cross_sigmoid"] = display_value
                scored.append(item)
            scored.sort(
                key=lambda value: (
                    -float(value.get("cross_logit") or 0.0),
                    str(value.get("id") or ""),
                )
            )
            # Only the configured pool is Cross-Encoded. Preserve the original
            # RRF tail so limiting inference does not silently reduce recall.
            tail = [dict(paper) for paper in papers[candidate_n:]]
            returned = (scored + tail)[:top_k]
            return returned, {
                "reranker": "bge-reranker",
                "model_revision": self.model_revision or "unresolved",
                "resolved_model_path": self.resolved_model_path,
                "applied": True,
                "candidates": len(scored),
                "cross_input_count": len(scored),
                "returned_count": len(returned),
                "batch_size": self.batch_size,
                "max_length": self.max_length,
                "device": self.device,
                "inference_ms": (time.perf_counter() - started) * 1000,
                "cache_hits": len(pairs) - len(missing),
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("Rerank failed: %s", exc)
            if self.required:
                raise RuntimeError(f"BGE reranker inference failed: {exc}") from exc
            fallback = self._fallback_order(papers, top_k)
            return fallback, {
                "reranker": "failed",
                "model_revision": self.model_revision or "unresolved",
                "applied": False,
                "error": str(exc),
                "cross_input_count": 0,
                "returned_count": len(fallback),
                "batch_size": self.batch_size,
                "device": self.device,
            }


@lru_cache
def get_bge_reranker() -> BGEReranker:
    return BGEReranker()
