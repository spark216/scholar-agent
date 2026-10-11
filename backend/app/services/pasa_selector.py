"""Optional PaSa 7B Selector — an additional academic-relevance signal fused
into EvidRank's weighted RRF.

Faithful to the official bytedance/pasa scoring protocol:
  - prompt = ``get_selected`` template (Title/Abstract/User Query +
    "Decision: True/False"), fed WITHOUT a chat template
  - score = softmax P("True") of the single greedy next token
    (``generate(max_new_tokens=1, do_sample=False)``)
  - P(True) > 0.5 is the official "selected" threshold

Disabled by default: loading the 7B costs ~5GB VRAM (4-bit) and ~38s, so this
is opt-in via ``PASA_SELECTOR_ENABLED``.  A missing bitsandbytes / CUDA
degrades gracefully to ``provider="unavailable"`` (no crash).
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.app.config import ROOT_DIR, get_settings

logger = logging.getLogger(__name__)

GET_SELECTED = (
    "You are an elite researcher in the field of AI, conducting research on {user_query}. "
    "Evaluate whether the following paper fully satisfies the detailed requirements of the "
    "user query and provide your reasoning. Ensure that your decision and reasoning are consistent.\n\n"
    "Searched Paper:\nTitle: {title}\nAbstract: {abstract}\n\n"
    "User Query: {user_query}\n\n"
    "Output format: Decision: True/False\nReason:... \nDecision:"
)


class PasaSelector:
    def __init__(self) -> None:
        settings = get_settings()
        self.enabled = bool(settings.pasa_selector_enabled)
        self.required = bool(settings.pasa_selector_required)
        self.model_path = settings.pasa_selector_model_path
        self.batch_size = max(1, settings.pasa_selector_batch_size)
        self.max_length = max(64, settings.pasa_selector_max_length)
        self.use_4bit = bool(settings.pasa_selector_use_4bit)
        self.max_cpu_memory_gb = max(1.0, settings.pasa_selector_max_cpu_memory_gb)
        self.max_gpu_memory_gb = max(1.0, settings.pasa_selector_max_gpu_memory_gb)
        self._tokenizer = None
        self._model = None
        self._torch = None
        self.device = "unresolved"
        self.resolved_model_path: str | None = None
        self.provider = "disabled" if not self.enabled else "unavailable"
        self._load_error: str | None = None
        self.last_inference_ms: float | None = None
        if self.enabled:
            self._load_model()

    def _resolve_model_path(self) -> str:
        path = Path(self.model_path)
        for candidate in (path, ROOT_DIR / path):
            try:
                if candidate.exists():
                    return str(candidate)
            except OSError:  # pragma: no cover - defensive
                continue
        return self.model_path  # lets HF snapshot_download resolve

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if self.use_4bit:
                import bitsandbytes  # noqa: F401 - ImportError guard
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                quantization_config = None

            path = self._resolve_model_path()
            self._tokenizer = AutoTokenizer.from_pretrained(
                path,
                local_files_only=True,
                padding_side="left",
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                path,
                local_files_only=True,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                # Bound the load: without the CPU cap the bf16 checkpoint
                # materialises in RAM (~13GB spike) and segfaults once the BGE
                # encoder/reranker are resident on a 16GB machine.
                max_memory={
                    "cpu": f"{self.max_cpu_memory_gb}GiB",
                    0: f"{self.max_gpu_memory_gb}GiB",
                },
                torch_dtype=torch.bfloat16,
            )
            self._model.eval()
            self._torch = torch
            self.provider = "pasa-7b-selector"
            self.device = str(next(self._model.parameters()).device)
            self.resolved_model_path = path
            logger.info(
                "Loaded PaSa selector from %s on %s (4bit=%s)",
                path,
                self.device,
                self.use_4bit,
            )
        except Exception as exc:  # noqa: BLE001
            self._tokenizer = None
            self._model = None
            self.provider = "unavailable"
            self._load_error = str(exc)
            logger.exception("Failed to load PaSa selector: %s", exc)

    def infer_scores(
        self,
        query: str,
        papers: list[dict[str, Any]],
    ) -> tuple[list[float] | None, dict[str, Any]]:
        """Score each paper's relevance to ``query``.

        Returns ``(scores | None, meta)`` — ``None`` scores mean the signal
        was not applied (disabled / unavailable / error).
        """
        started = time.perf_counter()
        if self._model is None or not self.enabled:
            if self.enabled and self.required:
                raise RuntimeError(
                    f"PaSa selector required but unavailable: {self._load_error or 'unknown error'}"
                )
            return None, {
                "provider": self.provider,
                "applied": False,
                "error": self._load_error,
                "scored_count": 0,
            }

        assert self._tokenizer is not None and self._torch is not None
        torch = self._torch
        prompts = [
            GET_SELECTED.format(
                user_query=query,
                title=str(paper.get("title") or ""),
                abstract=str(paper.get("abstract") or ""),
            )
            for paper in papers
        ]
        true_token_id = self._tokenizer.convert_tokens_to_ids("True")
        scores: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(prompts), self.batch_size):
                batch = prompts[start : start + self.batch_size]
                encoded = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = self._model.generate(
                    **encoded,
                    max_new_tokens=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                    do_sample=False,
                )
                probs = (
                    outputs.scores[0].softmax(dim=-1)[:, true_token_id].float().cpu().tolist()
                )
                scores.extend(float(value) for value in probs)
        self.last_inference_ms = (time.perf_counter() - started) * 1000
        return scores, {
            "provider": self.provider,
            "applied": True,
            "scored_count": len(scores),
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "device": self.device,
            "inference_ms": self.last_inference_ms,
        }


@lru_cache
def get_pasa_selector() -> PasaSelector:
    return PasaSelector()
