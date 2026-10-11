"""Local BGE-M3 encoder: dense semantic + sparse lexical vectors."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any

from backend.app.config import get_settings

logger = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)

# This project is PyTorch-only.  Disabling optional TensorFlow/Flax discovery
# avoids a Keras 3 vs transformers import conflict pulled in by some broad
# scientific Python environments.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")


def _resolve_device(preferred: str) -> str:
    pref = (preferred or "auto").lower()
    try:
        import torch

        if pref == "cpu":
            return "cpu"
        if pref == "mps":
            return "mps" if torch.backends.mps.is_available() else "cpu"
        if pref == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        return "cpu"
    return "cpu"


def _fallback_dense(text: str, dim: int) -> list[float]:
    vec = [0.0] * dim
    tokens = _TOKEN_RE.findall((text or "").lower()) or ["empty"]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _fallback_sparse(text: str) -> dict[int, float]:
    sparse: dict[int, float] = {}
    tokens = _TOKEN_RE.findall((text or "").lower()) or ["empty"]
    for token in tokens:
        idx = int.from_bytes(hashlib.md5(token.encode("utf-8")).digest()[:4], "little") % (2**31 - 1)
        sparse[idx] = sparse.get(idx, 0.0) + 1.0
    # l2 normalize sparse weights lightly
    norm = math.sqrt(sum(v * v for v in sparse.values())) or 1.0
    return {k: float(v / norm) for k, v in sparse.items()}


def lexical_weights_to_milvus_sparse(lexical_weights: dict[Any, float]) -> dict[int, float]:
    """Convert BGE-M3 lexical weights to Milvus sparse dict[int, float]."""
    sparse: dict[int, float] = {}
    for key, value in (lexical_weights or {}).items():
        weight = float(value)
        if weight == 0.0:
            continue
        if isinstance(key, int):
            idx = key
        elif isinstance(key, str) and key.isdigit():
            idx = int(key)
        else:
            token = str(key)
            idx = int.from_bytes(hashlib.md5(token.encode("utf-8")).digest()[:4], "little") % (2**31 - 1)
        sparse[idx] = sparse.get(idx, 0.0) + weight
    return sparse


class BGEM3Encoder:
    """Encode texts into dense + sparse vectors with BGE-M3."""

    def __init__(self) -> None:
        settings = get_settings()
        self.model_path = settings.bge_m3_model_path
        self.model_revision = settings.bge_m3_model_revision or None
        self.cache_dir = settings.bge_model_cache_dir or None
        self.local_files_only = bool(settings.bge_local_files_only)
        self.dim = settings.rag_embedding_dim
        self.device = _resolve_device(settings.bge_m3_device)
        self.use_fp16 = bool(settings.bge_m3_use_fp16) and self.device != "cpu"
        self.batch_size = settings.bge_m3_batch_size
        self.max_length = settings.bge_m3_max_length
        self.cache_size = max(0, settings.bge_embedding_cache_size)
        self.allow_hash_fallback = bool(settings.bge_allow_hash_fallback)
        self._model = None
        self.provider = "unavailable"
        self.resolved_model_path: str | None = None
        self._load_error: str | None = None
        self._cache: OrderedDict[
            str, tuple[list[float], dict[int, float]]
        ] = OrderedDict()
        # Keep the former private name as an alias for callers created before
        # the cache API was made explicit.
        self._embedding_cache = self._cache
        self._inference_lock = RLock()
        self._load_model()

    def _cache_key(self, text: str) -> str:
        model_identity = getattr(self, "model_revision", None) or self.model_path
        payload = f"{model_identity}\x1f{self.max_length}\x1f{text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load_model(self) -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel
            from huggingface_hub import snapshot_download

            model_path = self.model_path
            if not Path(model_path).exists():
                model_path = snapshot_download(
                    repo_id=self.model_path,
                    revision=self.model_revision,
                    cache_dir=self.cache_dir,
                    local_files_only=self.local_files_only,
                )
            self.resolved_model_path = str(model_path)

            # FlagEmbedding uses `devices` (cpu/cuda). Prefer cpu on Apple for stability.
            devices = self.device
            use_fp16 = self.use_fp16
            if devices == "mps":
                devices = "cpu"
                use_fp16 = False
            self._model = BGEM3FlagModel(
                model_path,
                use_fp16=use_fp16,
                devices=devices,
            )
            self.provider = "bge-m3"
            logger.info(
                "Loaded BGE-M3 from %s (revision=%s) on %s",
                model_path,
                getattr(self, "model_revision", None) or "unresolved",
                devices,
            )
        except Exception as exc:  # noqa: BLE001
            self._model = None
            self.provider = (
                "hash-fallback" if getattr(self, "allow_hash_fallback", False) else "unavailable"
            )
            self._load_error = str(exc)
            logger.exception("Failed to load BGE-M3: %s", exc)

    def encode(self, texts: list[str]) -> tuple[list[list[float]], list[dict[int, float]], str]:
        cleaned = [t if t and str(t).strip() else " " for t in texts]
        if self._model is None:
            if not getattr(self, "allow_hash_fallback", False):
                raise RuntimeError(
                    "BGE-M3 is unavailable and BGE_ALLOW_HASH_FALLBACK is disabled: "
                    f"{self._load_error or 'unknown load error'}"
                )
            dense = [_fallback_dense(t, self.dim) for t in cleaned]
            sparse = [_fallback_sparse(t) for t in cleaned]
            return dense, sparse, "hash-fallback"

        try:
            cache: OrderedDict[str, tuple[list[float], dict[int, float]]] = getattr(
                self, "_cache", getattr(self, "_embedding_cache", OrderedDict())
            )
            # __new__-constructed test doubles and older callers may expose
            # only one of the historical private cache names.
            self._cache = cache
            self._embedding_cache = cache
            cache_keys = [self._cache_key(text) for text in cleaned]
            lock = getattr(self, "_inference_lock", RLock())

            with lock:
                missing: dict[str, str] = {}
                resolved: dict[
                    str, tuple[list[float], dict[int, float]]
                ] = {}
                for key, text in zip(cache_keys, cleaned, strict=True):
                    if key in cache:
                        # Pin every cache hit needed by this request before new
                        # vectors are inserted. A full LRU cache used to evict
                        # one of these hits mid-request, leaving the final
                        # assembly with neither a cache value nor call_result.
                        resolved[key] = cache[key]
                        cache.move_to_end(key)
                    elif key not in missing:
                        missing[key] = text

                if missing:
                    missing_keys = list(missing)
                    output = self._model.encode(
                        [missing[key] for key in missing_keys],
                        batch_size=self.batch_size,
                        max_length=self.max_length,
                        return_dense=True,
                        return_sparse=True,
                        return_colbert_vecs=False,
                    )
                    for key, vector, lexical in zip(
                        missing_keys,
                        output["dense_vecs"],
                        output["lexical_weights"],
                        strict=True,
                    ):
                        dense_value = vector.tolist() if hasattr(vector, "tolist") else list(vector)
                        value = (dense_value, lexical_weights_to_milvus_sparse(lexical))
                        resolved[key] = value
                        if self.cache_size:
                            cache[key] = value
                            cache.move_to_end(key)
                            while len(cache) > self.cache_size:
                                cache.popitem(last=False)

                dense: list[list[float]] = []
                sparse: list[dict[int, float]] = []
                for key in cache_keys:
                    dense_value, sparse_value = resolved[key]
                    if key in cache:
                        cache.move_to_end(key)
                    dense.append(list(dense_value))
                    sparse.append(dict(sparse_value))
            if dense and len(dense[0]) != self.dim:
                raise ValueError(
                    "BGE-M3 dense dimension does not match RAG_EMBEDDING_DIM: "
                    f"{len(dense[0])} != {self.dim}"
                )
            return dense, sparse, "bge-m3"
        except Exception as exc:  # noqa: BLE001
            logger.exception("BGE-M3 encode failed: %s", exc)
            if not getattr(self, "allow_hash_fallback", False):
                raise RuntimeError(f"BGE-M3 encode failed: {exc}") from exc
            dense = [_fallback_dense(t, self.dim) for t in cleaned]
            sparse = [_fallback_sparse(t) for t in cleaned]
            return dense, sparse, "hash-fallback"

    def encode_query(self, text: str) -> tuple[list[float], dict[int, float], str]:
        dense, sparse, provider = self.encode([text])
        return dense[0], sparse[0], provider


@lru_cache
def get_bge_m3_encoder() -> BGEM3Encoder:
    return BGEM3Encoder()


# Backward-compatible helpers used by older imports.
def embed_texts(texts: list[str]) -> tuple[list[list[float]], str]:
    dense, _sparse, provider = get_bge_m3_encoder().encode(texts)
    return dense, provider


def embed_query(text: str) -> tuple[list[float], str]:
    dense, _sparse, provider = get_bge_m3_encoder().encode_query(text)
    return dense, provider
