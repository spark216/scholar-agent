"""AstaBench PaperFinding data loading and metric-compatible scoring.

The implementation follows the pinned AstaBench PaperFinding formulas while
using ScholarAgent as the solver. It intentionally does not import InspectAI so
the web backend can remain lightweight.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import dotenv_values
from huggingface_hub import hf_hub_download

from backend.app.config import get_settings
from backend.app.models.paper_test import AstaOfficialMetric
from backend.app.models.schemas import PaperItem, SearchResponse
from backend.app.services.academic_http import academic_http_request
from backend.app.services.academic_search import _wait_for_semantic_scholar_slot


ASTA_DATASET_REPO = "allenai/asta-bench"
ASTA_DATASET_REVISION = "a600dc767f850385f4664772e3ba7a7f8be17d5e"
ASTA_RELEASE_SUFFIX = "2025_05"
ASTA_NORMALIZER_FILE = "tasks/paper_finder_bench/normalizer_reference.json"
S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"


@dataclass
class AstaScoredResponse:
    official: AstaOfficialMetric
    perfect_flags: list[int]
    gains: list[int]
    matched_titles: list[str]
    relevant_count: int


def _asta_root() -> Path | None:
    settings = get_settings()
    candidates: list[Path] = []
    if settings.paper_test_dataset_root:
        configured = Path(settings.paper_test_dataset_root).expanduser()
        candidates.extend([configured, configured / "asta-bench-main"])
    candidates.append(Path.home() / "Downloads" / "paper-dataset" / "asta-bench-main")
    return next((path for path in candidates if (path / "README.md").is_file()), None)


def asta_source_available() -> bool:
    """The source checkout supplies configuration; official data may download lazily."""
    return _asta_root() is not None


def _asta_credentials() -> dict[str, str]:
    values: dict[str, str] = {}
    root = _asta_root()
    if root and (root / ".env").is_file():
        values.update(
            {
                str(key): str(value)
                for key, value in dotenv_values(root / ".env").items()
                if key and value
            }
        )
    for key in ("HF_ACCESS_TOKEN", "HF_TOKEN", "ASTA_TOOL_KEY"):
        if os.getenv(key):
            values[key] = str(os.getenv(key))
    return values


def _hf_token() -> str | None:
    values = _asta_credentials()
    return values.get("HF_ACCESS_TOKEN") or values.get("HF_TOKEN")


def _download(filename: str, *, local_files_only: bool = False) -> Path:
    return Path(
        hf_hub_download(
            repo_id=ASTA_DATASET_REPO,
            filename=filename,
            repo_type="dataset",
            revision=ASTA_DATASET_REVISION,
            token=_hf_token(),
            local_files_only=local_files_only,
        )
    )


def asta_cached_sample_count(split: str) -> int:
    try:
        path = _download(
            f"tasks/paper_finder_bench/{split}_{ASTA_RELEASE_SUFFIX}.json",
            local_files_only=True,
        )
    except Exception:  # noqa: BLE001 - a cache miss is an expected state
        return 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    return len(payload) if isinstance(payload, list) else 0


def asta_dataset_sha256(split: str) -> str:
    """Hash the exact pinned AstaBench split used by a completed run."""
    path = _download(
        f"tasks/paper_finder_bench/{split}_{ASTA_RELEASE_SUFFIX}.json",
        local_files_only=True,
    )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_asta_samples(split: str, *, limit: int, seed: int) -> list[dict[str, Any]]:
    import random

    data_path = _download(f"tasks/paper_finder_bench/{split}_{ASTA_RELEASE_SUFFIX}.json")
    normalizer_path = _download(ASTA_NORMALIZER_FILE)
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    normalizers = json.loads(normalizer_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("AstaBench PaperFinding split must be a JSON list")
    samples: list[dict[str, Any]] = []
    for item in payload:
        query = item.get("input") or {}
        qid = str(query.get("query_id") or "")
        if not qid or not query.get("query"):
            continue
        criteria = item.get("scorer_criteria") or {}
        perfect_count = int(normalizers.get(qid) or 0)
        if qid.startswith("semantic") and perfect_count:
            factor = max(2.0, 2 / math.log(perfect_count) if perfect_count > 1 else 10.0)
            total_relevant = math.ceil(perfect_count * factor)
        else:
            total_relevant = perfect_count or len(criteria.get("corpus_ids") or [])
        samples.append(
            {
                "qid": qid,
                "question": str(query["query"]),
                "answer": [],
                "answer_arxiv_id": [],
                "_asta_criteria": criteria,
                "_asta_total_relevant": total_relevant,
                "_asta_split": split,
            }
        )
    if limit >= len(samples):
        return samples
    indices = sorted(random.Random(seed).sample(range(len(samples)), limit))
    return [samples[index] for index in indices]


def _s2_identifier(paper: PaperItem) -> str | None:
    if getattr(paper, "semantic_scholar_id", None):
        return f"CorpusId:{paper.semantic_scholar_id}"
    if paper.doi:
        return f"DOI:{paper.doi}"
    if paper.arxiv_id:
        return f"ARXIV:{paper.arxiv_id}"
    if paper.pmid:
        return f"PMID:{paper.pmid}"
    return None


def resolve_corpus_ids(papers: list[PaperItem]) -> list[str | None]:
    """Resolve existing scholarly identifiers in one Semantic Scholar request."""
    identifiers: list[str] = []
    positions: list[int] = []
    resolved: list[str | None] = [None] * len(papers)
    for position, paper in enumerate(papers):
        identifier = _s2_identifier(paper)
        if identifier:
            positions.append(position)
            identifiers.append(identifier)
    if not identifiers:
        return resolved
    credentials = _asta_credentials()
    headers = {"Content-Type": "application/json"}
    api_key = get_settings().semantic_scholar_api_key or credentials.get("ASTA_TOOL_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    # Share the same cadence and retry/circuit policy as the first-round
    # Semantic Scholar search. Without this, Asta scoring can immediately make
    # a batch request after search and exceed the common 1 request/second key.
    _wait_for_semantic_scholar_slot()
    response = academic_http_request(
        "semantic-scholar",
        "POST",
        S2_BATCH_URL,
        params={"fields": "corpusId,title,year,publicationDate"},
        json={"ids": identifiers},
        headers=headers,
    )
    try:
        payload = response.json()
    finally:
        response.close()
    for position, item in zip(positions, payload, strict=False):
        if item and item.get("corpusId") is not None:
            resolved[position] = str(item["corpusId"])
    return resolved


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("judge output must be an object")
    return value


def _judge_semantic_relevance(
    papers: list[PaperItem],
    corpus_ids: list[str | None],
    criteria: list[dict[str, Any]],
    known_good: set[str],
) -> tuple[dict[str, int], str]:
    scores = {corpus_id: 3 for corpus_id in corpus_ids if corpus_id in known_good}
    pending = [
        (paper, corpus_id)
        for paper, corpus_id in zip(papers, corpus_ids, strict=False)
        if corpus_id and corpus_id not in scores
    ]
    if not pending:
        return scores, "not_required"
    settings = get_settings()
    if not settings.zhipuai_api_key:
        return scores, "unavailable:no_zhipu_key"
    try:
        from zhipuai import ZhipuAI

        try:
            client = ZhipuAI(
                api_key=settings.zhipuai_api_key,
                timeout=settings.evidrank_evidence_timeout_seconds,
                max_retries=0,
            )
        except TypeError:
            client = ZhipuAI(api_key=settings.zhipuai_api_key)
        batch_size = max(1, min(settings.evidrank_evidence_batch_size, 8))
        completed = 0
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset : offset + batch_size]
            prompt = {
                "instruction": (
                    "Judge every paper against every criterion. Use only title and abstract. "
                    "Return JSON with results; each result has corpus_id and criteria mapping. "
                    "Each criterion label must be Perfectly Relevant, Somewhat Relevant, or Not Relevant."
                ),
                "criteria": [
                    {"name": item.get("name"), "description": item.get("description")}
                    for item in criteria
                ],
                "papers": [
                    {
                        "corpus_id": corpus_id,
                        "title": paper.title,
                        "abstract": paper.abstract,
                    }
                    for paper, corpus_id in batch
                ],
            }
            kwargs = {
                "model": settings.evidrank_evidence_llm_model,
                "messages": [
                    {"role": "system", "content": "You are the AstaBench PaperFinding relevance judge."},
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
                "temperature": 0,
                "max_tokens": settings.evidrank_evidence_max_tokens,
                "response_format": {"type": "json_object"},
            }
            try:
                response = client.chat.completions.create(**kwargs)
            except TypeError:
                kwargs.pop("response_format", None)
                response = client.chat.completions.create(**kwargs)
            payload = _extract_json(response.choices[0].message.content or "")
            for result in payload.get("results") or []:
                corpus_id = str(result.get("corpus_id") or "")
                labels = result.get("criteria") or {}
                weighted = 0.0
                names_seen: set[str] = set()
                for criterion in criteria:
                    name = str(criterion.get("name") or "")
                    raw = labels.get(name)
                    label = raw.get("relevance") if isinstance(raw, dict) else raw
                    points = {
                        "perfectly relevant": 3,
                        "somewhat relevant": 1,
                        "not relevant": 0,
                    }.get(str(label or "").strip().casefold())
                    if points is None:
                        continue
                    names_seen.add(name)
                    weighted += float(criterion.get("weight") or 0.0) * points / 3
                if names_seen == {str(item.get("name") or "") for item in criteria}:
                    scores[corpus_id] = 0 if weighted <= 0.25 else 1 if weighted <= 0.67 else 2 if weighted <= 0.99 else 3
                    completed += 1
        status = "ok" if completed == len(pending) else f"partial:{completed}/{len(pending)}"
        return scores, status
    except Exception as exc:  # noqa: BLE001 - fail closed and surface degradation
        return scores, f"failed:{type(exc).__name__}"


def _dcg(values: list[int]) -> float:
    return sum(float(value) / math.log(index + 1) for index, value in enumerate(values, start=1))


def _lower_bound_ndcg(values: list[int]) -> float:
    if not values:
        return 0.0
    maximum = _dcg(sorted(values, reverse=True))
    minimum = _dcg(sorted(values))
    return (_dcg(values) - minimum) / (maximum - minimum) if maximum != minimum else 0.0


def _harmonic(left: float, right: float) -> float:
    return 2 * left * right / (left + right) if left and right else 0.0


def score_asta_response(sample: dict[str, Any], response: SearchResponse) -> AstaScoredResponse:
    papers = response.papers
    corpus_ids = resolve_corpus_ids(papers)
    criteria = sample.get("_asta_criteria") or {}
    qid = str(sample["qid"])
    query_type = qid.split("_", 1)[0]
    valid_count = sum(value is not None for value in corpus_ids)
    validity = valid_count / len(papers) if papers else 0.0
    matched_titles: list[str] = []

    if query_type == "semantic":
        known_good = {str(value) for value in criteria.get("known_to_be_good") or []}
        judgements, judge_status = _judge_semantic_relevance(
            papers,
            corpus_ids,
            list(criteria.get("relevance_criteria") or []),
            known_good,
        )
        gains = [judgements.get(value, 0) if value else 0 for value in corpus_ids]
        perfect_flags = [int(value == 3) for value in gains]
        perfect = sum(perfect_flags)
        total_relevant = max(1, int(sample.get("_asta_total_relevant") or 0))
        recall = min(1.0, perfect / total_relevant)
        rank = max(0.0, min(1.0, _lower_bound_ndcg(gains)))
        official = AstaOfficialMetric(
            score_type="semantic_f1",
            adjusted_f1=_harmonic(recall, rank),
            estimated_recall=recall,
            ndcg=rank,
            corpus_id_validity=validity,
            valid_corpus_ids=valid_count,
            submitted_papers=len(papers),
            perfectly_relevant=perfect,
            judge_status=judge_status,
        )
    else:
        gold = {str(value) for value in criteria.get("corpus_ids") or []}
        perfect_flags = [int(value in gold) if value else 0 for value in corpus_ids]
        gains = [3 * value for value in perfect_flags]
        perfect = sum(perfect_flags)
        total_relevant = max(1, len(gold))
        recall = perfect / total_relevant
        precision = perfect / len(papers) if papers else 0.0
        ideal = [1] * min(len(gold), len(papers))
        binary_dcg = sum(value / math.log2(index + 1) for index, value in enumerate(perfect_flags, start=1))
        ideal_dcg = sum(value / math.log2(index + 1) for index, value in enumerate(ideal, start=1))
        rank = binary_dcg / ideal_dcg if ideal_dcg else 0.0
        official = AstaOfficialMetric(
            score_type=f"{query_type}_f1",
            standard_f1=_harmonic(recall, precision),
            estimated_recall=recall,
            ndcg=rank,
            corpus_id_validity=validity,
            valid_corpus_ids=valid_count,
            submitted_papers=len(papers),
            perfectly_relevant=perfect,
        )
    for paper, flag in zip(papers, perfect_flags, strict=False):
        if flag:
            matched_titles.append(paper.title)
    return AstaScoredResponse(
        official=official,
        perfect_flags=perfect_flags,
        gains=gains,
        matched_titles=matched_titles,
        relevant_count=total_relevant,
    )
