"""Run ScholarAgent against local PaSa test sets and calculate IR metrics."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import tempfile
import time
import unicodedata
import subprocess
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.app.config import ROOT_DIR, get_settings
from backend.app.models.paper_test import (
    AstaOfficialMetric,
    PaperTestDatasetInfo,
    PaperTestHistoryResponse,
    PaperTestMetric,
    PaperTestQueryResult,
    PaperTestProviderDiagnostic,
    PaperTestReport,
    PaperTestRequest,
    PaperTestRunMetadata,
    PaperTestRunSummary,
    PaperTestStageMetric,
)
from backend.app.models.schemas import SearchResponse
from backend.app.services.search_service import run_paper_search
from backend.app.services.paper_test_asta import (
    ASTA_DATASET_REVISION,
    asta_dataset_sha256,
    asta_cached_sample_count,
    asta_source_available,
    load_asta_samples,
    score_asta_response,
)
from backend.app.services.local_pasa_corpus import PASA_INDEX_VERSION
from backend.app.services.persistent_cache import FrozenProviderCacheMiss


DATASET_SPECS = {
    "pasa-real": {
        "name": "PaSa RealScholarQuery",
        "relative_path": Path("RealScholarQuery/test.jsonl"),
        "description": "50 条真实学术检索问题，适合小规模端到端效果验证。",
    },
    "pasa-auto": {
        "name": "PaSa AutoScholarQuery",
        "relative_path": Path("AutoScholarQuery/test.jsonl"),
        "description": "1,000 条自动构造学术检索问题，适合扩大测试覆盖率。",
    },
    "pasa-auto-dev": {
        "name": "PaSa AutoScholarQuery Dev",
        "relative_path": Path("AutoScholarQuery/dev.jsonl"),
        "split": "dev",
        "description": "1,000 条开发集查询，适合阈值选择和功能迭代，不作为最终盲测成绩。",
    },
    "spar": {
        "name": "SPARBench",
        "relative_path": Path("SPARBench/test.jsonl"),
        "description": "50 条 SPARBench 论文检索问题；标题 + arXiv-ID 匹配（派生自 dataset/SPAR_Benchmark.jsonl）。",
    },
    "asta-paper-finder-validation": {
        "name": "AstaBench PaperFinding Validation",
        "split": "validation",
        "evaluation_family": "astabench",
        "description": "AstaBench 固定版本验证集；语义查询使用相关性 Judge 与官方 normalizer。",
    },
    "asta-paper-finder-test": {
        "name": "AstaBench PaperFinding Test",
        "split": "test",
        "evaluation_family": "astabench",
        "description": "AstaBench 固定版本测试集；需要 Hugging Face 访问权限和网络。",
    },
    "trec-pm-2017": {
        "name": "TREC Precision Medicine 2017",
        "relative_path": Path("eval/pm2017.jsonl"),
        "evaluation_family": "trec-pm",
        "description": "30 条临床病例检索问题；使用 PMID 精确匹配，测试 PubMed 医学召回。",
    },
    "trec-pm-2018": {
        "name": "TREC Precision Medicine 2018",
        "relative_path": Path("eval/pm2018.jsonl"),
        "evaluation_family": "trec-pm",
        "description": "50 条临床病例检索问题；使用 PMID 精确匹配，测试 PubMed 医学召回。",
    },
    "trec-pm-2020": {
        "name": "TREC Precision Medicine 2020",
        "relative_path": Path("eval/pm2020.jsonl"),
        "evaluation_family": "trec-pm",
        "description": "31 条临床病例检索问题；使用 NIST phase-1 PMID 金标准。",
    },
    "beir-scifact": {
        "name": "BEIR SciFact Test",
        "relative_path": Path("eval/beir_scifact.jsonl"),
        "evaluation_family": "beir-open",
        "description": "300 条科学事实检索问题；开放 Provider 模式按规范化标题匹配。",
    },
    "beir-nfcorpus": {
        "name": "BEIR NFCorpus Test",
        "relative_path": Path("eval/beir_nfcorpus.jsonl"),
        "evaluation_family": "beir-open",
        "description": "323 条医学全文检索问题；开放 Provider 模式按规范化标题匹配。",
    },
    "beir-arguana": {
        "name": "BEIR ArguAna Test（辅助）",
        "relative_path": Path("eval/beir_arguana.jsonl"),
        "evaluation_family": "beir-open",
        "description": "1,401 条论证反驳检索问题；用于泛化压力测试，不作为论文搜索主成绩。",
    },
    "beir-trec-covid": {
        "name": "BEIR TREC-COVID Test",
        "relative_path": Path("eval/beir_trec-covid.jsonl"),
        "evaluation_family": "beir-open",
        "description": "50 条 COVID-19 文献检索问题；相关文献很多，建议重点观察 nDCG。",
    },
    "beir-scidocs": {
        "name": "BEIR SciDocs Test",
        "relative_path": Path("eval/beir_scidocs.jsonl"),
        "evaluation_family": "beir-open",
        "description": "1,000 条科学文献检索问题；适合测试跨领域论文召回。",
    },
}

PAPER_TEST_DIR = ROOT_DIR / "backend" / "paper-test"
RESULTS_DIR = PAPER_TEST_DIR / "results"
RUNS_DIR = RESULTS_DIR / "runs"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_sha256(dataset: PaperTestDatasetInfo) -> str:
    if dataset.evaluation_family == "astabench":
        return asta_dataset_sha256(dataset.split)
    if not dataset.source_path:
        raise FileNotFoundError(f"dataset source is unavailable: {dataset.id}")
    return _sha256_file(Path(dataset.source_path))


def _git_state() -> tuple[str | None, bool | None]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout
        return revision or None, bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        return None, None


def _run_configuration(request: PaperTestRequest) -> dict[str, Any]:
    settings = get_settings()
    return {
        "dataset": request.dataset,
        "sample_limit": request.sample_limit,
        "ks": request.ks,
        "seed": request.seed,
        "use_mock": request.use_mock,
        "models": {
            "bge_m3": {
                "path": settings.bge_m3_model_path,
                "revision": settings.bge_m3_model_revision,
                "candidate_limit": settings.paper_test_bge_m3_candidate_limit,
                "max_candidate_limit": settings.paper_test_bge_m3_max_candidate_limit,
                "allow_hash_fallback": settings.bge_allow_hash_fallback,
            },
            "bge_reranker": {
                "enabled": settings.bge_reranker_enabled,
                "path": settings.bge_reranker_model_path,
                "revision": settings.bge_reranker_model_revision,
                "top_n": settings.paper_test_bge_reranker_top_n,
                "required": settings.bge_reranker_required,
            },
            "pasa_selector": {
                "enabled": settings.pasa_selector_enabled,
                "path": settings.pasa_selector_model_path,
                "top_n": settings.paper_test_pasa_selector_top_n,
                "rank_weight": settings.pasa_selector_rank_weight,
                "protect_head": settings.pasa_selector_protect_head,
                "use_4bit": settings.pasa_selector_use_4bit,
                "required": settings.pasa_selector_required,
            },
        },
        "retrieval": {
            "max_iterations": settings.agent_search_max_iterations,
            "minimum_evaluation_variants": settings.paper_test_min_query_variants,
            "max_api_calls": settings.agent_search_max_api_calls,
            "provider_limit": settings.academic_provider_limit,
            "local_provider_limit": settings.academic_local_provider_limit,
            "local_index_version": PASA_INDEX_VERSION,
            "local_vector_enabled": settings.academic_local_vector_enabled,
            "local_vector_collection": settings.academic_local_vector_collection,
            "local_vector_provider_limit": settings.academic_local_vector_provider_limit,
            "llm_evolution_enabled": settings.agent_search_llm_evolution_enabled,
            "llm_evolution_max_calls": settings.agent_search_llm_evolution_max_calls,
            "llm_evolution_prompt_version": settings.agent_search_llm_evolution_prompt_version,
            "semantic_scholar_enabled": settings.semantic_scholar_enabled,
            "provider_cache_enabled": settings.academic_provider_cache_enabled,
            "provider_cache_ttl_seconds": settings.academic_provider_cache_ttl_seconds,
        },
        "ranking": {
            "rrf_k": settings.rrf_k,
            "cross_rank_weight": settings.bge_reranker_cross_rank_weight,
            "facet_rank_weight": settings.bge_reranker_facet_rank_weight,
            "head_protect": settings.bge_reranker_head_protect,
            "candidate_provider_head_fraction": settings.bge_candidate_provider_head_fraction,
            "candidate_provider_head_per_source": settings.bge_candidate_provider_head_per_source,
            "candidate_anchor_fraction": settings.bge_candidate_anchor_fraction,
            "candidate_anchor_min_chars": settings.bge_candidate_anchor_min_chars,
            "evidrank_enabled": settings.evidrank_enabled,
            "evidrank_decision_policy": settings.evidrank_decision_policy,
            "evidrank_threshold_version": settings.evidrank_threshold_version,
        },
        "evaluation": {
            "fast_mode": settings.paper_test_fast_mode,
            "asta_dataset_revision": ASTA_DATASET_REVISION,
        },
    }


def _run_metadata(
    request: PaperTestRequest,
    dataset: PaperTestDatasetInfo,
    *,
    provider_snapshot_sha256: str | None = None,
    sample_qids: list[str] | None = None,
    provider_snapshot_query_count: int = 0,
) -> PaperTestRunMetadata:
    configuration = _run_configuration(request)
    serialized = json.dumps(
        configuration,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    code_commit, code_dirty = _git_state()
    dataset_hash = _dataset_sha256(dataset)
    normalized_qids = sorted(str(value) for value in (sample_qids or []))
    sample_qids_sha256 = (
        hashlib.sha256(
            json.dumps(normalized_qids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if normalized_qids
        else None
    )
    return PaperTestRunMetadata(
        dataset_sha256=dataset_hash,
        config_fingerprint=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        code_commit=code_commit,
        code_dirty=code_dirty,
        fast_mode=get_settings().paper_test_fast_mode,
        configuration=configuration,
        provider_snapshot_sha256=provider_snapshot_sha256,
        sample_qids_sha256=sample_qids_sha256,
        provider_snapshot_query_count=provider_snapshot_query_count,
        provider_snapshot_complete=(
            bool(normalized_qids)
            and provider_snapshot_query_count == len(normalized_qids)
        ),
        provider_cache_mode=request.provider_cache_mode,
    )


def _dataset_roots() -> list[Path]:
    settings = get_settings()
    roots: list[Path] = []
    if settings.paper_test_dataset_root:
        roots.append(Path(settings.paper_test_dataset_root).expanduser())
    roots.extend(
        [
            PAPER_TEST_DIR / "datasets",
            Path.home() / "Downloads" / "paper-dataset" / "pasa-dataset",
            Path.home() / "Downloads" / "dataset",
        ]
    )
    expanded: list[Path] = []
    for root in roots:
        expanded.append(root)
        expanded.append(root / "pasa-dataset")
    return list(dict.fromkeys(expanded))


def _resolve_dataset(dataset_id: str) -> Path | None:
    spec = DATASET_SPECS[dataset_id]
    if spec.get("evaluation_family") == "astabench":
        return None
    for root in _dataset_roots():
        relative_path = Path(spec["relative_path"])
        candidates = [root / relative_path]
        # PAPER_TEST_DATASET_ROOT may point either at the dataset package or
        # directly at its eval directory.
        if relative_path.parts and relative_path.parts[0] == "eval":
            candidates.append(root / relative_path.name)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    return None


@lru_cache(maxsize=8)
def _line_count(path: str, modified_ns: int) -> int:
    del modified_ns
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def list_paper_test_datasets() -> list[PaperTestDatasetInfo]:
    datasets: list[PaperTestDatasetInfo] = []
    for dataset_id, spec in DATASET_SPECS.items():
        if spec.get("evaluation_family") == "astabench":
            split = str(spec["split"])
            available = asta_source_available()
            datasets.append(
                PaperTestDatasetInfo(
                    id=dataset_id,
                    name=str(spec["name"]),
                    split=split,
                    available=available,
                    sample_count=asta_cached_sample_count(split) if available else 0,
                    source_path="allenai/asta-bench@a600dc7" if available else None,
                    description=str(spec["description"]),
                    evaluation_family="astabench",
                    requires_network=True,
                    evaluation_protocol="official-judge",
                    leaderboard_comparable=False,
                )
            )
            continue
        path = _resolve_dataset(dataset_id)
        count = _line_count(str(path), path.stat().st_mtime_ns) if path else 0
        datasets.append(
            PaperTestDatasetInfo(
                id=dataset_id,
                name=str(spec["name"]),
                split=str(spec.get("split") or "test"),
                available=path is not None,
                sample_count=count,
                source_path=str(path) if path else None,
                description=str(spec["description"]),
                evaluation_family=str(spec.get("evaluation_family") or "pasa"),
                requires_network=True,
                evaluation_protocol=(
                    "open-provider-proxy"
                    if spec.get("evaluation_family") == "beir-open"
                    else "reference-qrels"
                ),
                leaderboard_comparable=False,
            )
        )
    return datasets


def _dataset_info(dataset_id: str) -> PaperTestDatasetInfo:
    return next(item for item in list_paper_test_datasets() if item.id == dataset_id)


def validate_paper_test_sample_limit(
    request: PaperTestRequest,
) -> PaperTestDatasetInfo:
    """Validate sampling against the selected dataset's real row count."""
    dataset = _dataset_info(request.dataset)
    if dataset.sample_count and request.sample_limit > dataset.sample_count:
        raise ValueError(
            f"抽样数量不能超过测试集 {dataset.name} 的总量 {dataset.sample_count}"
        )
    return dataset


def _load_samples(path: Path, *, limit: int, seed: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not value.get("question") or not value.get("qid"):
                raise ValueError(f"invalid PaSa sample at line {line_number}")
            samples.append(value)
    if limit >= len(samples):
        return samples
    indices = sorted(random.Random(seed).sample(range(len(samples)), limit))
    return [samples[index] for index in indices]


def normalize_arxiv_id(value: str | None) -> str:
    normalized = str(value or "").strip().casefold()
    normalized = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", normalized)
    normalized = re.sub(r"^arxiv:\s*", "", normalized)
    normalized = normalized.removesuffix(".pdf")
    return re.sub(r"v\d+$", "", normalized)


def normalize_pmid(value: str | None) -> str:
    normalized = str(value or "").strip().casefold()
    normalized = re.sub(r"^https?://pubmed\.ncbi\.nlm\.nih\.gov/", "", normalized)
    normalized = re.sub(r"^(?:pmid|pubmed):\s*", "", normalized)
    return normalized.strip().strip("/")


def normalize_doi(value: str | None) -> str:
    """Normalize DOI identifiers for evaluator identity matching.

    DOI matching is deliberately conservative: URL/prefix decoration and
    surrounding whitespace are removed, while the DOI body is retained.
    Titles remain only the final fallback when no stable identifier exists.
    """
    normalized = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    normalized = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", normalized)
    normalized = re.sub(r"^doi:\s*", "", normalized)
    return normalized.strip().strip("/")


def normalize_corpus_id(value: str | None) -> str:
    """Normalize Semantic Scholar/Asta-style corpus identifiers."""
    normalized = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    normalized = re.sub(r"^(?:corpusid|corpus_id|s2)\s*[:#]\s*", "", normalized)
    return normalized.strip()


def normalize_title(value: str | None) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def _mapping_identifier(values: dict[str, Any], *names: str) -> Any:
    wanted = {name.casefold() for name in names}
    return next(
        (value for key, value in values.items() if str(key).casefold() in wanted),
        None,
    )


def _paper_keys(paper: Any) -> tuple[str, str, str, str, str]:
    if isinstance(paper, dict):
        external_ids = paper.get("external_ids") or paper.get("identifiers") or {}
        arxiv_id = normalize_arxiv_id(
            paper.get("arxiv_id") or _mapping_identifier(external_ids, "arxiv")
        )
        pmid = normalize_pmid(
            paper.get("pmid") or _mapping_identifier(external_ids, "pmid", "pubmed")
        )
        doi = normalize_doi(
            paper.get("doi") or _mapping_identifier(external_ids, "doi")
        )
        corpus_id = normalize_corpus_id(
            paper.get("corpus_id")
            or _mapping_identifier(external_ids, "corpus_id", "corpusid", "semantic_scholar_corpus_id")
        )
        title = normalize_title(paper.get("title"))
        return arxiv_id, pmid, doi, corpus_id, title
    arxiv_id = normalize_arxiv_id(getattr(paper, "arxiv_id", None))
    pmid = normalize_pmid(getattr(paper, "pmid", None))
    doi = normalize_doi(getattr(paper, "doi", None))
    corpus_id = normalize_corpus_id(
        getattr(paper, "corpus_id", None)
        or getattr(paper, "semantic_scholar_corpus_id", None)
    )
    title = normalize_title(getattr(paper, "title", None))
    return arxiv_id, pmid, doi, corpus_id, title


def _gold_records(
    gold_arxiv_ids: list[str],
    gold_titles: list[str],
    gold_pmids: list[str] | None = None,
    gold_dois: list[str] | None = None,
    gold_corpus_ids: list[str] | None = None,
) -> list[tuple[str, str, str, str, str, str]]:
    """Pair each qrel's strong ID and title as alternative identity keys."""
    pmids = gold_pmids or []
    dois = gold_dois or []
    corpus_ids = gold_corpus_ids or []
    records: list[tuple[str, str, str, str, str, str]] = []
    for index in range(max(len(gold_arxiv_ids), len(gold_titles), len(pmids), len(dois), len(corpus_ids))):
        raw_id = gold_arxiv_ids[index] if index < len(gold_arxiv_ids) else ""
        raw_pmid = pmids[index] if index < len(pmids) else ""
        raw_doi = dois[index] if index < len(dois) else ""
        raw_corpus_id = corpus_ids[index] if index < len(corpus_ids) else ""
        raw_title = gold_titles[index] if index < len(gold_titles) else ""
        arxiv_id = normalize_arxiv_id(raw_id)
        pmid = normalize_pmid(raw_pmid)
        doi = normalize_doi(raw_doi)
        corpus_id = normalize_corpus_id(raw_corpus_id)
        title = normalize_title(raw_title)
        if arxiv_id or pmid or doi or corpus_id or title:
            records.append(
                (
                    arxiv_id,
                    pmid,
                    doi,
                    corpus_id,
                    title,
                    str(raw_title or raw_doi or raw_corpus_id or raw_pmid or raw_id),
                )
            )
    return records


def _relevance_flags(
    papers: list[Any],
    gold_arxiv_ids: list[str],
    gold_titles: list[str],
    gold_pmids: list[str] | None = None,
    gold_dois: list[str] | None = None,
    gold_corpus_ids: list[str] | None = None,
) -> tuple[list[int], list[str]]:
    gold = _gold_records(
        gold_arxiv_ids,
        gold_titles,
        gold_pmids,
        gold_dois,
        gold_corpus_ids,
    )
    unmatched = set(range(len(gold)))
    flags: list[int] = []
    matches: list[str] = []
    for paper in papers:
        arxiv_id, pmid, doi, corpus_id, title = _paper_keys(paper)
        matched_index = next(
            (
                index
                for index in unmatched
                if arxiv_id and gold[index][0] and arxiv_id == gold[index][0]
            ),
            None,
        )
        if matched_index is None:
            matched_index = next(
                (
                    index
                    for index in unmatched
                if pmid and gold[index][1] and pmid == gold[index][1]
            ),
            None,
        )
        if matched_index is None:
            matched_index = next(
                (
                    index
                    for index in unmatched
                    if doi and gold[index][2] and doi == gold[index][2]
                ),
                None,
            )
        if matched_index is None:
            matched_index = next(
                (
                    index
                    for index in unmatched
                    if corpus_id and gold[index][3] and corpus_id == gold[index][3]
                ),
                None,
            )
        if matched_index is None:
            matched_index = next(
                (
                    index
                    for index in unmatched
                    if title
                    and gold[index][4]
                    and title == gold[index][4]
                    # A conflicting identifier in the same namespace is
                    # evidence of a different paper, not a title fallback.
                    and not (arxiv_id and gold[index][0] and arxiv_id != gold[index][0])
                    and not (pmid and gold[index][1] and pmid != gold[index][1])
                    and not (doi and gold[index][2] and doi != gold[index][2])
                    and not (
                        corpus_id
                        and gold[index][3]
                        and corpus_id != gold[index][3]
                    )
                ),
                None,
            )
        flags.append(1 if matched_index is not None else 0)
        if matched_index is not None:
            unmatched.remove(matched_index)
            matches.append(gold[matched_index][5])
    return flags, matches


def _metric_at_k(flags: list[int], *, relevant_count: int, k: int) -> PaperTestMetric:
    selected = flags[:k]
    hits = sum(selected)
    precision = hits / k
    recall = hits / relevant_count if relevant_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    dcg = sum(flag / math.log2(rank + 1) for rank, flag in enumerate(selected, start=1))
    ideal_hits = min(relevant_count, k)
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
    first_hit = next((rank for rank, flag in enumerate(selected, start=1) if flag), None)
    return PaperTestMetric(
        k=k,
        precision=precision,
        recall=recall,
        f1=f1,
        ndcg=ndcg,
        mrr=(1 / first_hit) if first_hit else 0.0,
        hits=hits,
    )


def _stage_diagnostics(
    response: SearchResponse,
    gold_arxiv_ids: list[str],
    gold_titles: list[str],
    gold_pmids: list[str] | None = None,
    gold_dois: list[str] | None = None,
    gold_corpus_ids: list[str] | None = None,
) -> tuple[dict[str, PaperTestStageMetric], str | None, str | None]:
    relevant_count = len(
        _gold_records(
            gold_arxiv_ids,
            gold_titles,
            gold_pmids,
            gold_dois,
            gold_corpus_ids,
        )
    )
    trace = dict(response.meta.get("candidate_trace") or {})
    trace["final"] = trace.get("final") or response.papers
    stages: dict[str, PaperTestStageMetric] = {}
    first_seen: str | None = None
    for stage in ("provider", "merged", "budget", "ranking", "final"):
        if stage not in trace:
            continue
        candidates = list(trace.get(stage) or [])
        flags, matches = _relevance_flags(
            candidates,
            gold_arxiv_ids,
            gold_titles,
            gold_pmids,
            gold_dois,
            gold_corpus_ids,
        )
        hits = sum(flags)
        stages[stage] = PaperTestStageMetric(
            candidate_count=len(candidates),
            gold_hits=hits,
            candidate_recall=(hits / relevant_count) if relevant_count else 0.0,
            matched_titles=matches,
            matched_ranks=[rank for rank, flag in enumerate(flags, start=1) if flag],
        )
        if first_seen is None and hits:
            first_seen = stage
    loss_stage: str | None = None
    previous_hits = 0
    for stage in ("provider", "merged", "budget", "ranking", "final"):
        current = stages.get(stage)
        if current is None:
            continue
        if previous_hits and current.gold_hits < previous_hits:
            loss_stage = stage
            break
        previous_hits = max(previous_hits, current.gold_hits)
    return stages, first_seen, loss_stage


def _provider_diagnostics(response: SearchResponse) -> list[PaperTestProviderDiagnostic]:
    diagnostics: list[PaperTestProviderDiagnostic] = []
    rounds = list((response.meta.get("search_agent") or {}).get("round_history") or [])
    for round_info in rounds:
        iteration = max(1, int(round_info.get("iteration") or 1))
        round_query = str(round_info.get("query") or "") or None
        academic = dict(round_info.get("academic") or {})
        for provider, provider_meta_raw in (academic.get("providers") or {}).items():
            provider_meta = dict(provider_meta_raw or {})
            attempts = list(provider_meta.get("attempts") or [])
            if not attempts:
                attempts = [provider_meta]
            for attempt in attempts:
                raw_queries = attempt.get("queries") or []
                diagnostics.append(
                    PaperTestProviderDiagnostic(
                        iteration=iteration,
                        provider=str(provider),
                        query=str(attempt.get("query") or round_query or "") or None,
                        queries=[str(value) for value in raw_queries if value],
                        offset=max(0, int(attempt.get("offset") or 0)),
                        raw_count=max(0, int(attempt.get("raw_count") or 0)),
                        returned_count=max(0, int(attempt.get("returned_count") or 0)),
                        api_calls=max(0, int(attempt.get("api_calls") or 0)),
                        latency_ms=max(0.0, float(attempt.get("latency_ms") or 0.0)),
                        error=(
                            str(attempt.get("error") or provider_meta.get("error"))
                            if attempt.get("error") or provider_meta.get("error")
                            else None
                        ),
                        circuit_open=bool(
                            attempt.get("circuit_open")
                            or provider_meta.get("circuit_open")
                        ),
                        cache_hit=bool(
                            attempt.get("cache_hit")
                            or provider_meta.get("cache_hit")
                        ),
                        skipped=str(attempt.get("skipped") or provider_meta.get("skipped") or "") or None,
                    )
                )
        tavily = dict(round_info.get("tavily") or {})
        round_errors = dict(round_info.get("errors") or {})
        diagnostics.append(
            PaperTestProviderDiagnostic(
                iteration=iteration,
                provider="tavily",
                query=round_query,
                raw_count=max(0, int(tavily.get("tavily_raw_count") or tavily.get("raw_count") or 0)),
                returned_count=max(0, int(tavily.get("returned_count") or 0)),
                api_calls=max(0, int(tavily.get("api_calls") or 0)),
                latency_ms=max(0.0, float(tavily.get("latency_ms") or 0.0)),
                error=str(round_errors["tavily"]) if round_errors.get("tavily") else None,
                cache_hit=bool(tavily.get("cache_hit")),
                skipped=str(tavily.get("skipped") or "") or None,
            )
        )
    return diagnostics


def _sample_values(sample: dict[str, Any], *keys: str) -> list[str]:
    """Read identifier aliases without treating a scalar string as characters."""
    for key in keys:
        value = sample.get(key)
        if value is None or value == "":
            continue
        # Preserve empty placeholders in aligned identifier arrays: index i
        # across title/arXiv/PMID/DOI/CorpusID describes the same qrel.
        values = value if isinstance(value, list) else [value]
        return [str(item or "") for item in values]
    return []


def _classify_query_error(error: str | None) -> str | None:
    """Map unstable exception text to stable, reportable failure classes."""
    if not error:
        return None
    value = str(error).casefold()
    if "frozen" in value and "cache" in value:
        return "frozen_snapshot_miss"
    if "intent" in value or "论文检索" in value or "academic" in value:
        return "intent_or_query_validation"
    if any(token in value for token in ("timeout", "timed out", "pooltimeout")):
        return "provider_timeout"
    if any(token in value for token in ("429", "ratelimit", "rate limit", "circuitopen")):
        return "provider_rate_limit_or_circuit"
    if any(token in value for token in ("connection", "connecterror", "network")):
        return "provider_connection"
    if "json" in value or "decode" in value:
        return "invalid_provider_or_model_payload"
    return "other"


def score_search_response(
    sample: dict[str, Any],
    response: SearchResponse,
    ks: list[int],
    *,
    latency_ms: float,
) -> PaperTestQueryResult:
    gold_titles = [str(value) for value in sample.get("answer") or []]
    gold_ids = [str(value) for value in sample.get("answer_arxiv_id") or []]
    gold_pmids = [str(value) for value in sample.get("answer_pmid") or []]
    gold_dois = _sample_values(sample, "answer_doi", "answer_dois", "gold_doi")
    gold_corpus_ids = _sample_values(
        sample,
        "answer_corpus_id",
        "answer_corpus_ids",
        "answer_semantic_scholar_corpus_id",
    )
    # Alternative identifiers at the same position describe the same qrel.
    relevant_count = len(
        _gold_records(
            gold_ids,
            gold_titles,
            gold_pmids,
            gold_dois,
            gold_corpus_ids,
        )
    )
    flags, matches = _relevance_flags(
        response.papers,
        gold_ids,
        gold_titles,
        gold_pmids,
        gold_dois,
        gold_corpus_ids,
    )
    stage_metrics, first_seen_stage, loss_stage = _stage_diagnostics(
        response,
        gold_ids,
        gold_titles,
        gold_pmids,
        gold_dois,
        gold_corpus_ids,
    )
    return PaperTestQueryResult(
        qid=str(sample["qid"]),
        query=str(sample["question"]).strip(),
        relevant_count=relevant_count,
        returned_count=len(response.papers),
        matched_titles=matches,
        metrics={str(k): _metric_at_k(flags, relevant_count=relevant_count, k=k) for k in ks},
        latency_ms=latency_ms,
        api_calls=max(0, int(response.meta.get("api_calls") or 0)),
        token_usage=max(0, int(response.meta.get("token_usage") or 0)),
        stage_metrics=stage_metrics,
        first_seen_stage=first_seen_stage,
        loss_stage=loss_stage,
        provider_diagnostics=_provider_diagnostics(response),
        provider_snapshot_sha256=str(
            response.meta.get("provider_snapshot_sha256") or ""
        ) or None,
        candidate_trace=response.meta.get("candidate_trace") or None,
    )


def _failed_query(sample: dict[str, Any], ks: list[int], latency_ms: float, error: str) -> PaperTestQueryResult:
    gold_dois = _sample_values(sample, "answer_doi", "answer_dois", "gold_doi")
    gold_corpus_ids = _sample_values(
        sample,
        "answer_corpus_id",
        "answer_corpus_ids",
        "answer_semantic_scholar_corpus_id",
    )
    relevant_count = len(
        _gold_records(
            [str(value) for value in sample.get("answer_arxiv_id") or []],
            [str(value) for value in sample.get("answer") or []],
            [str(value) for value in sample.get("answer_pmid") or []],
            gold_dois,
            gold_corpus_ids,
        )
    )
    return PaperTestQueryResult(
        qid=str(sample["qid"]),
        query=str(sample["question"]).strip(),
        relevant_count=relevant_count,
        returned_count=0,
        metrics={str(k): _metric_at_k([], relevant_count=relevant_count, k=k) for k in ks},
        latency_ms=latency_ms,
        error=error,
        error_category=_classify_query_error(error),
    )


def _aggregate(
    queries: list[PaperTestQueryResult],
    ks: list[int],
    *,
    include_failures: bool = True,
) -> dict[str, PaperTestMetric]:
    if not include_failures:
        queries = [query for query in queries if query.error is None]
    denominator = len(queries) or 1
    result: dict[str, PaperTestMetric] = {}
    for k in ks:
        values = [query.metrics[str(k)] for query in queries]
        result[str(k)] = PaperTestMetric(
            k=k,
            precision=sum(item.precision for item in values) / denominator,
            recall=sum(item.recall for item in values) / denominator,
            f1=sum(item.f1 for item in values) / denominator,
            ndcg=sum(item.ndcg for item in values) / denominator,
            mrr=sum(item.mrr for item in values) / denominator,
            hits=sum(item.hits for item in values),
        )
    return result


def _aggregate_asta(queries: list[PaperTestQueryResult]) -> AstaOfficialMetric | None:
    values = [item.asta_official for item in queries if item.asta_official is not None]
    if not values:
        return None
    semantic = [item.adjusted_f1 for item in values if item.adjusted_f1 is not None]
    specific = [item.standard_f1 for item in values if item.standard_f1 is not None]
    statuses = [item.judge_status for item in values]
    degraded = [status for status in statuses if status not in {"ok", "not_required"}]
    return AstaOfficialMetric(
        score_type="paper_finder_macro",
        adjusted_f1=(sum(semantic) / len(semantic)) if semantic else None,
        standard_f1=(sum(specific) / len(specific)) if specific else None,
        estimated_recall=sum(item.estimated_recall for item in values) / len(values),
        ndcg=sum(item.ndcg for item in values) / len(values),
        corpus_id_validity=sum(item.corpus_id_validity for item in values) / len(values),
        valid_corpus_ids=sum(item.valid_corpus_ids for item in values),
        submitted_papers=sum(item.submitted_papers for item in values),
        perfectly_relevant=sum(item.perfectly_relevant for item in values),
        judge_status=(f"degraded:{len(degraded)}/{len(values)}" if degraded else "ok"),
    )


def _nearest_rank_percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(max(0.0, float(value)) for value in values)
    rank = max(1, min(len(ordered), math.ceil(len(ordered) * quantile)))
    return ordered[rank - 1]


def _build_report(request: PaperTestRequest, dataset: PaperTestDatasetInfo, queries: list[PaperTestQueryResult], run_id: str) -> PaperTestReport:
    total_latency = sum(item.latency_ms for item in queries)
    latencies = [item.latency_ms for item in queries]
    failed = sum(item.error is not None for item in queries)
    failure_breakdown: dict[str, int] = {}
    for item in queries:
        if item.error is None:
            continue
        category = item.error_category or _classify_query_error(item.error) or "other"
        failure_breakdown[category] = failure_breakdown.get(category, 0) + 1
    snapshot_rows = sorted(
        (item.qid, item.provider_snapshot_sha256)
        for item in queries
        if item.provider_snapshot_sha256
    )
    provider_snapshot_sha256 = (
        hashlib.sha256(
            json.dumps(
                snapshot_rows,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if snapshot_rows
        else None
    )
    run_metadata = _run_metadata(
        request,
        dataset,
        provider_snapshot_sha256=provider_snapshot_sha256,
        sample_qids=[item.qid for item in queries],
        provider_snapshot_query_count=len(snapshot_rows),
    )
    return PaperTestReport(
        run_id=run_id,
        dataset=dataset,
        sample_limit=request.sample_limit,
        sample_count=len(queries),
        total_relevant=sum(item.relevant_count for item in queries),
        ks=request.ks,
        seed=request.seed,
        use_mock=request.use_mock,
        aggregate=_aggregate(queries, request.ks),
        aggregate_successful=_aggregate(
            queries,
            request.ks,
            include_failures=False,
        ),
        queries=queries,
        successful_queries=len(queries) - failed,
        failed_queries=failed,
        failure_breakdown=failure_breakdown,
        failure_rate=(failed / len(queries)) if queries else 0.0,
        mean_latency_ms=total_latency / len(queries) if queries else 0.0,
        p50_latency_ms=_nearest_rank_percentile(latencies, 0.50),
        p95_latency_ms=_nearest_rank_percentile(latencies, 0.95),
        total_latency_ms=total_latency,
        total_api_calls=sum(item.api_calls for item in queries),
        total_token_usage=sum(item.token_usage for item in queries),
        generated_at=datetime.now(UTC).isoformat(),
        matching_policy=(
            "semantic-scholar-corpus-id+asta-judge+official-normalizer-v1"
            if dataset.evaluation_family == "astabench"
            else (
                "pmid-exact-one-to-one-v1; macro-average; failed queries score zero"
                if dataset.evaluation_family == "trec-pm"
                else (
                    "normalized-title-proxy-one-to-one-v2; open-provider non-official BEIR; "
                    "ID-first (arXiv/PMID/DOI/CorpusID), title fallback; "
                    "macro-average; failed queries score zero; successful-only view also reported"
                    if dataset.evaluation_family == "beir-open"
                    else "arxiv-id-or-pmid-or-doi-or-corpus-id-or-title-one-to-one-v4; "
                    "ID-first, title fallback; macro-average; failed queries score zero; "
                    "successful-only view also reported"
                )
            )
        ),
        asta_official=_aggregate_asta(queries),
        dataset_sha256=run_metadata.dataset_sha256,
        run_metadata=run_metadata,
        provider_snapshot_sha256=provider_snapshot_sha256,
    )


def _write_report_atomic(report: PaperTestReport, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump_json(indent=2)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
        handle.write(payload)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(target)


def _run_filename(report: PaperTestReport) -> str:
    generated = datetime.fromisoformat(report.generated_at.replace("Z", "+00:00"))
    timestamp = generated.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    safe_run_id = re.sub(r"[^A-Za-z0-9_-]", "", report.run_id)[:64]
    if not safe_run_id:
        raise ValueError("report run_id cannot be used as a local filename")
    return f"{timestamp}_{safe_run_id}.json"


def _archive_report(report: PaperTestReport) -> Path:
    target = RUNS_DIR / _run_filename(report)
    if not target.is_file():
        _write_report_atomic(report, target)
    return target


def _save_report(report: PaperTestReport) -> None:
    """Persist an immutable per-run report, then atomically advance latest."""
    _archive_report(report)
    _write_report_atomic(report, RESULTS_DIR / "latest.json")


def load_latest_paper_test_report() -> PaperTestReport | None:
    path = RESULTS_DIR / "latest.json"
    if not path.is_file():
        return None
    try:
        report = PaperTestReport.model_validate_json(path.read_text(encoding="utf-8"))
        # Transparently migrate the pre-history latest.json from older versions.
        _archive_report(report)
        return report
    except (OSError, ValueError):
        return None


def _summary(report: PaperTestReport) -> PaperTestRunSummary:
    stage_recall: dict[str, float] = {}
    for stage in ("provider", "merged", "budget", "ranking", "final"):
        values = [
            query.stage_metrics[stage].candidate_recall
            for query in report.queries
            if stage in query.stage_metrics
        ]
        if values:
            # Missing stage diagnostics (for example a failed query) score as
            # zero so the history value remains comparable with macro metrics.
            stage_recall[stage] = sum(values) / max(1, report.sample_count)
    return PaperTestRunSummary(
        run_id=report.run_id,
        dataset_id=report.dataset.id,
        dataset_name=report.dataset.name,
        evaluation_family=report.dataset.evaluation_family,
        sample_count=report.sample_count,
        total_relevant=(
            report.total_relevant
            or sum(item.relevant_count for item in report.queries)
        ),
        successful_queries=report.successful_queries,
        failed_queries=report.failed_queries,
        failure_breakdown=report.failure_breakdown,
        failure_rate=report.failure_rate,
        ks=report.ks,
        max_k=max(report.ks),
        seed=report.seed,
        aggregate=report.aggregate,
        aggregate_successful=report.aggregate_successful,
        stage_recall=stage_recall,
        asta_official=report.asta_official,
        mean_latency_ms=report.mean_latency_ms,
        p50_latency_ms=report.p50_latency_ms,
        p95_latency_ms=report.p95_latency_ms,
        total_latency_ms=report.total_latency_ms,
        total_api_calls=report.total_api_calls,
        total_token_usage=report.total_token_usage,
        generated_at=report.generated_at,
        dataset_sha256=report.dataset_sha256,
        config_fingerprint=(
            report.run_metadata.config_fingerprint
            if report.run_metadata is not None
            else None
        ),
        provider_snapshot_sha256=(
            report.provider_snapshot_sha256
            or (
                report.run_metadata.provider_snapshot_sha256
                if report.run_metadata is not None
                else None
            )
        ),
    )


def list_paper_test_history(*, limit: int = 50, offset: int = 0) -> PaperTestHistoryResponse:
    load_latest_paper_test_report()
    reports: list[PaperTestRunSummary] = []
    if RUNS_DIR.is_dir():
        for path in sorted(RUNS_DIR.glob("*.json"), reverse=True):
            try:
                report = PaperTestReport.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            reports.append(_summary(report))
    return PaperTestHistoryResponse(
        items=reports[offset : offset + limit],
        total=len(reports),
        limit=limit,
        offset=offset,
    )


def load_paper_test_report(run_id: str) -> PaperTestReport | None:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id):
        return None
    load_latest_paper_test_report()
    if not RUNS_DIR.is_dir():
        return None
    for path in RUNS_DIR.glob(f"*_{run_id}.json"):
        try:
            report = PaperTestReport.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if report.run_id == run_id:
            return report
    return None


SearchRunner = Callable[..., SearchResponse]


def iter_paper_test_events(
    request: PaperTestRequest,
    *,
    search_runner: SearchRunner = run_paper_search,
) -> Iterator[dict[str, Any]]:
    dataset = validate_paper_test_sample_limit(request)
    if not dataset.available or not dataset.source_path:
        raise FileNotFoundError(
            "Paper test dataset is unavailable. Set PAPER_TEST_DATASET_ROOT or place local "
            "data in backend/paper-test/datasets; for AstaBench, configure HF_ACCESS_TOKEN "
            "or HF_TOKEN."
        )
    if dataset.evaluation_family == "astabench":
        samples = load_asta_samples(
            dataset.split,
            limit=request.sample_limit,
            seed=request.seed,
        )
    else:
        samples = _load_samples(
            Path(dataset.source_path),
            limit=request.sample_limit,
            seed=request.seed,
        )
    # AstaBench may be downloaded lazily, so its exact count can be unknown
    # when the dataset list is first rendered. Validate again after loading.
    if request.sample_limit > len(samples):
        raise ValueError(
            f"抽样数量不能超过测试集 {dataset.name} 的总量 {len(samples)}"
        )
    run_id = uuid4().hex
    queries: list[PaperTestQueryResult] = []
    yield {
        "type": "start",
        "run_id": run_id,
        "dataset": dataset.model_dump(mode="json"),
        "total": len(samples),
        "ks": request.ks,
        "provider_cache_mode": request.provider_cache_mode,
    }
    for index, sample in enumerate(samples, start=1):
        yield {
            "type": "progress",
            "current": index,
            "total": len(samples),
            "qid": sample["qid"],
            "query": str(sample["question"]).strip(),
        }
        started = time.perf_counter()
        try:
            search_kwargs = {
                "max_papers": max(request.ks),
                "use_mock": request.use_mock,
            }
            if search_runner is run_paper_search:
                search_kwargs["evaluation_fast_mode"] = get_settings().paper_test_fast_mode
                search_kwargs["provider_cache_mode"] = request.provider_cache_mode
                search_kwargs["benchmark_mode"] = True
            response = search_runner(str(sample["question"]).strip(), **search_kwargs)
            if dataset.evaluation_family == "astabench":
                asta = score_asta_response(sample, response)
                latency_ms = (time.perf_counter() - started) * 1000
                query_result = PaperTestQueryResult(
                    qid=str(sample["qid"]),
                    query=str(sample["question"]).strip(),
                    relevant_count=asta.relevant_count,
                    returned_count=len(response.papers),
                    matched_titles=asta.matched_titles,
                    metrics={
                        str(k): _metric_at_k(
                            asta.perfect_flags,
                            relevant_count=asta.relevant_count,
                            k=k,
                        )
                        for k in request.ks
                    },
                    latency_ms=latency_ms,
                    api_calls=max(0, int(response.meta.get("api_calls") or 0)),
                    token_usage=max(0, int(response.meta.get("token_usage") or 0)),
                    asta_official=asta.official,
                    provider_diagnostics=_provider_diagnostics(response),
                    provider_snapshot_sha256=str(
                        response.meta.get("provider_snapshot_sha256") or ""
                    ) or None,
                )
            else:
                latency_ms = (time.perf_counter() - started) * 1000
                query_result = score_search_response(
                    sample,
                    response,
                    request.ks,
                    latency_ms=latency_ms,
                )
        except FrozenProviderCacheMiss:
            # A partially frozen run is not a valid benchmark.  Fail before
            # saving it instead of silently turning cache misses into zeros.
            raise
        except Exception as exc:  # noqa: BLE001 - one bad query must not discard the run
            latency_ms = (time.perf_counter() - started) * 1000
            query_result = _failed_query(sample, request.ks, latency_ms, str(exc))
        queries.append(query_result)
        yield {
            "type": "query_result",
            "current": index,
            "total": len(samples),
            "query_result": query_result.model_dump(mode="json"),
        }
    report = _build_report(request, dataset, queries, run_id)
    _save_report(report)
    yield {"type": "result", "report": report.model_dump(mode="json")}


def run_paper_test(
    request: PaperTestRequest,
    *,
    search_runner: SearchRunner = run_paper_search,
) -> PaperTestReport:
    final: PaperTestReport | None = None
    for event in iter_paper_test_events(request, search_runner=search_runner):
        if event["type"] == "result":
            final = PaperTestReport.model_validate(event["report"])
    if final is None:
        raise RuntimeError("paper test did not produce a report")
    return final
