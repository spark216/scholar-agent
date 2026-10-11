"""Application configuration loaded from environment."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    zhipuai_api_key: str = ""
    zhipu_model: str = "glm-4-flash"
    zhipu_intent_model: str = "glm-4-flash"

    tavily_api_key: str = ""
    tavily_max_results: int = 50
    tavily_search_depth: str = "advanced"

    # Academic search providers
    serper_api_key: str = ""
    semantic_scholar_enabled: bool = True
    semantic_scholar_api_key: str = ""
    # Semantic Scholar API keys start with a conservative 1 request/second
    # allowance. Serialize this provider without slowing the other sources.
    semantic_scholar_min_interval_seconds: float = 1.0
    # One-hop citation expansion is intentionally tiny and requires an API key.
    # It adds RefChain-style exploration without loading a full-corpus vector
    # index or making unauthenticated Semantic Scholar throttling worse.
    semantic_scholar_citation_expansion_enabled: bool = True
    semantic_scholar_citation_expansion_require_api_key: bool = True
    semantic_scholar_citation_expansion_limit: int = 20
    # PaSa-style citation tracing via OpenAlex (anonymous-compatible): expand
    # along citation edges every round after the first, using multiple
    # high-confidence seeds and both directions (newer work + foundational
    # work). Direction names match the OpenAlex filter values.
    semantic_scholar_citation_expansion_every_round: bool = True
    semantic_scholar_citation_expansion_seed_top_n: int = 3
    semantic_scholar_citation_expansion_directions: list[str] = ["cites", "referenced_works"]
    openalex_api_key: str = ""
    ncbi_api_key: str = ""
    ncbi_tool: str = "huawei-paper-search"
    ncbi_email: str = ""
    academic_min_candidates: int = 50
    academic_require_min_candidates: bool = False
    academic_provider_limit: int = 40
    academic_request_timeout: float = 20.0
    academic_connect_timeout: float = 5.0
    academic_pool_timeout: float = 2.0
    academic_request_max_attempts: int = 2
    academic_retry_backoff_seconds: float = 0.35
    academic_circuit_failure_threshold: int = 2
    academic_circuit_cooldown_seconds: float = 120.0
    academic_http_max_connections: int = 24
    academic_http_max_keepalive_connections: int = 12
    academic_local_corpus_enabled: bool = True
    academic_local_corpus_path: str = ""
    academic_local_index_path: str = ""
    academic_local_documents_path: str = ""
    academic_local_abstract_max_chars: int = 4000
    # Local FTS is cheap and deterministic, so retain a wider recall pool than
    # the network providers before the shared BGE candidate budget is applied.
    academic_local_provider_limit: int = 100
    # Optional offline BGE-M3 index over the full PaSa paper corpus. Keep it
    # disabled until backend/scripts/build_pasa_vector_index.py completes.
    academic_local_vector_enabled: bool = False
    academic_local_vector_collection: str = "pasa_full_corpus_bge_m3"
    academic_local_vector_provider_limit: int = 80
    academic_local_vector_recall_limit: int = 120
    # Vector backend for the full-corpus BGE-M3 index. "faiss" is the default:
    # it runs fully offline (no Docker/Milvus) and persists under
    # academic_local_vector_index_dir. "milvus" requires the Milvus service.
    academic_vector_backend: str = "faiss"
    academic_local_vector_index_dir: str = ""
    # Multi-variant vector retrieval (original query + intent parser rewrites).
    # Disable to isolate the contribution of query variants from reranking.
    academic_local_vector_multivariant: bool = True
    agent_search_max_iterations: int = 2
    agent_search_round_target: int = 40
    agent_search_min_new_candidates: int = 5
    agent_search_min_coverage: float = 0.6
    agent_search_max_api_calls: int = 12
    # Use one bounded LLM call only after the controller observes a genuine
    # recall/coverage gap. Evaluation-only forced second variants stay free.
    agent_search_llm_evolution_enabled: bool = True
    agent_search_llm_evolution_max_calls: int = 1
    agent_search_llm_evolution_max_tokens: int = 240
    agent_search_llm_evolution_prompt_version: str = "adaptive-query-v1"
    # Pseudo-relevance-feedback second-round search: when enabled, the
    # evolved query combines the user query with high-discrimination keywords
    # extracted from the high-confidence seed pool (title+abstract). When
    # disabled, the older generic topic-cue fallback is used (baseline).
    agent_search_prf_enabled: bool = True
    agent_search_tavily_limit: int = 20
    agent_search_tavily_conditional: bool = True
    agent_search_tavily_min_academic_candidates: int = 50
    agent_search_tavily_min_coverage: float = 0.8
    academic_provider_cache_enabled: bool = True
    academic_provider_cache_ttl_seconds: int = 604800
    academic_provider_cache_path: str = ""
    # Multi-angle arXiv retrieval (main query + sub-problems). Measured neutral
    # on pasa-real (no new gold, +API cost), so it is off by default.
    academic_arxiv_multi_query: bool = False

    postgres_uri: str = ""
    # PostgreSQL connect timeout in seconds. Kept short so a missing/stalled
    # database fails fast instead of blocking the search response for the OS
    # default TCP timeout (which can be ~2 minutes on Windows).
    postgres_connect_timeout: float = 3.0

    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_collection: str = "paper_search_bge_m3"

    # Local BGE-M3 (dense + sparse) + optional reranker
    bge_m3_model_path: str = "BAAI/bge-m3"
    bge_m3_model_revision: str = "5617a9f61b028005a4858fdac845db406aefb181"
    bge_model_cache_dir: str = ""
    bge_local_files_only: bool = False
    bge_m3_use_fp16: bool = True
    bge_m3_device: str = "auto"  # auto | cpu | mps | cuda
    bge_m3_batch_size: int = 8
    bge_m3_max_length: int = 512
    bge_embedding_cache_size: int = 4096
    bge_allow_hash_fallback: bool = False
    bge_m3_candidate_limit: int = 80
    bge_m3_max_candidate_limit: int = 100
    bge_candidate_relevance_fraction: float = 0.6
    # Reserve exact entity/phrase hits before semantic scoring.  This is a
    # recall guard for short queries and named datasets, not a relevance label.
    bge_candidate_anchor_fraction: float = 0.15
    bge_candidate_anchor_min_chars: int = 3
    # Reserve a bounded part of the embedding budget for the highest-ranked
    # papers from every provider before global lexical selection.  This keeps
    # one noisy/high-volume source from crowding out another source's best hit.
    bge_candidate_provider_head_fraction: float = 0.25
    bge_candidate_provider_head_per_source: int = 5
    rag_embedding_dim: int = 1024
    rag_top_k: int = 5
    hybrid_recall_limit: int = 50
    rrf_k: int = 60

    bge_reranker_enabled: bool = True
    bge_reranker_model_path: str = "BAAI/bge-reranker-v2-m3"
    bge_reranker_model_revision: str = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    # Optional fine-tuned Selector adapter (PaSA-style academic relevance).
    # When set to a PEFT LoRA checkpoint directory, the reranker loads the
    # base model plus this adapter instead of the base model alone.
    bge_reranker_selector_path: str = ""
    bge_reranker_device: str = "auto"
    bge_reranker_use_fp16: bool = True
    bge_reranker_batch_size: int = 8
    bge_reranker_max_length: int = 384
    bge_reranker_top_n: int = 40
    bge_reranker_cross_rank_weight: float = 0.65
    bge_reranker_facet_rank_weight: float = 0.15
    bge_reranker_head_protect: int = 3
    bge_reranker_cache_size: int = 8192
    bge_reranker_required: bool = True

    # PaSa 7B selector: optional additional ranking signal fused into
    # EvidRank's weighted RRF. Disabled by default; loading costs ~5GB VRAM
    # (4-bit) and ~38s, so it is opt-in (benchmark/lab only).
    pasa_selector_enabled: bool = False
    pasa_selector_required: bool = False
    pasa_selector_model_path: str = "models/pasa-7b-selector-sharded"
    pasa_selector_top_n: int = 80
    pasa_selector_batch_size: int = 8
    pasa_selector_max_length: int = 2048
    pasa_selector_rank_weight: float = 0.30
    pasa_selector_protect_head: bool = True
    pasa_selector_use_4bit: bool = True
    # Bounds for the 4-bit load (max_memory). The bf16 checkpoint materialises
    # during quantisation; without the CPU cap the load spikes to ~13GB RAM and
    # segfaults once the BGE models are already resident on a 16GB machine.
    pasa_selector_max_cpu_memory_gb: float = 6.0
    pasa_selector_max_gpu_memory_gb: float = 11.0

    # PaSa Crawler query fusion: merge pre-generated crawler search queries
    # into the LLM-intent understanding as AUXILIARY variants (never a
    # replacement). include_adaptive adds the coverage-gap follow-up set.
    crawler_query_fusion_enabled: bool = False
    crawler_query_include_adaptive: bool = False
    crawler_query_cache_path: str = ".modeling-audit/crawler_queries_cache.json"
    crawler_query_max_variants: int = 6

    # Feature three: auditable EvidRank pipeline
    evidrank_enabled: bool = True
    # ``recall_first`` keeps unresolved evidence visible instead of turning
    # every unknown facet into a verification item. ``strict`` preserves the
    # original evidence-first behaviour for audits that require it.
    evidrank_decision_policy: Literal["strict", "recall_first"] = "recall_first"
    evidrank_evidence_top_k: int = 5
    evidrank_evidence_pool_multiplier: int = 1
    evidrank_high_rank_cutoff: int = 5
    evidrank_default_facet_target: int = 1
    evidrank_threshold_version: str = "evidrank-v1-rank-evidence-rules"
    evidrank_threshold_profile: str = ""
    evidrank_evidence_llm_enabled: bool = True
    evidrank_evidence_llm_model: str = "glm-4-flash"
    evidrank_evidence_prompt_version: str = "evidence-facet-v2-json-batch"
    evidrank_evidence_max_tokens: int = 1600
    evidrank_evidence_batch_size: int = 5
    evidrank_evidence_max_retries: int = 0
    evidrank_evidence_timeout_seconds: float = 20.0

    # PaSa-backed retrieval evaluation. The root may point either to
    # ``paper-dataset`` or directly to its ``pasa-dataset`` child.
    paper_test_dataset_root: str = ""
    paper_test_fast_mode: bool = True
    # A benchmark query must exercise at least two genuinely different query
    # variants when the controller/API budgets permit it.  Interactive search
    # keeps its early-stop behaviour.
    paper_test_min_query_variants: int = 2
    # Benchmark-only recall profile.  It avoids slowing normal interactive
    # searches while still testing the larger pool recommended by the PaSa
    # stage-loss analysis.
    paper_test_bge_m3_candidate_limit: int = 120
    paper_test_bge_m3_max_candidate_limit: int = 160
    paper_test_bge_reranker_top_n: int = 80
    # Benchmark override for the PaSa selector scoring pool (fast mode).
    paper_test_pasa_selector_top_n: int = 80

    # legacy key kept for compatibility; ignored by BGE pipeline
    rag_embedding_model: str = "bge-m3"

    use_mock: bool = False
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
