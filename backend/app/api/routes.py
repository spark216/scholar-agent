"""API routes."""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from backend.app.config import get_settings
from backend.app.models.paper_test import (
    PaperTestDatasetInfo,
    PaperTestHistoryResponse,
    PaperTestReport,
    PaperTestRequest,
)
from backend.app.models.schemas import HealthResponse, SearchRequest, SearchResponse
from backend.app.services.paper_test import (
    iter_paper_test_events,
    list_paper_test_datasets,
    list_paper_test_history,
    load_latest_paper_test_report,
    load_paper_test_report,
    run_paper_test,
    validate_paper_test_sample_limit,
)
from backend.app.services.postgres_store import healthcheck as pg_health
from backend.app.services.search_service import (
    InvalidPaperQueryError,
    iter_paper_search_events,
    run_paper_search,
)

router = APIRouter()


def _local_model_snapshot_ready(path: Path) -> bool:
    """Check local model completeness without loading or downloading weights."""
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    return any(
        candidate.is_file() and candidate.stat().st_size >= 100 * 1024 * 1024
        for pattern in ("*.safetensors", "pytorch_model*.bin")
        for candidate in path.rglob(pattern)
    )


def _model_reference_ready(
    model_path: str,
    *,
    revision: str | None,
    cache_dir: str | None,
) -> bool:
    """Resolve a local path or an already-cached Hub revision without network I/O."""
    direct_path = Path(model_path)
    if _local_model_snapshot_ready(direct_path):
        return True
    try:
        from huggingface_hub import snapshot_download

        resolved = snapshot_download(
            repo_id=model_path,
            revision=revision,
            cache_dir=cache_dir or None,
            local_files_only=True,
        )
    except Exception:  # noqa: BLE001 - cache miss is a normal health state
        return False
    return _local_model_snapshot_ready(Path(resolved))


logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    milvus_status = "unknown"
    client = None
    try:
        from pymilvus import MilvusClient

        client = MilvusClient(uri=settings.milvus_uri)
        milvus_status = "ok" if client.list_collections() is not None else "error"
    except Exception as exc:  # noqa: BLE001
        milvus_status = f"error: {exc}"
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass

    if not settings.evidrank_enabled:
        evidrank_status = "disabled"
    elif settings.bge_local_files_only:
        model_references = [
            (
                "bge-m3",
                settings.bge_m3_model_path,
                settings.bge_m3_model_revision,
            )
        ]
        if settings.bge_reranker_enabled:
            model_references.append(
                (
                    "reranker",
                    settings.bge_reranker_model_path,
                    settings.bge_reranker_model_revision,
                )
            )
        missing = [
            label
            for label, model_path, revision in model_references
            if not _model_reference_ready(
                model_path,
                revision=revision,
                cache_dir=settings.bge_model_cache_dir,
            )
        ]
        evidrank_status = "ready" if not missing else f"model_missing: {','.join(missing)}"
    else:
        evidrank_status = "configured_remote_resolution"

    postgres_status = pg_health()
    overall_status = (
        "ok"
        if postgres_status == "ok"
        and milvus_status == "ok"
        and not evidrank_status.startswith("model_missing")
        else "degraded"
    )
    return HealthResponse(
        status=overall_status,
        service="huawei-paper-search",
        version="0.4.0",
        postgres=postgres_status,
        milvus=milvus_status,
        evidrank=evidrank_status,
    )


@router.post("/search", response_model=SearchResponse)
def search_papers(payload: SearchRequest) -> SearchResponse:
    settings = get_settings()
    use_mock = settings.use_mock if payload.use_mock is None else payload.use_mock

    try:
        return run_paper_search(
            payload.query,
            max_papers=payload.max_papers,
            use_mock=use_mock,
            controller_fulltext_documents=payload.controller_fulltext_documents,
            controller_fulltext_budget=payload.controller_fulltext_budget,
        )
    except InvalidPaperQueryError as exc:
        raise HTTPException(status_code=422, detail=exc.detail) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"搜索失败: {exc}") from exc


@router.post("/search/stream")
def search_papers_stream(payload: SearchRequest) -> StreamingResponse:
    """SSE stream: step / understanding / result / error."""
    settings = get_settings()
    use_mock = settings.use_mock if payload.use_mock is None else payload.use_mock

    def event_generator():
        try:
            for event in iter_paper_search_events(
                payload.query,
                max_papers=payload.max_papers,
                use_mock=use_mock,
                controller_fulltext_documents=payload.controller_fulltext_documents,
                controller_fulltext_budget=payload.controller_fulltext_budget,
            ):
                event_name = event.get("type", "message")
                data = {k: v for k, v in event.items() if k != "type"}
                yield f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Streaming paper search failed")
            payload_err = {"detail": f"搜索失败: {exc}"}
            yield f"event: error\ndata: {json.dumps(payload_err, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/paper-test/datasets", response_model=list[PaperTestDatasetInfo])
def paper_test_datasets() -> list[PaperTestDatasetInfo]:
    """List PaSa test sets available to the local ScholarAgent instance."""
    return list_paper_test_datasets()


@router.get("/paper-test/latest", response_model=PaperTestReport | None)
def paper_test_latest() -> PaperTestReport | None:
    """Return the most recently completed report without rerunning retrieval."""
    return load_latest_paper_test_report()


@router.get("/paper-test/history", response_model=PaperTestHistoryResponse)
def paper_test_history(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> PaperTestHistoryResponse:
    """List locally persisted evaluation runs, newest first."""
    return list_paper_test_history(limit=limit, offset=offset)


@router.get("/paper-test/history/{run_id}", response_model=PaperTestReport)
def paper_test_history_detail(run_id: str) -> PaperTestReport:
    """Load one complete historical evaluation report."""
    report = load_paper_test_report(run_id)
    if report is None:
        raise HTTPException(status_code=404, detail="评测历史记录不存在")
    return report


@router.post("/paper-test/run", response_model=PaperTestReport)
def paper_test_run(payload: PaperTestRequest) -> PaperTestReport:
    """Run a bounded PaSa evaluation synchronously (primarily for API clients)."""
    try:
        validate_paper_test_sample_limit(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    try:
        return run_paper_test(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"评测失败: {exc}") from exc


@router.post("/paper-test/run/stream")
def paper_test_run_stream(payload: PaperTestRequest) -> StreamingResponse:
    """Stream query-level PaSa evaluation progress and the final report."""
    try:
        validate_paper_test_sample_limit(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    def event_generator():
        try:
            for event in iter_paper_test_events(payload):
                event_name = str(event.get("type") or "message")
                data = {key: value for key, value in event.items() if key != "type"}
                yield f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Paper test failed")
            data = {"detail": f"评测失败: {exc}"}
            yield f"event: error\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
