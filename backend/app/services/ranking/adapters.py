"""Compatibility adapters for the current feature-one and feature-two outputs."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from backend.app.models.evidrank import (
    CandidatePaper,
    Constraint,
    ConstraintHardness,
    ConstraintPolarity,
    FacetDefinition,
    FieldProvenance,
    MetadataConflict,
    PaperSource,
    PaperVersion,
    Relation,
    ResearchIntent,
)
from backend.app.models.schemas import QueryUnderstanding


def _stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _normalize_document_language(value: Any) -> str | None:
    normalized = str(value or "").strip().casefold().replace("_", "-")
    if normalized in {"en", "eng", "english", "en-us", "en-gb"}:
        return "en"
    if normalized in {"zh", "zho", "chi", "chinese", "zh-cn", "zh-tw"}:
        return "zh"
    return None


def detect_document_language(title: str, abstract: str | None) -> str | None:
    """Cheap deterministic zh/en detection for missing provider metadata."""
    sample = f"{title}\n{abstract or ''}"[:6000]
    latin = sum(char.isascii() and char.isalpha() for char in sample)
    cjk = sum("\u4e00" <= char <= "\u9fff" for char in sample)
    if cjk >= 4 and cjk >= latin * 0.2:
        return "zh"
    if latin >= 20 and latin >= cjk * 4:
        return "en"
    return None


def _explicit_year_source_span(query: str) -> str:
    """Return the exact user-authored year phrase used for a legacy hard gate."""
    patterns = (
        r"(?:since|after|before|until|newer\s+than|no\s+later\s+than)\s+(?:19|20)\d{2}",
        r"(?:19|20)\d{2}\s*年?\s*(?:到|至|-|—|–)\s*(?:19|20)\d{2}\s*年?",
        r"(?:19|20)\d{2}\s*年?\s*(?:以后|之后|以来|起|以前|之前|截至|不晚于)",
        r"近\s*\d+\s*年",
        r"(?:19|20)\d{2}\s*年?",
    )
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return match.group(0)
    return ""


def _parse_constraints(raw: dict[str, Any]) -> list[Constraint]:
    parsed: list[Constraint] = []
    for index, item in enumerate(raw.get("constraints") or []):
        if not isinstance(item, dict):
            continue
        try:
            parsed.append(
                Constraint(
                    id=str(item.get("id") or f"C{index + 1:03d}"),
                    facet=str(item.get("facet") or item.get("field") or "unknown"),
                    operator=str(item.get("operator") or "contains"),
                    value=item.get("value"),
                    hardness=ConstraintHardness(str(item.get("hardness") or "soft").lower()),
                    polarity=ConstraintPolarity(str(item.get("polarity") or "positive").lower()),
                    scope=str(item.get("scope") or "global"),
                    source_span=dict(item.get("source_span") or {}),
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid ResearchIntent constraint at index {index}: {exc}") from exc
    return parsed


def _parse_relations(raw: dict[str, Any]) -> list[Relation]:
    parsed: list[Relation] = []
    for index, item in enumerate(raw.get("relations") or []):
        if not isinstance(item, dict):
            raise ValueError(f"invalid ResearchIntent relation at index {index}: expected object")
        source = item.get("source")
        relation = item.get("relation") or item.get("type")
        target = item.get("target")
        if source and relation and target:
            parsed.append(
                Relation(
                    source=str(source),
                    relation=str(relation),
                    target=str(target),
                    required=bool(item.get("required", True)),
                )
            )
        else:
            raise ValueError(
                f"invalid ResearchIntent relation at index {index}: source/relation/target required"
            )
    return parsed


def research_intent_from_understanding(
    *,
    session_id: str,
    query: str,
    understanding: QueryUnderstanding,
) -> ResearchIntent:
    """Build the formal feature-three intent without inventing hard constraints."""
    raw = dict(understanding.raw_json or {})
    constraints = _parse_constraints(raw)
    relations = _parse_relations(raw)

    dynamic_facets: dict[str, FacetDefinition] = {}
    raw_facets = raw.get("dynamic_facets") or {}
    if isinstance(raw_facets, dict):
        for facet_id, value in raw_facets.items():
            data = value if isinstance(value, dict) else {"values": [str(value)]}
            dynamic_facets[str(facet_id)] = FacetDefinition(
                facet_id=str(facet_id),
                label=str(data.get("label") or facet_id),
                values=[str(v) for v in data.get("values") or []],
                is_hard=bool(data.get("is_hard", False)),
                target_count=max(1, int(data.get("target_count") or 1)),
            )

    # Dynamic facets are not presentation-only columns. They must enter the
    # same evidence contract as ordinary constraints so gaps and uncertainty
    # items can be resolved rather than remaining permanently absent.
    existing_constraint_facets = {constraint.facet for constraint in constraints}
    for facet_id, definition in dynamic_facets.items():
        if facet_id in existing_constraint_facets:
            continue
        constraints.append(
            Constraint(
                id=f"dynamic:{facet_id}",
                facet=facet_id,
                operator="contains",
                value=definition.values,
                hardness=(
                    ConstraintHardness.HARD
                    if definition.is_hard
                    else ConstraintHardness.SOFT
                ),
            )
        )

    # The legacy feature-one schema has only a time-range string.  Treat it as
    # hard only when it can be traced to an explicit year expression.
    has_explicit_year_constraint = any(
        constraint.facet.casefold() == "year" for constraint in constraints
    )
    if (
        not has_explicit_year_constraint
        and understanding.time_range
        and re.search(r"(19|20)\d{2}|近\s*\d+\s*年", query)
    ):
        years = [int(y) for y in re.findall(r"(19\d{2}|20\d{2})", understanding.time_range)]
        source_span = {"quoted_text": _explicit_year_source_span(query)}
        if years:
            if len(set(years)) >= 2:
                constraints.extend([
                    Constraint(
                        id="legacy-year-min",
                        facet="year",
                        operator="gte",
                        value=min(years),
                        hardness=ConstraintHardness.HARD,
                        source_span=source_span,
                    ),
                    Constraint(
                        id="legacy-year-max",
                        facet="year",
                        operator="lte",
                        value=max(years),
                        hardness=ConstraintHardness.HARD,
                        source_span=source_span,
                    ),
                ])
            elif re.search(r"以后|之后|以来|起|since|after|newer than", query, re.IGNORECASE):
                constraints.append(
                    Constraint(
                        id="legacy-year-min",
                        facet="year",
                        operator="gte",
                        value=years[0],
                        hardness=ConstraintHardness.HARD,
                        source_span=source_span,
                    )
                )
            elif re.search(r"以前|之前|截至|不晚于|before|until|no later than", query, re.IGNORECASE):
                constraints.append(
                    Constraint(
                        id="legacy-year-max",
                        facet="year",
                        operator="lte",
                        value=years[0],
                        hardness=ConstraintHardness.HARD,
                        source_span=source_span,
                    )
                )
            else:
                constraints.append(
                    Constraint(
                        id="legacy-year-exact",
                        facet="year",
                        operator="eq",
                        value=years[0],
                        hardness=ConstraintHardness.HARD,
                        source_span=source_span,
                    )
                )

    # Legacy methods/datasets are evidence facets, not automatic hard gates.
    existing_facets = {c.facet for c in constraints}
    for index, value in enumerate(understanding.methods):
        if value and "method" not in existing_facets:
            constraints.append(
                Constraint(
                    id=f"legacy-method-{index + 1}",
                    facet="method",
                    operator="contains",
                    value=value,
                    hardness=ConstraintHardness.SOFT,
                    scope="core",
                )
            )
    for index, value in enumerate(understanding.datasets):
        if value and "dataset" not in existing_facets:
            constraints.append(
                Constraint(
                    id=f"legacy-dataset-{index + 1}",
                    facet="dataset",
                    operator="contains",
                    value=value,
                    hardness=ConstraintHardness.SOFT,
                    scope="core",
                )
            )

    background_policy = str(raw.get("background_policy") or "allow_partial")
    output_intent = [str(v) for v in raw.get("output_intent") or ["ranked_list"]]
    parser_version = str(raw.get("parser_version") or "legacy-query-understanding-adapter-v1")
    intent_payload = {
        "query": query,
        "tasks": understanding.topics,
        "methods": understanding.methods,
        "datasets": understanding.datasets,
        "relations": [item.model_dump(mode="json") for item in relations],
        "constraints": [item.model_dump(mode="json") for item in constraints],
        "dynamic_facets": {k: v.model_dump(mode="json") for k, v in dynamic_facets.items()},
        "background_policy": background_policy,
        "output_intent": output_intent,
        "parser_version": parser_version,
    }
    return ResearchIntent(
        session_id=session_id,
        raw_query=query,
        tasks=list(understanding.topics),
        methods=list(understanding.methods),
        datasets=list(understanding.datasets),
        relations=relations,
        constraints=constraints,
        dynamic_facets=dynamic_facets,
        background_policy=background_policy,
        output_intent=output_intent,
        intent_hash=_stable_hash(intent_payload),
        parser_version=parser_version,
    )


def candidate_from_row(row: dict[str, Any], provider_rank: int) -> CandidatePaper:
    """Adapt a registry row; canonicalization remains an upstream responsibility."""
    canonical_id = str(row.get("canonical_paper_id") or row.get("id") or "")
    if not canonical_id:
        canonical_id = f"missing:{_stable_hash(row)[:16]}"
    preferred_version_id = str(row.get("preferred_version_id") or f"{canonical_id}:preferred")
    source = str(row.get("source") or row.get("provider") or "unknown")
    provider_score = float(row.get("score") or row.get("provider_score") or 0.0)
    declared_language = _normalize_document_language(row.get("language"))
    language = declared_language or detect_document_language(
        str(row.get("title") or ""),
        str(row.get("abstract") or ""),
    )

    versions: list[PaperVersion] = []
    for value in row.get("versions") or []:
        try:
            versions.append(PaperVersion.model_validate(value))
        except (TypeError, ValueError):
            continue
    if not versions:
        versions.append(
            PaperVersion(
                version_id=preferred_version_id,
                version_type=str(row.get("version_type") or "unknown"),
                title=str(row.get("title") or ""),
                year=row.get("year"),
                is_retracted=bool(row.get("is_retracted", False)),
            )
        )

    provenance: list[PaperSource] = []
    for value in row.get("provenance") or []:
        try:
            provenance.append(PaperSource.model_validate(value))
        except (TypeError, ValueError):
            continue
    if not provenance:
        confidence = "weak" if source in {"tavily", "web", "unknown"} else "trusted"
        provenance.append(
            PaperSource(
                provider=source,
                provider_id=str(row.get("provider_id") or canonical_id),
                response_hash=row.get("response_hash"),
                confidence_level=confidence,
            )
        )

    field_provenance: list[FieldProvenance] = []
    for value in row.get("field_provenance") or []:
        try:
            field_provenance.append(FieldProvenance.model_validate(value))
        except (TypeError, ValueError):
            continue
    if not field_provenance:
        confidence = "weak" if source in {"tavily", "web", "unknown"} else "trusted"
        for field in ("title", "abstract", "authors", "year", "venue", "publication_type"):
            value = row.get(field)
            if value not in (None, "", []):
                field_provenance.append(
                    FieldProvenance(
                        field=field,
                        value=value,
                        provider=source,
                        response_hash=row.get("response_hash"),
                        confidence_level=confidence,
                    )
                )
    if language and not any(item.field == "language" for item in field_provenance):
        field_provenance.append(
            FieldProvenance(
                field="language",
                value=language,
                provider=(source if declared_language else "local-language-heuristic-v1"),
                response_hash=row.get("response_hash"),
                confidence_level=(
                    "weak"
                    if declared_language and source in {"tavily", "web", "unknown"}
                    else "trusted"
                ),
            )
        )

    conflicts: list[MetadataConflict] = []
    for value in row.get("metadata_conflicts") or []:
        try:
            conflicts.append(MetadataConflict.model_validate(value))
        except (TypeError, ValueError):
            continue

    provider_scores = {str(k): float(v) for k, v in (row.get("provider_scores") or {}).items()}
    provider_ranks = {str(k): int(v) for k, v in (row.get("provider_ranks") or {}).items()}
    provider_scores.setdefault(source, provider_score)
    provider_ranks.setdefault(source, provider_rank)

    return CandidatePaper(
        canonical_paper_id=canonical_id,
        preferred_version_id=preferred_version_id,
        title=str(row.get("title") or "Untitled"),
        abstract=str(row.get("abstract") or "") or None,
        authors=[str(v) for v in row.get("authors") or []],
        year=int(row["year"]) if row.get("year") else None,
        language=language,
        venue=row.get("venue"),
        publication_type=row.get("publication_type"),
        identifiers={
            str(k).casefold(): str(v)
            for k, v in (row.get("identifiers") or row.get("external_ids") or {}).items()
            if v not in (None, "")
        },
        versions=versions,
        provenance=provenance,
        field_provenance=field_provenance,
        provider_scores=provider_scores,
        provider_ranks=provider_ranks,
        oa_status=row.get("oa_status"),
        pdf_url=row.get("pdf_url"),
        url=row.get("url"),
        citation_count=int(row.get("citation_count") or 0),
        tags=[str(v) for v in row.get("tags") or []],
        metadata_conflicts=conflicts,
        is_retracted=bool(row.get("is_retracted", False)),
        is_duplicate_version=bool(row.get("is_duplicate_version", False)),
    )


def candidates_from_rows(rows: list[dict[str, Any]]) -> list[CandidatePaper]:
    papers = [candidate_from_row(row, index) for index, row in enumerate(rows, start=1)]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for paper in papers:
        if paper.canonical_paper_id in seen:
            duplicates.add(paper.canonical_paper_id)
        seen.add(paper.canonical_paper_id)
    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise ValueError(
            "Candidate Registry snapshot contains duplicate canonical_paper_id values: "
            f"{duplicate_list}"
        )
    return papers
