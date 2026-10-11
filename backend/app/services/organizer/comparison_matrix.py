"""Create a query-adaptive comparison matrix without filling unknown cells."""

from __future__ import annotations

from backend.app.models.evidrank import EvidenceLabel, FinalRankingBundle, ResearchIntent
from backend.app.models.scimap import ComparisonCell, ComparisonMatrix, ComparisonRow


def _facet_order(intent: ResearchIntent, bundle: FinalRankingBundle) -> list[str]:
    ordered = list(intent.dynamic_facets)
    ordered.extend(constraint.facet for constraint in intent.constraints)
    ordered.extend(
        facet
        for item in bundle.items
        for facet in item.facet_labels
    )
    return list(dict.fromkeys(value for value in ordered if value))


def build_comparison_matrix(
    bundle: FinalRankingBundle,
    intent: ResearchIntent,
) -> ComparisonMatrix:
    facets = _facet_order(intent, bundle)
    evidence_index: dict[tuple[str, str], list[str]] = {}
    for record in bundle.evidence_matrix:
        if record.validation_status == "valid":
            evidence_index.setdefault((record.paper_id, record.facet_id), []).append(
                record.evidence_id
            )

    rows: list[ComparisonRow] = []
    visible_ids = set(bundle.high + bundle.partial + bundle.verify)
    for item in bundle.items:
        if item.paper_id not in visible_ids:
            continue
        cells = {
            facet: ComparisonCell(
                facet_id=facet,
                status=item.facet_labels.get(facet, EvidenceLabel.UNKNOWN),
                evidence_ids=sorted(evidence_index.get((item.paper_id, facet), [])),
            )
            for facet in facets
        }
        rows.append(
            ComparisonRow(
                paper_id=item.paper_id,
                level=item.level,
                final_rank=item.final_rank or len(rows) + 1,
                cells=cells,
            )
        )
    return ComparisonMatrix(
        facets=facets,
        rows=sorted(rows, key=lambda value: (value.final_rank, value.paper_id)),
    )
