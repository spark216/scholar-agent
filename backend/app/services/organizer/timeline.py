"""Create a publication timeline without inferring progress relations."""

from __future__ import annotations

from backend.app.models.evidrank import FinalRankingBundle, RankingLevel
from backend.app.models.scimap import TimelineEntry


def build_timeline(bundle: FinalRankingBundle) -> list[TimelineEntry]:
    evidence_by_paper: dict[str, list[str]] = {}
    for record in bundle.evidence_matrix:
        if record.validation_status == "valid":
            evidence_by_paper.setdefault(record.paper_id, []).append(record.evidence_id)

    entries = [
        TimelineEntry(
            paper_id=item.paper_id,
            title=item.title,
            year=item.year,
            level=item.level,
            final_rank=item.final_rank or 10**9,
            evidence_ids=sorted(evidence_by_paper.get(item.paper_id, [])),
        )
        for item in bundle.items
        if item.level in {RankingLevel.HIGH, RankingLevel.PARTIAL}
    ]
    return sorted(
        entries,
        key=lambda value: (
            value.year is None,
            value.year or 10**9,
            value.final_rank,
            value.paper_id,
        ),
    )
