"""Build deterministic paper evidence cards from a frozen ranking bundle."""

from __future__ import annotations

from backend.app.models.evidrank import EvidenceLabel, FinalRankingBundle, RankingLevel
from backend.app.models.scimap import EvidenceCitation, PaperEvidenceCard


def build_ranked_list(bundle: FinalRankingBundle) -> list[PaperEvidenceCard]:
    evidence_by_paper: dict[str, list[EvidenceCitation]] = {}
    effective_labels = {
        (item.paper_id, facet_id): label
        for item in bundle.items
        for facet_id, label in item.facet_labels.items()
    }
    for record in bundle.evidence_matrix:
        if (
            record.validation_status != "valid"
            or record.label == EvidenceLabel.UNKNOWN
            or effective_labels.get((record.paper_id, record.facet_id)) != record.label
        ):
            continue
        evidence_by_paper.setdefault(record.paper_id, []).append(
            EvidenceCitation(
                evidence_id=record.evidence_id,
                paper_id=record.paper_id,
                facet_id=record.facet_id,
                source_type=record.source_type,
                quoted_text=record.quoted_text,
                document_version_id=record.document_version_id,
                content_hash=record.content_hash,
                char_start=record.char_start,
                char_end=record.char_end,
            )
        )

    cards: list[PaperEvidenceCard] = []
    for item in bundle.items:
        if item.level == RankingLevel.EXCLUDED:
            continue
        labels = item.facet_labels
        cards.append(
            PaperEvidenceCard(
                paper_id=item.paper_id,
                title=item.title,
                authors=item.authors,
                year=item.year,
                venue=item.venue,
                level=item.level,
                final_rank=item.final_rank or len(cards) + 1,
                satisfied_facets=sorted(
                    facet for facet, label in labels.items() if label.value == "support"
                ),
                violated_facets=sorted(
                    facet for facet, label in labels.items() if label.value == "violate"
                ),
                unknown_facets=sorted(
                    facet for facet, label in labels.items() if label.value == "unknown"
                ),
                reason_codes=item.level_reason_codes,
                citations=sorted(
                    evidence_by_paper.get(item.paper_id, []),
                    key=lambda value: (value.facet_id, value.evidence_id),
                ),
            )
        )
    return sorted(cards, key=lambda value: (value.final_rank, value.paper_id))
