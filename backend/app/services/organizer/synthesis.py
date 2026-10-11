"""Generate conservative, deterministic feature-four synthesis claims."""

from __future__ import annotations

import hashlib

from backend.app.models.evidrank import EvidenceLabel, FinalRankingBundle, ResearchIntent
from backend.app.models.scimap import (
    ClaimValidationSummary,
    EvidenceSynthesis,
    SynthesisClaim,
)
from backend.app.services.organizer.claim_validator import validate_claims


def _claim_id(kind: str, key: str) -> str:
    digest = hashlib.sha256(f"{kind}\x1f{key}".encode("utf-8")).hexdigest()[:16]
    return f"claim-{kind}-{digest}"


def build_synthesis(
    bundle: FinalRankingBundle,
    intent: ResearchIntent,
) -> tuple[EvidenceSynthesis, ClaimValidationSummary]:
    del intent  # The bundle already contains intent-aligned facet decisions.
    visible_ids = set(bundle.high + bundle.partial)
    records = [
        record
        for record in bundle.evidence_matrix
        if record.paper_id in visible_ids and record.validation_status == "valid"
    ]
    item_by_id = {item.paper_id: item for item in bundle.items}
    by_facet: dict[str, list] = {}
    for record in records:
        by_facet.setdefault(record.facet_id, []).append(record)

    claims: list[SynthesisClaim] = []
    for facet, facet_records in sorted(by_facet.items()):
        support = [r for r in facet_records if r.label == EvidenceLabel.SUPPORT]
        violate = [r for r in facet_records if r.label == EvidenceLabel.VIOLATE]
        support_papers = sorted({r.paper_id for r in support})
        violate_papers = sorted({r.paper_id for r in violate})
        if support_papers:
            claim_type = "consensus" if len(support_papers) >= 2 else "evidence"
            cited_titles = [f"《{item_by_id[paper_id].title}》" for paper_id in support_papers[:3]]
            title_text = "、".join(cited_titles)
            if len(support_papers) > 3:
                title_text += f"等 {len(support_papers)} 篇论文"
            claims.append(
                SynthesisClaim(
                    claim_id=_claim_id(claim_type, facet),
                    claim_type=claim_type,
                    text=f"{title_text}中存在支持检索分面“{facet}”的可定位证据。",
                    paper_ids=support_papers,
                    evidence_ids=sorted(r.evidence_id for r in support),
                    strong_assertion=True,
                    assertion_level="B" if len(support_papers) >= 2 else "A",
                )
            )
        if support_papers and violate_papers:
            related = sorted(set(support_papers + violate_papers))
            claims.append(
                SynthesisClaim(
                    claim_id=_claim_id("disagreement", facet),
                    claim_type="disagreement",
                    text=f"返回论文对检索分面“{facet}”同时存在支持与违反证据，不能合并成一致结论。",
                    paper_ids=related,
                    evidence_ids=sorted(
                        r.evidence_id for r in support + violate
                    ),
                    strong_assertion=True,
                    assertion_level="C",
                )
            )

        support_with_year = [
            paper_id
            for paper_id in support_papers
            if item_by_id[paper_id].year is not None
        ]
        years = sorted({item_by_id[paper_id].year for paper_id in support_with_year})
        if len(support_with_year) >= 2 and len(years) >= 2:
            year_evidence = [
                record
                for record in support
                if record.paper_id in support_with_year
            ]
            claims.append(
                SynthesisClaim(
                    claim_id=_claim_id("trend", facet),
                    claim_type="trend",
                    text=(
                        f"在当前返回集内，分面“{facet}”的支持证据分布于 "
                        f"{years[0]}—{years[-1]} 年；这只描述证据的时间分布，"
                        "不推断性能提升或因果趋势。"
                    ),
                    paper_ids=support_with_year,
                    evidence_ids=sorted(record.evidence_id for record in year_evidence),
                    strong_assertion=True,
                    assertion_level="B",
                )
            )

    for gap in sorted(bundle.coverage_gaps, key=lambda value: value.gap_id):
        claims.append(
            SynthesisClaim(
                claim_id=_claim_id("gap", gap.gap_id),
                claim_type="gap",
                text=(
                    f"检索分面“{gap.facet_id}”当前有 {gap.supported_count} 篇论文获得支持证据，"
                    f"未达到目标 {gap.target_count} 篇。"
                ),
                strong_assertion=False,
                assertion_level="D",
            )
        )

    unknown_by_paper: dict[str, list[str]] = {}
    for item in bundle.items:
        if item.paper_id not in set(bundle.high + bundle.partial + bundle.verify):
            continue
        unknown = sorted(
            facet
            for facet, label in item.facet_labels.items()
            if label == EvidenceLabel.UNKNOWN
        )
        if unknown:
            unknown_by_paper[item.paper_id] = unknown
    for paper_id, facets in sorted(unknown_by_paper.items()):
        claims.append(
            SynthesisClaim(
                claim_id=_claim_id("incomparable", f"{paper_id}:{','.join(facets)}"),
                claim_type="incomparable",
                text=(
                    f"论文《{item_by_id[paper_id].title}》在分面“{'、'.join(facets)}”上的"
                    "证据状态为未知，不能据此进行确定性比较。"
                ),
                paper_ids=[paper_id],
                strong_assertion=False,
                assertion_level="D",
            )
        )

    validated, summary = validate_claims(claims, bundle)
    synthesis = EvidenceSynthesis(
        claims=validated,
        evidence_claim_ids=[c.claim_id for c in validated if c.claim_type == "evidence"],
        consensus_claim_ids=[c.claim_id for c in validated if c.claim_type == "consensus"],
        disagreement_claim_ids=[c.claim_id for c in validated if c.claim_type == "disagreement"],
        incomparable_claim_ids=[c.claim_id for c in validated if c.claim_type == "incomparable"],
        trend_claim_ids=[c.claim_id for c in validated if c.claim_type == "trend"],
        gap_claim_ids=[c.claim_id for c in validated if c.claim_type == "gap"],
    )
    return synthesis, summary
