"""Non-compensatory hard-constraint gate."""

from __future__ import annotations

from typing import Any

from backend.app.models.evidrank import (
    CandidatePaper,
    Constraint,
    ConstraintHardness,
    ConstraintPolarity,
    GateDecision,
    GateResult,
    IdentityReview,
    IdentityStatus,
    MetadataValidation,
    ResearchIntent,
)


_METADATA_FACETS = {
    "year",
    # "language" is intentionally NOT a gateable metadata facet: the intent
    # parser emits hard constraints like `language contains zh+en` (the query
    # language, not a paper requirement) and candidate papers carry no
    # reliable language metadata, so a hard language gate would reject every
    # paper. Language constraints still flow to evidence judging as soft
    # facets; they just can never hard-exclude a candidate here.
    "publication_type",
    "paper_type",
    "venue",
    "paper_id",
    "canonical_paper_id",
    "retracted",
}


def _compare(actual: Any, constraint: Constraint) -> bool | None:
    if actual is None:
        return None
    operator = constraint.operator.lower()
    expected = constraint.value
    try:
        if operator in {"gte", ">="}:
            return float(actual) >= float(expected)
        if operator in {"gt", ">"}:
            return float(actual) > float(expected)
        if operator in {"lte", "<="}:
            return float(actual) <= float(expected)
        if operator in {"lt", "<"}:
            return float(actual) < float(expected)
    except (TypeError, ValueError):
        return None
    if operator in {"eq", "=", "equals"}:
        return str(actual).casefold() == str(expected).casefold()
    if operator in {"in", "one_of"}:
        values = expected if isinstance(expected, list) else [expected]
        return str(actual).casefold() in {str(v).casefold() for v in values}
    if operator in {"contains", "include"}:
        return str(expected).casefold() in str(actual).casefold()
    if operator in {"not_contains", "exclude"}:
        return str(expected).casefold() not in str(actual).casefold()
    return None


def _actual_value(paper: CandidatePaper, facet: str) -> Any:
    normalized = facet.lower()
    if normalized in {"paper_type", "publication_type"}:
        return paper.publication_type
    if normalized in {"paper_id", "canonical_paper_id"}:
        return paper.canonical_paper_id
    if normalized == "retracted":
        return paper.is_retracted
    return getattr(paper, normalized, None)


def apply_hard_gate(
    paper: CandidatePaper,
    intent: ResearchIntent,
    identity: IdentityReview,
    metadata: MetadataValidation,
) -> GateResult:
    violated: list[str] = []
    deferred: list[str] = []
    reasons: list[str] = []

    if paper.is_duplicate_version:
        violated.append("registry-duplicate-version")
        reasons.append("DUPLICATE_VERSION")
    if paper.is_retracted:
        violated.append("registry-retracted")
        reasons.append("CONFIRMED_RETRACTION")
    if identity.status != IdentityStatus.VALID:
        deferred.append("identity")
        reasons.extend(identity.reason_codes)

    trusted_fields = set(metadata.trusted_fields)
    for constraint in intent.constraints:
        if constraint.hardness != ConstraintHardness.HARD:
            continue
        facet = constraint.facet.lower()
        if facet not in _METADATA_FACETS:
            deferred.append(constraint.id)
            reasons.append(f"SEMANTIC_HARD_CONSTRAINT_DEFERRED:{constraint.id}")
            continue

        actual = _actual_value(paper, facet)
        result = _compare(actual, constraint)
        if result is None:
            deferred.append(constraint.id)
            reasons.append(f"HARD_METADATA_UNKNOWN:{constraint.id}")
            continue

        # Weak web-derived metadata is not strong enough to reject.
        provenance_field = "publication_type" if facet == "paper_type" else facet
        if provenance_field not in trusted_fields and facet not in {"paper_id", "canonical_paper_id", "retracted"}:
            deferred.append(constraint.id)
            reasons.append(f"HARD_METADATA_WEAK:{constraint.id}")
            continue

        satisfied = result
        if constraint.polarity == ConstraintPolarity.NEGATIVE:
            satisfied = not result
        if not satisfied:
            violated.append(constraint.id)
            reasons.append(f"RELIABLE_HARD_VIOLATION:{constraint.id}")

    if violated:
        decision = GateDecision.REJECT
    elif deferred:
        decision = GateDecision.DEFER
    else:
        decision = GateDecision.KEEP
    return GateResult(
        paper_id=paper.canonical_paper_id,
        decision=decision,
        reason_codes=list(dict.fromkeys(reasons)),
        violated_constraint_ids=violated,
        deferred_constraint_ids=deferred,
    )
