"""Conservative abstract/metadata evidence matrix builder.

The deterministic judge provides a safe baseline and exact spans.  Semantic
conditions that cannot be established by an exact span remain ``unknown``;
an optional LLM judge can later upgrade only those records.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from backend.app.models.evidrank import (
    CandidatePaper,
    Constraint,
    ConstraintPolarity,
    EvidenceLabel,
    EvidenceRecord,
    Relation,
    ResearchIntent,
)
from backend.app.services.ranking.evidence_span_validator import (
    normalized_text_hash,
    validate_evidence_span,
)


def _evidence_id(*parts: str) -> str:
    raw = "\x1f".join(parts)
    return "ev-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _metadata_record(
    *,
    session_id: str,
    paper: CandidatePaper,
    constraint: Constraint,
    label: EvidenceLabel,
    actual: Any,
    reason: str,
) -> EvidenceRecord:
    value = "" if actual is None else str(actual)
    return EvidenceRecord(
        evidence_id=_evidence_id(session_id, paper.canonical_paper_id, constraint.id, value),
        session_id=session_id,
        paper_id=paper.canonical_paper_id,
        paper_version_id=paper.preferred_version_id,
        facet_id=constraint.facet,
        constraint_id=constraint.id,
        label=label,
        confidence=1.0 if label != EvidenceLabel.UNKNOWN else 0.0,
        source_type="metadata",
        quoted_text=value or None,
        document_version_id=paper.preferred_version_id,
        content_hash=hashlib.sha256(value.encode("utf-8")).hexdigest() if value else None,
        normalized_text_hash=hashlib.sha256(value.encode("utf-8")).hexdigest() if value else None,
        parser_version="metadata-direct-v1",
        judge_model="deterministic-evidence-v1",
        validation_status="valid",
        reason_codes=[reason],
    )


def _compare_metadata(actual: Any, constraint: Constraint) -> bool | None:
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


def _metadata_value(paper: CandidatePaper, facet: str) -> Any:
    name = facet.lower()
    if name in {"paper_type", "publication_type"}:
        return paper.publication_type
    if name in {"paper_id", "canonical_paper_id"}:
        return paper.canonical_paper_id
    if name == "retracted":
        return paper.is_retracted
    return getattr(paper, name, None)


def _find_text(text: str, value: Any) -> tuple[int, int, str] | None:
    values = value if isinstance(value, (list, tuple, set)) else [value]
    for candidate in values:
        term = str(candidate or "").strip()
        if not term:
            continue
        match = re.search(re.escape(term), text, flags=re.IGNORECASE)
        if match:
            return match.start(), match.end(), text[match.start():match.end()]
    return None


_GENERIC_METHOD_VALUES = {
    "method",
    "methods",
    "technique",
    "techniques",
    "agricultural method",
    "agricultural methods",
    "agricultural technique",
    "agricultural techniques",
    "增产方法",
    "农业方法",
    "农业技术",
}
_GENERIC_METHOD_CUES = (
    "method",
    "approach",
    "technique",
    "strategy",
    "management",
    "optimization",
    "treatment",
    "intervention",
    "sowing",
    "planting",
    "irrigation",
    "fertilization",
    "nitrogen",
    "photosynthesis",
    "cultivation",
    "boosting",
    "through",
    "方法",
    "技术",
    "管理",
    "优化",
    "处理",
    "播种",
    "种植",
    "灌溉",
    "施肥",
    "栽培",
)


class DeterministicEvidenceJudge:
    provider = "deterministic-evidence-v1"
    prompt_version = "none"

    def judge_constraint(
        self,
        *,
        session_id: str,
        paper: CandidatePaper,
        constraint: Constraint,
    ) -> EvidenceRecord:
        facet = constraint.facet.lower()
        if facet in {
            "year",
            "language",
            "publication_type",
            "paper_type",
            "venue",
            "paper_id",
            "canonical_paper_id",
            "retracted",
        }:
            actual = _metadata_value(paper, facet)
            matched = _compare_metadata(actual, constraint)
            if matched is None:
                label = EvidenceLabel.UNKNOWN
            else:
                satisfied = matched
                if constraint.polarity == ConstraintPolarity.NEGATIVE:
                    satisfied = not matched
                label = EvidenceLabel.SUPPORT if satisfied else EvidenceLabel.VIOLATE
            return _metadata_record(
                session_id=session_id,
                paper=paper,
                constraint=constraint,
                label=label,
                actual=actual,
                reason="DIRECT_METADATA_CHECK",
            )

        abstract = paper.abstract or ""
        trusted_abstract = any(
            item.field == "abstract"
            and item.confidence_level in {"authoritative", "trusted"}
            for item in paper.field_provenance
        )
        if abstract and not trusted_abstract:
            return EvidenceRecord(
                evidence_id=_evidence_id(
                    session_id,
                    paper.canonical_paper_id,
                    constraint.id,
                    "weak-abstract",
                ),
                session_id=session_id,
                paper_id=paper.canonical_paper_id,
                paper_version_id=paper.preferred_version_id,
                facet_id=constraint.facet,
                constraint_id=constraint.id,
                label=EvidenceLabel.UNKNOWN,
                confidence=0.0,
                source_type="abstract",
                document_version_id=paper.preferred_version_id,
                content_hash=normalized_text_hash(abstract),
                normalized_text_hash=normalized_text_hash(abstract),
                parser_version="abstract-identity-v1",
                judge_model=self.provider,
                prompt_version=self.prompt_version,
                validation_status="untrusted_source",
                reason_codes=["WEAK_ABSTRACT_PROVENANCE"],
            )
        match = _find_text(abstract, constraint.value)
        generic_method_match = False
        normalized_value = str(constraint.value or "").strip().casefold()
        if (
            match is None
            and facet == "method"
            and normalized_value in _GENERIC_METHOD_VALUES
        ):
            match = _find_text(abstract, _GENERIC_METHOD_CUES)
            generic_method_match = match is not None
        counter_evidence = None
        if (
            match is None
            and constraint.polarity == ConstraintPolarity.NEGATIVE
            and constraint.facet.lower() == "simulation_only"
        ):
            counter_evidence = _find_text(
                abstract,
                ["real robot", "physical robot", "real-world experiment", "真实机器人"],
            )
        if match is None:
            if counter_evidence is not None:
                start, end, quote = counter_evidence
                label = EvidenceLabel.SUPPORT
                reasons = ["COUNTER_EVIDENCE_RULE:REAL_ROBOT_NOT_SIMULATION_ONLY"]
            else:
                label = EvidenceLabel.UNKNOWN
                start = end = None
                quote = None
                reasons = ["ABSTRACT_DOES_NOT_ESTABLISH_FACET"]
        else:
            start, end, quote = match
            label = (
                EvidenceLabel.VIOLATE
                if constraint.polarity == ConstraintPolarity.NEGATIVE
                else EvidenceLabel.SUPPORT
            )
            reasons = [
                "GENERIC_METHOD_CUE"
                if generic_method_match
                else "EXACT_ABSTRACT_SPAN"
            ]
        record = EvidenceRecord(
            evidence_id=_evidence_id(
                session_id,
                paper.canonical_paper_id,
                constraint.id,
                quote or "unknown",
            ),
            session_id=session_id,
            paper_id=paper.canonical_paper_id,
            paper_version_id=paper.preferred_version_id,
            facet_id=constraint.facet,
            constraint_id=constraint.id,
            label=label,
            confidence=(0.8 if generic_method_match else 0.95) if quote else 0.0,
            source_type="abstract",
            quoted_text=quote,
            document_version_id=paper.preferred_version_id,
            content_hash=normalized_text_hash(abstract),
            normalized_text_hash=normalized_text_hash(abstract),
            parser_version="abstract-identity-v1",
            char_start=start,
            char_end=end,
            judge_model=self.provider,
            prompt_version=self.prompt_version,
            validation_status="valid",
            reason_codes=reasons,
        )
        return validate_evidence_span(record, abstract)

    def judge_relation(
        self,
        *,
        session_id: str,
        paper: CandidatePaper,
        relation: Relation,
    ) -> EvidenceRecord:
        abstract = paper.abstract or ""
        trusted_abstract = any(
            item.field == "abstract"
            and item.confidence_level in {"authoritative", "trusted"}
            for item in paper.field_provenance
        )
        if abstract and not trusted_abstract:
            facet_id = f"relation:{relation.relation}:{relation.source}:{relation.target}"
            return EvidenceRecord(
                evidence_id=_evidence_id(
                    session_id,
                    paper.canonical_paper_id,
                    facet_id,
                    "weak-abstract",
                ),
                session_id=session_id,
                paper_id=paper.canonical_paper_id,
                paper_version_id=paper.preferred_version_id,
                facet_id=facet_id,
                label=EvidenceLabel.UNKNOWN,
                confidence=0.0,
                source_type="abstract",
                document_version_id=paper.preferred_version_id,
                content_hash=normalized_text_hash(abstract),
                normalized_text_hash=normalized_text_hash(abstract),
                parser_version="abstract-identity-v1",
                judge_model=self.provider,
                prompt_version=self.prompt_version,
                validation_status="untrusted_source",
                reason_codes=["WEAK_ABSTRACT_PROVENANCE"],
            )
        source_match = _find_text(abstract, relation.source)
        target_match = _find_text(abstract, relation.target)
        relation_cues = {
            "compare": ("compare", "comparison", "versus", " vs ", "比较", "对比"),
            "use": ("use", "using", "采用", "使用"),
            "evaluate_on": ("evaluate", "evaluation", "benchmark", "评估", "实验"),
            "extend": ("extend", "build on", "improve", "扩展", "改进"),
        }
        cue_match = _find_text(abstract, relation_cues.get(relation.relation, (relation.relation,)))
        if source_match and target_match and cue_match:
            start = min(source_match[0], target_match[0], cue_match[0])
            end = max(source_match[1], target_match[1], cue_match[1])
            quote = abstract[start:end]
            label = EvidenceLabel.SUPPORT
            reasons = ["DIRECT_RELATION_SPAN"]
        else:
            start = end = None
            quote = None
            label = EvidenceLabel.UNKNOWN
            reasons = ["ABSTRACT_DOES_NOT_ESTABLISH_RELATION"]
        facet_id = f"relation:{relation.relation}:{relation.source}:{relation.target}"
        record = EvidenceRecord(
            evidence_id=_evidence_id(session_id, paper.canonical_paper_id, facet_id, quote or "unknown"),
            session_id=session_id,
            paper_id=paper.canonical_paper_id,
            paper_version_id=paper.preferred_version_id,
            facet_id=facet_id,
            label=label,
            confidence=0.9 if quote else 0.0,
            source_type="abstract",
            quoted_text=quote,
            document_version_id=paper.preferred_version_id,
            content_hash=normalized_text_hash(abstract),
            normalized_text_hash=normalized_text_hash(abstract),
            parser_version="abstract-identity-v1",
            char_start=start,
            char_end=end,
            judge_model=self.provider,
            prompt_version=self.prompt_version,
            validation_status="valid",
            reason_codes=reasons,
        )
        return validate_evidence_span(record, abstract)

    def judge(
        self,
        *,
        session_id: str,
        paper: CandidatePaper,
        intent: ResearchIntent,
    ) -> list[EvidenceRecord]:
        records = [
            self.judge_constraint(session_id=session_id, paper=paper, constraint=constraint)
            for constraint in intent.constraints
        ]
        records.extend(
            self.judge_relation(session_id=session_id, paper=paper, relation=relation)
            for relation in intent.relations
            if relation.required
        )
        return records
