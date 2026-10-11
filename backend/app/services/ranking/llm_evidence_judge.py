"""Optional strict-JSON LLM upgrade for unresolved abstract evidence."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from backend.app.config import get_settings
from backend.app.models.evidrank import (
    CandidatePaper,
    EvidenceLabel,
    EvidenceRecord,
    ResearchIntent,
)
from backend.app.services.ranking.evidence_judge import DeterministicEvidenceJudge
from backend.app.services.ranking.evidence_span_validator import (
    normalized_text_hash,
    validate_evidence_span,
)


def _extract_json(text: str) -> dict[str, Any]:
    value = (text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", value)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("evidence judge must return a JSON object")
    return payload


class ZhipuEvidenceJudge:
    """Upgrade deterministic ``unknown`` records; never assign ranking levels."""

    def __init__(self) -> None:
        settings = get_settings()
        self.enabled = bool(settings.evidrank_evidence_llm_enabled)
        self.api_key = settings.zhipuai_api_key
        self.model = settings.evidrank_evidence_llm_model
        self.prompt_version = settings.evidrank_evidence_prompt_version
        self.max_tokens = settings.evidrank_evidence_max_tokens
        self.batch_size = max(
            1,
            int(getattr(settings, "evidrank_evidence_batch_size", 4)),
        )
        self.max_retries = max(
            0,
            int(getattr(settings, "evidrank_evidence_max_retries", 1)),
        )
        self.timeout_seconds = max(
            1.0,
            float(getattr(settings, "evidrank_evidence_timeout_seconds", 20.0)),
        )
        self.base = DeterministicEvidenceJudge()
        self.provider = (
            f"zhipu:{self.model}:{self.prompt_version}"
            if self.enabled and self.api_key
            else self.base.provider
        )
        self.last_meta: dict[str, Any] = {
            "api_calls": 0,
            "token_usage": 0,
            "enabled": self.enabled,
            "configured": bool(self.api_key),
        }
        self.reset_metrics()

    def reset_metrics(self) -> None:
        self._metrics = {
            "api_calls": 0,
            "token_usage": 0,
            "errors": 0,
            "retries": 0,
            "schema_errors": 0,
            "partial_responses": 0,
            "papers_submitted": 0,
        }

    def get_metrics(self) -> dict[str, int]:
        return dict(self._metrics)

    def _prompt(
        self,
        paper: CandidatePaper,
        unresolved: list[EvidenceRecord],
        intent: ResearchIntent,
    ) -> tuple[str, str]:
        constraints = []
        by_id = {constraint.id: constraint for constraint in intent.constraints}
        for record in unresolved:
            constraint = by_id.get(record.constraint_id or "")
            if constraint is not None:
                constraints.append(
                    {
                        "evidence_key": constraint.id,
                        "facet_id": constraint.facet,
                        "operator": constraint.operator,
                        "value": constraint.value,
                        "polarity": constraint.polarity.value,
                        "hardness": constraint.hardness.value,
                    }
                )
            else:
                constraints.append(
                    {
                        "evidence_key": record.facet_id,
                        "facet_id": record.facet_id,
                        "operator": "establish_relation",
                        "value": record.facet_id,
                        "polarity": "positive",
                        "hardness": "required_relation",
                    }
                )
        system = """You are a constrained scientific evidence extractor.
For every supplied constraint, decide only support, violate, or unknown from the
provided abstract. Absence of a statement is unknown, not violate. Return an
exact contiguous quote copied from the abstract for support/violate. Do not
assign paper relevance or invent facts. Return strict JSON:
{"records":[{"evidence_key":"...","label":"support|violate|unknown","quoted_text":null|"exact quote"}]}"""
        user = json.dumps(
            {
                "query": intent.raw_query,
                "paper_id": paper.canonical_paper_id,
                "title": paper.title,
                "abstract": paper.abstract or "",
                "constraints": constraints,
            },
            ensure_ascii=False,
        )
        return system, user

    def _batch_prompt(
        self,
        papers: list[CandidatePaper],
        unresolved_by_paper: dict[str, list[EvidenceRecord]],
        intent: ResearchIntent,
    ) -> tuple[str, str]:
        paper_payloads: list[dict[str, Any]] = []
        by_id = {constraint.id: constraint for constraint in intent.constraints}
        for paper in papers:
            unresolved = unresolved_by_paper[paper.canonical_paper_id]
            constraints: list[dict[str, Any]] = []
            for record in unresolved:
                constraint = by_id.get(record.constraint_id or "")
                if constraint is not None:
                    constraints.append(
                        {
                            "evidence_key": constraint.id,
                            "facet_id": constraint.facet,
                            "operator": constraint.operator,
                            "value": constraint.value,
                            "polarity": constraint.polarity.value,
                            "hardness": constraint.hardness.value,
                        }
                    )
                else:
                    constraints.append(
                        {
                            "evidence_key": record.facet_id,
                            "facet_id": record.facet_id,
                            "operator": "establish_relation",
                            "value": record.facet_id,
                            "polarity": "positive",
                            "hardness": "required_relation",
                        }
                    )
            paper_payloads.append(
                {
                    "paper_id": paper.canonical_paper_id,
                    "title": paper.title,
                    "abstract": paper.abstract or "",
                    "constraints": constraints,
                }
            )

        system = """You are a constrained scientific evidence extractor.
For every paper and every supplied constraint, decide only support, violate, or
unknown from that paper's abstract. Absence is unknown, not violate. For
support/violate, return an exact contiguous quote copied from the same abstract.
Do not assign paper relevance, omit records, duplicate IDs, or invent facts.
Return strict JSON only:
{"results":[{"paper_id":"...","records":[{"evidence_key":"...","label":"support|violate|unknown","quoted_text":null|"exact quote"}]}]}"""
        user = json.dumps(
            {
                "query": intent.raw_query,
                "papers": paper_payloads,
            },
            ensure_ascii=False,
        )
        return system, user

    def _request(self, system: str, user: str) -> tuple[dict[str, Any], int]:
        from zhipuai import ZhipuAI

        try:
            client = ZhipuAI(
                api_key=self.api_key,
                timeout=self.timeout_seconds,
                max_retries=0,
            )
        except TypeError:
            client = ZhipuAI(api_key=self.api_key)
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        try:
            response = client.chat.completions.create(
                **request,
                response_format={"type": "json_object"},
            )
        except TypeError:
            response = client.chat.completions.create(**request)
        payload = _extract_json(response.choices[0].message.content or "")
        usage = getattr(response, "usage", None)
        return payload, int(getattr(usage, "total_tokens", 0) or 0)

    @staticmethod
    def _normalize_label(value: Any) -> EvidenceLabel:
        aliases = {
            "支持": "support",
            "满足": "support",
            "违反": "violate",
            "不满足": "violate",
            "未知": "unknown",
            "不确定": "unknown",
        }
        normalized = aliases.get(str(value or "").strip(), str(value or "unknown").strip().lower())
        return EvidenceLabel(normalized)

    @staticmethod
    def _validate_batch_payload(
        payload: dict[str, Any],
        expected_keys: dict[str, set[str]],
        *,
        require_complete: bool = True,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        raw_results = payload.get("results")
        if isinstance(raw_results, dict):
            raw_results = [
                {"paper_id": paper_id, "records": records}
                for paper_id, records in raw_results.items()
            ]
        if (
            not isinstance(raw_results, list)
            and isinstance(payload.get("records"), list)
            and len(expected_keys) == 1
        ):
            raw_results = [
                {
                    "paper_id": next(iter(expected_keys)),
                    "records": payload["records"],
                }
            ]
        if not isinstance(raw_results, list):
            raise ValueError("evidence batch result must contain a results list")
        parsed: dict[str, dict[str, dict[str, Any]]] = {}
        for raw_result in raw_results:
            if not isinstance(raw_result, dict):
                if not require_complete:
                    continue
                raise ValueError("every evidence batch result must be an object")
            paper_id = str(raw_result.get("paper_id") or "")
            if paper_id not in expected_keys:
                if not require_complete:
                    continue
                raise ValueError(f"unexpected evidence paper_id: {paper_id or '<empty>'}")
            if paper_id in parsed:
                if not require_complete:
                    continue
                raise ValueError(f"duplicate evidence paper_id: {paper_id}")
            raw_records = raw_result.get("records")
            if not isinstance(raw_records, list):
                if not require_complete:
                    continue
                raise ValueError(f"records must be a list for paper_id={paper_id}")
            by_key: dict[str, dict[str, Any]] = {}
            for raw_record in raw_records:
                if not isinstance(raw_record, dict):
                    if not require_complete:
                        continue
                    raise ValueError(f"invalid evidence record for paper_id={paper_id}")
                evidence_key = str(
                    raw_record.get("evidence_key")
                    or raw_record.get("constraint_id")
                    or raw_record.get("facet_id")
                    or ""
                )
                if evidence_key not in expected_keys[paper_id]:
                    if not require_complete:
                        continue
                    raise ValueError(
                        f"unexpected evidence_key={evidence_key or '<empty>'} for paper_id={paper_id}"
                    )
                if evidence_key in by_key:
                    if not require_complete:
                        continue
                    raise ValueError(
                        f"duplicate evidence_key={evidence_key} for paper_id={paper_id}"
                    )
                try:
                    label = ZhipuEvidenceJudge._normalize_label(raw_record.get("label"))
                except ValueError as exc:
                    if not require_complete:
                        continue
                    raise ValueError(
                        f"invalid evidence label for paper_id={paper_id}, key={evidence_key}"
                    ) from exc
                by_key[evidence_key] = {**raw_record, "label": label.value}
            missing = expected_keys[paper_id] - set(by_key)
            if missing and require_complete:
                raise ValueError(
                    f"missing evidence keys for paper_id={paper_id}: {sorted(missing)}"
                )
            if by_key:
                parsed[paper_id] = by_key
        missing_papers = set(expected_keys) - set(parsed)
        if missing_papers and require_complete:
            raise ValueError(f"missing evidence paper_ids: {sorted(missing_papers)}")
        return parsed

    @staticmethod
    def _upgrade(
        record: EvidenceRecord,
        result: dict[str, Any],
        abstract: str,
        *,
        model: str,
        prompt_version: str,
    ) -> EvidenceRecord:
        try:
            label = ZhipuEvidenceJudge._normalize_label(result.get("label"))
        except ValueError:
            label = EvidenceLabel.UNKNOWN
        quote = result.get("quoted_text")
        if label == EvidenceLabel.UNKNOWN or not isinstance(quote, str) or not quote:
            return record.model_copy(
                update={
                    "judge_model": model,
                    "prompt_version": prompt_version,
                    "reason_codes": list(dict.fromkeys(record.reason_codes + ["LLM_RETURNED_UNKNOWN"])),
                }
            )
        start = abstract.find(quote)
        if start < 0:
            candidate = record.model_copy(
                update={
                    "label": label,
                    "quoted_text": quote,
                    "char_start": None,
                    "char_end": None,
                    "judge_model": model,
                    "prompt_version": prompt_version,
                    "validation_status": "invalid",
                    "reason_codes": list(dict.fromkeys(record.reason_codes + ["LLM_SPAN_NOT_FOUND"])),
                }
            )
            return validate_evidence_span(candidate, abstract)
        end = start + len(quote)
        evidence_id = "ev-" + hashlib.sha256(
            (
                f"{record.session_id}\x1f{record.paper_id}\x1f"
                f"{record.constraint_id or record.facet_id}\x1f{quote}"
            ).encode("utf-8")
        ).hexdigest()[:20]
        candidate = record.model_copy(
            update={
                "evidence_id": evidence_id,
                "label": label,
                "confidence": 1.0,
                "quoted_text": quote,
                "char_start": start,
                "char_end": end,
                "content_hash": normalized_text_hash(abstract),
                "normalized_text_hash": normalized_text_hash(abstract),
                "judge_model": model,
                "prompt_version": prompt_version,
                "validation_status": "valid",
                "reason_codes": ["LLM_EXACT_ABSTRACT_SPAN"],
            }
        )
        return validate_evidence_span(candidate, abstract)

    def judge(
        self,
        *,
        session_id: str,
        paper: CandidatePaper,
        intent: ResearchIntent,
    ) -> list[EvidenceRecord]:
        return self.judge_batch(
            session_id=session_id,
            papers=[paper],
            intent=intent,
        )[paper.canonical_paper_id]

    def judge_batch(
        self,
        *,
        session_id: str,
        papers: list[CandidatePaper],
        intent: ResearchIntent,
    ) -> dict[str, list[EvidenceRecord]]:
        records_by_paper = {
            paper.canonical_paper_id: self.base.judge(
                session_id=session_id,
                paper=paper,
                intent=intent,
            )
            for paper in papers
        }
        unresolved_by_paper = {
            paper_id: [
                record
                for record in records
                if record.source_type == "abstract"
                and record.label == EvidenceLabel.UNKNOWN
                and record.validation_status != "untrusted_source"
            ]
            for paper_id, records in records_by_paper.items()
        }
        pending = [
            paper
            for paper in papers
            if unresolved_by_paper[paper.canonical_paper_id]
        ]
        if not pending or not self.enabled or not self.api_key:
            return records_by_paper

        expected_keys = {
            paper.canonical_paper_id: {
                record.constraint_id or record.facet_id
                for record in unresolved_by_paper[paper.canonical_paper_id]
            }
            for paper in pending
        }
        system, user = self._batch_prompt(pending, unresolved_by_paper, intent)
        self._metrics["papers_submitted"] += len(pending)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                self._metrics["retries"] += 1
            self._metrics["api_calls"] += 1
            try:
                payload, token_usage = self._request(system, user)
                partial_response = False
                try:
                    parsed = self._validate_batch_payload(payload, expected_keys)
                except ValueError as strict_error:
                    self._metrics["schema_errors"] += 1
                    parsed = self._validate_batch_payload(
                        payload,
                        expected_keys,
                        require_complete=False,
                    )
                    if not parsed:
                        raise strict_error
                    partial_response = True
                    self._metrics["errors"] += 1
                    self._metrics["partial_responses"] += 1
                self._metrics["token_usage"] += token_usage
                self.last_meta = {
                    "api_calls": attempt + 1,
                    "token_usage": token_usage,
                    "enabled": True,
                    "configured": True,
                    "paper_count": len(pending),
                    "batch_size": self.batch_size,
                    "partial_response": partial_response,
                }
                pending_by_id = {
                    paper.canonical_paper_id: paper
                    for paper in pending
                }
                for paper_id, by_key in parsed.items():
                    abstract = pending_by_id[paper_id].abstract or ""
                    records_by_paper[paper_id] = [
                        self._upgrade(
                            record,
                            by_key[record.constraint_id or record.facet_id],
                            abstract,
                            model=f"zhipu:{self.model}",
                            prompt_version=self.prompt_version,
                        )
                        if (record.constraint_id or record.facet_id) in by_key
                        else record
                        for record in records_by_paper[paper_id]
                    ]
                if partial_response:
                    for paper in pending:
                        paper_id = paper.canonical_paper_id
                        returned_keys = set(parsed.get(paper_id, {}))
                        records_by_paper[paper_id] = [
                            record.model_copy(
                                update={
                                    "reason_codes": list(
                                        dict.fromkeys(
                                            record.reason_codes
                                            + ["EVIDENCE_LLM_PARTIAL_RESPONSE"]
                                        )
                                    )
                                }
                            )
                            if (
                                record in unresolved_by_paper[paper_id]
                                and (record.constraint_id or record.facet_id)
                                not in returned_keys
                            )
                            else record
                            for record in records_by_paper[paper_id]
                        ]
                return records_by_paper
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self._metrics["errors"] += 1
                if isinstance(exc, ValueError):
                    self._metrics["schema_errors"] += 1

        self.last_meta = {
            "api_calls": self.max_retries + 1,
            "token_usage": 0,
            "enabled": True,
            "configured": True,
            "paper_count": len(pending),
            "batch_size": self.batch_size,
            "error": str(last_error or "unknown evidence batch failure"),
        }
        for paper in pending:
            paper_id = paper.canonical_paper_id
            unresolved = unresolved_by_paper[paper_id]
            records_by_paper[paper_id] = [
                record.model_copy(
                    update={
                        "reason_codes": list(
                            dict.fromkeys(record.reason_codes + ["EVIDENCE_LLM_FAILED"])
                        )
                    }
                )
                if record in unresolved
                else record
                for record in records_by_paper[paper_id]
            ]
        return records_by_paper


def get_evidence_judge() -> ZhipuEvidenceJudge:
    return ZhipuEvidenceJudge()
