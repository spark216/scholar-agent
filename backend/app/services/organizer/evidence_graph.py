"""Build evidence-backed graph edges; unknown never becomes a fact edge."""

from __future__ import annotations

from backend.app.models.evidrank import EvidenceLabel, FinalRankingBundle
from backend.app.models.scimap import EvidenceGraph, EvidenceGraphEdge, EvidenceGraphNode


def build_evidence_graph(bundle: FinalRankingBundle) -> EvidenceGraph:
    visible_ids = set(bundle.high + bundle.partial + bundle.verify)
    item_by_id = {item.paper_id: item for item in bundle.items}
    effective_labels = {
        (item.paper_id, facet_id): label
        for item in bundle.items
        for facet_id, label in item.facet_labels.items()
    }
    nodes: dict[str, EvidenceGraphNode] = {}
    grouped: dict[tuple[str, str, EvidenceLabel], list] = {}

    for paper_id in sorted(visible_ids):
        item = item_by_id[paper_id]
        nodes[f"paper:{paper_id}"] = EvidenceGraphNode(
            node_id=f"paper:{paper_id}",
            node_type="paper",
            label=item.title,
            paper_id=paper_id,
        )

    for record in bundle.evidence_matrix:
        if (
            record.paper_id not in visible_ids
            or record.validation_status != "valid"
            or record.label == EvidenceLabel.UNKNOWN
            or effective_labels.get((record.paper_id, record.facet_id)) != record.label
        ):
            continue
        facet_node = f"facet:{record.facet_id}"
        nodes.setdefault(
            facet_node,
            EvidenceGraphNode(
                node_id=facet_node,
                node_type="facet",
                label=record.facet_id,
            ),
        )
        grouped.setdefault((record.paper_id, record.facet_id, record.label), []).append(record)

    edges: list[EvidenceGraphEdge] = []
    for (paper_id, facet_id, label), records in sorted(
        grouped.items(),
        key=lambda value: (value[0][0], value[0][1], value[0][2].value),
    ):
        edges.append(
            EvidenceGraphEdge(
                source=f"paper:{paper_id}",
                target=f"facet:{facet_id}",
                relation="supports" if label == EvidenceLabel.SUPPORT else "violates",
                evidence_ids=sorted(record.evidence_id for record in records),
                source_types=sorted({record.source_type for record in records}),
                confidence=max(record.confidence for record in records),
                is_inferred=False,
            )
        )
    return EvidenceGraph(
        nodes=sorted(nodes.values(), key=lambda value: value.node_id),
        edges=edges,
    )
