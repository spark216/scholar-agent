"""Compare the selector A/B runs (OFF baseline vs ON fused) with the offline
per-model hit rates and the historical best run.

Produces:
  - .modeling-audit/SELECTOR_INTEGRATION_REPORT.md  (human-readable)
  - .modeling-audit/selector_ab_detail.json         (full per-query data)

Usage:
    python -m backend.scripts.compare_selector_ab --off <run.json> --on <run.json>
(omit --on for an OFF-only report; historical + offline data are auto-loaded)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL = (
    ROOT / "backend" / "paper-test" / "results" / "runs"
    / "20260822T045637385248Z_f3557136432d442f8217b6d6596f88f0.json"
)
OFFLINE = ROOT / ".modeling-audit" / "selector_rerank_results.json"
DETAIL_OUT = ROOT / ".modeling-audit" / "selector_ab_detail.json"
REPORT_OUT = ROOT / ".modeling-audit" / "SELECTOR_INTEGRATION_REPORT.md"


def _safe_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="ascii", errors="replace")
    except Exception:  # noqa: BLE001
        pass


def _load(path: Path):
    if path and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _per_query_metrics(run):
    if not run:
        return {}
    out = {}
    for q in run.get("queries", []):
        out[q["qid"]] = {
            "metrics": q.get("metrics", {}),
            "latency_ms": q.get("latency_ms"),
            "api_calls": q.get("api_calls"),
            "token_usage": q.get("token_usage"),
            "matched": q.get("matched_titles", []),
            "returned_count": q.get("returned_count"),
            "relevant_count": q.get("relevant_count"),
            "candidate_trace": q.get("candidate_trace"),
        }
    return out


def _fmt(value, missing="—"):
    return missing if value is None else f"{value:.3f}"


def _agg_table(metric_key, runs, ks=(5, 10, 20)):
    rows = []
    for label, run in runs:
        agg = (run or {}).get("aggregate", {})
        if not agg:
            continue
        cells = [
            _fmt(agg.get(str(k), {}).get(metric_key)) if agg.get(str(k)) else "—"
            for k in ks
        ]
        rows.append(f"| {label} | {' | '.join(cells)} |")
    return rows


def _hits_table(runs, ks=(5, 10, 20)):
    rows = []
    for label, run in runs:
        agg = (run or {}).get("aggregate", {})
        if not agg:
            continue
        cells = [str(agg[str(k)]["hits"]) if agg.get(str(k)) else "—" for k in ks]
        rows.append(f"| {label} | {' | '.join(cells)} |")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--off", type=str, default="")
    parser.add_argument("--on", type=str, default="")
    args = parser.parse_args()
    _safe_stdout()

    off = _load(Path(args.off)) if args.off else None
    on = _load(Path(args.on)) if args.on else None
    hist = _load(HISTORICAL)
    offline = _load(OFFLINE)

    off_q = _per_query_metrics(off)
    on_q = _per_query_metrics(on)
    hist_q = _per_query_metrics(hist)

    ks = [5, 10, 20]
    detail = {
        "runs": {
            "off": args.off or None,
            "on": args.on or None,
            "historical": str(HISTORICAL),
            "offline": str(OFFLINE),
        },
        "aggregate": {},
        "per_query": {},
    }
    for label, run in (("off", off), ("on", on), ("historical", hist)):
        detail["aggregate"][label] = (run or {}).get("aggregate", {})

    lines = ["# Selector 融入主程序：A/B 评测报告", ""]
    lines.append(f"生成时间：2026-08-24  |  pasa-real 5 条基准（seed 42，k 5/10/20）")
    lines.append("")

    # ---- hit-rate table -------------------------------------------------
    lines.append("## 1. 命中率（hits，两模型对比）")
    lines.append("")
    lines.append("| 配置 @k | @5 | @10 | @20 |")
    lines.append("|---|---|---|---|")
    runs = [
        ("历史 A+B 全管道 (f3557136)", hist),
        ("OFF 基线（当前 reranker 管道）", off),
    ]
    if on is not None:
        runs.append(("ON 融合（+selector）", on))
    lines.extend(_hits_table(runs, ks))
    lines.append("")

    # offline per-model standalone hit rates
    if offline:
        lines.append("### 离线同池单模型命中率（参照）")
        lines.append("")
        lines.append("| 模型 @k | @5 | @10 | @20 |")
        lines.append("|---|---|---|---|")
        for name in ("selector", "reranker", "rrf_fusion"):
            agg = (offline.get(name) or {}).get("agg", {})
            if not agg:
                continue
            cells = [str(agg.get(str(k), {}).get("hits", 0)) for k in ks]
            lines.append(f"| {name} | {' | '.join(cells)} |")
        lines.append("")

    # ---- metric tables -------------------------------------------------
    for index, metric in enumerate(("precision", "recall", "f1", "ndcg", "mrr"), start=2):
        lines.append(f"## {index}. {metric.upper()} @5/10/20")
        lines.append("")
        lines.append("| 配置 @k | @5 | @10 | @20 |")
        lines.append("|---|---|---|---|")
        lines.extend(_agg_table(metric, runs, ks))
        lines.append("")

    # ---- cost -----------------------------------------------------------
    lines.append("## 7. 成本（token / API 调用 / 延迟）")
    lines.append("")
    lines.append("| 配置 | token 总量 | API 调用 | 总延迟 (s) |")
    lines.append("|---|---|---|---|")
    for label, run in runs:
        if not run:
            continue
        lines.append(
            f"| {label} | {run.get('total_token_usage', 0)} | "
            f"{run.get('total_api_calls', 0)} | {run.get('total_latency_ms', 0) / 1000:.0f} |"
        )
    lines.append("")

    # ---- per-query detail ----------------------------------------------
    lines.append("## 8. 逐查询命中与返回明细")
    lines.append("")
    all_qids = sorted(set(off_q) | set(on_q) | set(hist_q))
    per_query_detail = {}
    for qid in all_qids:
        lines.append(f"### {qid}")
        lines.append("")
        lines.append("| 查询/配置 | hits@5 | hits@10 | hits@20 | R@20 | MRR@5 | 返回数 | 匹配 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        row = {}
        for label, qmap in (("历史", hist_q), ("OFF", off_q), ("ON", on_q)):
            q = qmap.get(qid)
            if not q:
                continue
            m = q["metrics"]
            hits5 = m.get("5", {}).get("hits", 0)
            hits10 = m.get("10", {}).get("hits", 0)
            hits20 = m.get("20", {}).get("hits") if m.get("20") else "—"
            r20 = m.get("20", {}).get("recall", 0) if m.get("20") else "—"
            mrr5 = m.get("5", {}).get("mrr", 0)
            lines.append(
                f"| {label} | {hits5} | {hits10} | {hits20} | {r20} | {mrr5:.3f} "
                f"| {q['returned_count']} | {len(q['matched'])} |"
            )
            row[label] = q
        lines.append("")
        # returned papers from candidate_trace (ON run preferred)
        trace_src = (on_q.get(qid) or off_q.get(qid) or {}).get("candidate_trace")
        if trace_src:
            ranking = trace_src.get("ranking") or []
            lines.append("返回论文（ranking 阶段，rank/title/level/sel_score）:")
            lines.append("")
            lines.append("| rank | level | sel_score | title |")
            lines.append("|---|---|---|---|")
            for r in ranking[:20]:
                ss = r.get("sel_score")
                ss_txt = f"{ss:.3f}" if isinstance(ss, (int, float)) else "-"
                lines.append(f"| {r.get('rank')} | {r.get('level')} | {ss_txt} | {str(r.get('title'))[:80]} |")
            lines.append("")
        per_query_detail[qid] = row

    # costs per query
    lines.append("## 9. 逐查询成本")
    lines.append("")
    lines.append("| qid | 配置 | token | API | 延迟 (s) |")
    lines.append("|---|---|---|---|---|")
    for qid in all_qids:
        for label, qmap in (("OFF", off_q), ("ON", on_q)):
            q = qmap.get(qid)
            if not q:
                continue
            lines.append(
                f"| {qid} | {label} | {q['token_usage']} | {q['api_calls']} "
                f"| {q['latency_ms'] / 1000:.0f} |"
            )
    lines.append("")

    # ---- conclusion -----------------------------------------------------
    lines.append("## 10. 结论")
    lines.append("")
    if on is not None and off is not None:
        o5 = off["aggregate"].get("5", {}).get("hits", 0)
        n5 = on["aggregate"].get("5", {}).get("hits", 0)
        o20 = off["aggregate"].get("20", {}).get("hits", 0)
        n20 = on["aggregate"].get("20", {}).get("hits", 0)
        lines.append(
            f"- 命中率：@5 {o5} → {n5}（{'提升' if n5 > o5 else '持平/下降'}），"
            f"@20 {o20} → {n20}（{'提升' if n20 > o20 else '持平/下降'}）"
        )
        f1_off = off["aggregate"].get("5", {}).get("f1", 0)
        f1_on = on["aggregate"].get("5", {}).get("f1", 0)
        mrr_off = off["aggregate"].get("5", {}).get("mrr", 0)
        mrr_on = on["aggregate"].get("5", {}).get("mrr", 0)
        lines.append(f"- F1@5 {f1_off:.3f} → {f1_on:.3f}；MRR@5 {mrr_off:.3f} → {mrr_on:.3f}")
        lines.append("")
        lines.append("> 注意：本报告的 ON 为融合模式（selector 权重 0.30，头部保护开，打分 top-80）。")
        lines.append("> 两个模型的取舍请结合『离线同池单模型命中率』与生产延迟预算决策。")
    else:
        lines.append("（仅 OFF 基线；ON 运行后重跑本脚本补全对比。）")

    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")
    detail["per_query"] = per_query_detail
    DETAIL_OUT.write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report -> {REPORT_OUT}", flush=True)
    print(f"detail -> {DETAIL_OUT}", flush=True)


if __name__ == "__main__":
    main()
