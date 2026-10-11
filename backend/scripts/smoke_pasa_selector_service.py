"""Smoke test for the PasaSelector service wrapper (4-bit load + coexistence
with the BGE encoder and reranker on the same GPU).

Run with the selector enabled:
    PASA_SELECTOR_ENABLED=1 python -m backend.scripts.smoke_pasa_selector_service
"""
from __future__ import annotations

import sys
import time

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="ascii", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    import torch
    from backend.app.services.embeddings import get_bge_m3_encoder
    from backend.app.services.pasa_selector import get_pasa_selector
    from backend.app.services.reranker import get_bge_reranker

    # Load the 7B selector FIRST (its load-time RAM spike must happen while
    # memory is free — it segfaults after the BGE models on 16GB machines).
    selector = get_pasa_selector()
    encoder = get_bge_m3_encoder()
    reranker = get_bge_reranker()
    print(
        f"encoder={encoder.provider} reranker={reranker.provider} "
        f"selector={selector.provider}",
        flush=True,
    )
    if not selector.enabled or selector._model is None:
        print("SKIP: selector disabled/unavailable — cannot smoke test", flush=True)
        return 1

    query = (
        "Give me papers which shows that using a smaller dataset in large language "
        "model pre-training can result in better models than using bigger datasets."
    )
    papers = [
        {
            "title": "Specialized Language Models with Cheap Inference from Limited Domain Data",
            "abstract": (
                "Large language models have emerged as a versatile tool but are challenging "
                "to apply to tasks lacking large inference budgets and large in-domain "
                "training sets. This work compares approaches across the pretraining budget, "
                "specialization budget, inference budget, and in-domain training set size, "
                "showing small models trained on importance sampled datasets are attractive."
            ),
        },
        {
            "title": "A Survey of Solar Panel Efficiency in Arid Climates",
            "abstract": (
                "We survey photovoltaic panel efficiency across arid regions, comparing "
                "silicon and perovskite technologies under high irradiance and dust "
                "accumulation conditions."
            ),
        },
    ]
    t0 = time.perf_counter()
    scores, meta = selector.infer_scores(query, papers)
    print(f"infer in {time.perf_counter() - t0:.1f}s: {scores} meta={meta}", flush=True)
    assert scores is not None and len(scores) == 2
    assert scores[0] > 0.5, f"gold paper should score >0.5, got {scores[0]}"
    assert scores[1] < 0.5, f"irrelevant paper should score <0.5, got {scores[1]}"

    peak = torch.cuda.max_memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"peak VRAM while all three models resident: {peak:.2f}GB / {total:.2f}GB", flush=True)
    assert peak < total * 0.95, "VRAM too close to the limit — reduce selector batch"
    print("SMOKE OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
