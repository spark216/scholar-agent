"""Download and validate the exact model snapshots used by EvidRank.

Run from the repository root:

    python -m backend.scripts.download_evidrank_models

The model directory is gitignored.  A completion manifest is written only
after a real weight file is present, so a Git-LFS pointer or interrupted
download cannot be mistaken for a usable model.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = {
    "bge-m3": {
        "repo_id": "BAAI/bge-m3",
        "revision": "5617a9f61b028005a4858fdac845db406aefb181",
    },
    "bge-reranker-v2-m3": {
        "repo_id": "BAAI/bge-reranker-v2-m3",
        "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    },
}
MIN_WEIGHT_BYTES = 100 * 1024 * 1024


def _validate_snapshot(path: Path) -> list[dict[str, int | str]]:
    weights = [
        candidate
        for pattern in ("*.safetensors", "pytorch_model*.bin", "*.pt")
        for candidate in path.rglob(pattern)
        if candidate.is_file() and candidate.stat().st_size >= MIN_WEIGHT_BYTES
    ]
    if not weights:
        raise RuntimeError(
            f"{path} has no weight file larger than {MIN_WEIGHT_BYTES} bytes; "
            "the download is incomplete or contains only Git-LFS pointers"
        )
    return [
        {
            "file": str(value.relative_to(path)),
            "bytes": value.stat().st_size,
        }
        for value in sorted(weights)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["all", *MODELS],
        default="all",
    )
    parser.add_argument("--target-root", type=Path, default=Path.cwd() / "models")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    if args.max_workers <= 0:
        raise SystemExit("--max-workers must be positive")

    selected = MODELS if args.model == "all" else {args.model: MODELS[args.model]}
    for name, spec in selected.items():
        target = (args.target_root / f"{name}-{spec['revision'][:8]}").resolve()
        resolved = Path(
            snapshot_download(
                repo_id=spec["repo_id"],
                revision=spec["revision"],
                local_dir=target,
                max_workers=args.max_workers,
            )
        ).resolve()
        weights = _validate_snapshot(resolved)
        manifest = {
            "model": name,
            "repo_id": spec["repo_id"],
            "revision": spec["revision"],
            "resolved_path": str(resolved),
            "weight_files": weights,
            "completed_at": datetime.now(UTC).isoformat(),
        }
        (resolved / "evidrank-model-manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
