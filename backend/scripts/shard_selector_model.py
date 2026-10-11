"""Shard the single 15.2GB selector safetensors into 4 shards + index.json.

Why: transformers' meta-model load path (used whenever device_map is set, e.g.
4-bit loading) reliably crashes on THIS machine with a single huge shard
(Windows access violation in torch.storage.__getitem__ during
load_shard_file), while sharded layouts (crawler: 4 x ~4GB) load fine.

This is a pure byte-level copy: the safetensors header JSON is read once, the
tensor byte ranges are repartitioned across 4 shards, and each shard's raw
bytes are copied with seek/read/write -- no torch tensors are materialized, so
RAM stays flat.

Output: models/pasa-7b-selector-sharded/  (config/tokenizer copied too, so
AutoModelForCausalLM.from_pretrained can load from there directly).  The
original models/pasa-7b-selector/ is left untouched.

Run: python -m backend.scripts.shard_selector_model
"""
from __future__ import annotations

import json
import shutil
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "models" / "pasa-7b-selector" / "model.safetensors"
OUT = ROOT / "models" / "pasa-7b-selector-sharded"
SHARD_COUNT = 4
COPY_EXT = {".json", ".txt", ".md"}


def _safe_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="ascii", errors="replace")
    except Exception:  # noqa: BLE001
        pass


def _read_header(path: Path):
    """Read the safetensors header JSON (8-byte length prefix + JSON)."""
    with path.open("rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_len))
    return header, 8 + header_len


def main() -> None:
    _safe_stdout()
    if not SRC.is_file():
        raise SystemExit(f"missing {SRC}")
    header, data_offset = _read_header(SRC)
    tensors = header.get("__metadata__") or {}
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    total_bytes = (SRC.stat().st_size - data_offset)
    print(f"tensors: {len(tensors)}, data bytes: {total_bytes / 1e9:.1f}GB", flush=True)

    # Greedy partition into SHARD_COUNT balanced buckets (by byte size).
    buckets: list[list[str]] = [[] for _ in range(SHARD_COUNT)]
    bucket_bytes = [0] * SHARD_COUNT
    for name in sorted(tensors, key=lambda n: -(tensors[n]["data_offsets"][1] - tensors[n]["data_offsets"][0])):
        size = tensors[name]["data_offsets"][1] - tensors[name]["data_offsets"][0]
        target = min(range(SHARD_COUNT), key=lambda i: bucket_bytes[i])
        buckets[target].append(name)
        bucket_bytes[target] += size

    OUT.mkdir(parents=True, exist_ok=True)
    src_handle = SRC.open("rb")

    weight_map: dict[str, str] = {}
    for shard_index, names in enumerate(buckets, start=1):
        shard_path = OUT / f"model-{shard_index:05d}-of-{SHARD_COUNT:05d}.safetensors"
        # Rebase each tensor's data offsets to this shard's local data area
        # (relative to 8 + header_len, which is where the data begins).
        local_headers: dict = {}
        local_offset = 0
        for name in names:
            meta = dict(tensors[name])
            start, end = meta["data_offsets"]
            meta["data_offsets"] = [local_offset, local_offset + (end - start)]
            local_headers[name] = meta
            local_offset += end - start
            weight_map[name] = shard_path.name
        header_json = json.dumps(local_headers, separators=(",", ":")).encode()
        with shard_path.open("wb") as out_handle:
            out_handle.write(struct.pack("<Q", len(header_json)))
            out_handle.write(header_json)
            for name in names:
                start, end = tensors[name]["data_offsets"]
                src_handle.seek(data_offset + start)
                out_handle.write(src_handle.read(end - start))
        print(
            f"  {shard_path.name}: {len(names)} tensors, {local_offset / 1e9:.1f}GB",
            flush=True,
        )

    src_handle.close()
    index = {
        "metadata": {"total_size": SRC.stat().st_size},
        "weight_map": weight_map,
    }
    (OUT / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )
    for source in (ROOT / "models" / "pasa-7b-selector").iterdir():
        if source.suffix in COPY_EXT and source.name != "README.md":
            shutil.copy2(source, OUT / source.name)
    print(f"done -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
