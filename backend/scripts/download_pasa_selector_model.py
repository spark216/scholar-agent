"""Robust resumable downloader for bytedance-research/pasa-7b-selector model.safetensors.

Why not snapshot_download: hf-mirror's HEAD/metadata check fails with
"FileMetadataError: Distant resource does not seem to be on huggingface.co"
inside this network, but plain Range GETs work fine (verified: HTTP 206).

Strategy: stream model.safetensors (15.2 GB) with Range resume.  Any
interruption can be restarted cheaply; the script appends to the existing
partial file.  Final size + sha256 are verified.

Usage:
    python backend/scripts/download_pasa_selector_model.py
"""
import hashlib
import os
import sys
import time

import requests

URL = "https://hf-mirror.com/bytedance-research/pasa-7b-selector/resolve/main/model.safetensors"
EXPECTED_SIZE = 15231272152
EXPECTED_SHA256 = "84f93d461c93fab5306d67932835992dc58ba0d23872a277c4b113ca8ff4e280"

DEST = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "models", "pasa-7b-selector", "model.safetensors",
)
DEST = os.path.abspath(DEST)
PARTIAL = DEST + ".part"

CHUNK = 4 * 1024 * 1024  # 4 MiB per read
RETRY_SLEEP = 5          # seconds between retries
MAX_EMPTY_READS = 30     # consecutive failed reads before giving up on a session


def main() -> int:
    session = requests.Session()
    session.trust_env = False  # ignore inherited HTTP_PROXY (Clash not running)

    # resume from existing partial
    start = os.path.getsize(PARTIAL) if os.path.exists(PARTIAL) else 0
    if start > 0:
        print(f"[resume] partial file exists: {start / 1e9:.2f} GB", flush=True)

    mode = "ab" if start > 0 else "wb"
    last_log = time.time()
    empty_reads = 0
    with open(PARTIAL, mode) as fh:
        while True:
            current = os.path.getsize(PARTIAL)
            if current >= EXPECTED_SIZE:
                break
            headers = {"Range": f"bytes={current}-"}
            try:
                r = session.get(URL, headers=headers, stream=True, timeout=60)
                if r.status_code not in (200, 206):
                    print(f"[warn] status {r.status_code}, retrying...", flush=True)
                    time.sleep(RETRY_SLEEP)
                    continue
                for chunk in r.iter_content(chunk_size=CHUNK):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    fh.flush()
                r.close()
                empty_reads = 0
                now = time.time()
                if now - last_log >= 10:
                    got = os.path.getsize(PARTIAL)
                    print(f"[progress] {got / 1e9:.2f} / {EXPECTED_SIZE / 1e9:.2f} GB "
                          f"({got / EXPECTED_SIZE * 100:.1f}%)", flush=True)
                    last_log = now
            except (requests.RequestException, OSError) as exc:
                empty_reads += 1
                print(f"[error] {exc!r} (attempt {empty_reads})", flush=True)
                if empty_reads >= MAX_EMPTY_READS:
                    print("giving up after repeated failures", flush=True)
                    return 1
                time.sleep(RETRY_SLEEP)

    size = os.path.getsize(PARTIAL)
    if size != EXPECTED_SIZE:
        print(f"[FAIL] size mismatch: got {size}, expected {EXPECTED_SIZE}", flush=True)
        return 1

    print("[verify] hashing...", flush=True)
    h = hashlib.sha256()
    with open(PARTIAL, "rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(block)
    digest = h.hexdigest()
    if digest != EXPECTED_SHA256:
        print(f"[FAIL] sha256 mismatch: {digest}", flush=True)
        return 1

    os.replace(PARTIAL, DEST)
    print(f"[OK] downloaded + verified -> {DEST}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
