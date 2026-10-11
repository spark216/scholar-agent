"""Download bytedance-research/pasa-7b-selector via hf-mirror (Chinese mirror).

Usage:
    HF_ENDPOINT=https://hf-mirror.com python backend/scripts/download_pasa_selector.py

Notes:
    - hf-mirror.com is used because huggingface.co is blocked in CN.
    - ignore_patterns=[".DS_Store"]: hf-mirror returns 403 on .DS_Store which
      aborts the whole snapshot download otherwise.
    - Proxy env vars are cleared: Bash sessions inherit HTTP_PROXY=127.0.0.1:7890
      (Clash usually not running) which would make requests fail.
"""
import os
import sys

# Force direct connection: clear proxy env inherited from the shell.
for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(key, None)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download

REPO_ID = "bytedance-research/pasa-7b-selector"
LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "pasa-7b-selector")


def main():
    print(f"Downloading {REPO_ID} -> {os.path.abspath(LOCAL_DIR)} via {os.environ['HF_ENDPOINT']}", flush=True)
    path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=LOCAL_DIR,
        ignore_patterns=[".DS_Store", "*.h5", "*.msgpack", "*.onnx", "imgs/*"],
        max_workers=4,
    )
    print(f"OK snapshot at {path}", flush=True)
    # sanity: model.safetensors should be ~15GB for a 7B bf16 model
    st = os.path.join(LOCAL_DIR, "model.safetensors")
    if os.path.exists(st):
        print(f"model.safetensors size: {os.path.getsize(st) / 1e9:.2f} GB", flush=True)
    else:
        print("WARNING: model.safetensors missing!", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
