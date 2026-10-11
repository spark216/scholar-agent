"""Probe the RAM/VRAM spike of the 4-bit selector load with the BGE-M3 encoder
already resident (the real eval order), and test whether bounded memory options
avoid the segfault.

Run: python -m backend.scripts.probe_selector_mem
"""
from __future__ import annotations

import sys
import threading
import time

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import psutil

MAX_RAM = float("inf")


def _sampler(stop: threading.Event) -> None:
    peak = 0.0
    while not stop.is_set():
        peak = max(peak, psutil.Process().memory_info().rss / 1e9)
        time.sleep(0.2)
    print(f"PEAK RAM during sampler window: {peak:.2f}GB", flush=True)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="ascii", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    import torch
    from backend.app.services.embeddings import get_bge_m3_encoder

    print("loading BGE-M3 encoder (as the real eval does during retrieval)...", flush=True)
    enc = get_bge_m3_encoder()
    print(f"encoder provider={enc.provider} RAM={psutil.Process().memory_info().rss/1e9:.2f}GB "
          f"VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_dir = ROOT / "models" / "pasa-7b-selector"
    stop = threading.Event()
    t = threading.Thread(target=_sampler, args=(stop,))
    t.start()
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        low_cpu_mem_usage=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        ),
        device_map="auto",
        max_memory={"cpu": "6GiB", 0: "11GiB"},
        torch_dtype=torch.bfloat16,
    )
    stop.set()
    t.join()
    print(f"selector loaded in {time.perf_counter()-t0:.1f}s "
          f"RAM={psutil.Process().memory_info().rss/1e9:.2f}GB "
          f"VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB",
          flush=True)
    print("PROBE OK", flush=True)


if __name__ == "__main__":
    main()
