"""Build once: python -m src.build_index [--offline] [--batch-size 8]."""

import argparse
import json
import platform
import time
from contextlib import ExitStack

import numpy as np
import psutil
from PIL import Image

from .config import INDEX_DIR
from .dataset import load_dataset
from .encoder import ClipEncoder
from .index import IndexError, read_index, write_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--offline", action="store_true", help="Use cached model files only")
    parser.add_argument("--force", action="store_true", help="Rebuild an existing valid index")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 32 or args.threads < 1:
        parser.error("batch-size must be 1..32 and threads must be positive")
    started = time.perf_counter()
    records, fingerprint = load_dataset()
    print(f"Validated {len(records)} images. Dataset SHA256: {fingerprint}", flush=True)
    if not args.force:
        try:
            _, _, metadata = read_index()
            if metadata["dataset_sha256"] == fingerprint:
                print("Index is already current; use --force to rebuild.")
                return
        except IndexError:
            pass
    mode = "cached files only" if args.offline else "local files preferred; download if needed"
    print(f"Loading pinned CLIP model on CPU ({mode})...", flush=True)
    encoder = ClipEncoder(offline=args.offline, threads=args.threads)
    chunks = []
    for offset in range(0, len(records), args.batch_size):
        batch = records[offset:offset + args.batch_size]
        with ExitStack() as stack:
            originals = [stack.enter_context(Image.open(row.absolute_path())) for row in batch]
            images = [stack.enter_context(image.convert("RGB")) for image in originals]
            chunks.append(encoder.encode_images(images))
        print(f"Encoded {min(offset + len(batch), len(records))}/{len(records)} images", flush=True)
    vectors = np.concatenate(chunks, axis=0)
    memory = psutil.Process().memory_info()
    metrics = {
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "peak_rss_mb": round(getattr(memory, "peak_wset", memory.rss) / 1024**2, 2),
        "peak_rss_source": "Windows peak working set" if hasattr(memory, "peak_wset") else "end-of-build RSS (not a peak)",
        "batch_size": args.batch_size,
        "threads": args.threads,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine_ram_gb": round(psutil.virtual_memory().total / 1024**3, 2),
        "device": "cpu",
    }
    destination = write_index(vectors, records, fingerprint, metrics)
    print(json.dumps({"index": str(destination), **metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()

