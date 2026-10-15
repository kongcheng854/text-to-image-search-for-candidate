"""Save three real query results and CPU timings, without using COCO captions."""

import argparse
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import psutil

from .config import ROOT
from .search import SearchEngine

QUERIES = [
    "A dog running on the beach.",
    "A person riding a bicycle on a street.",
    "A plate of food on a table.",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    output = ROOT / "examples"
    output.mkdir(exist_ok=True)
    start = time.perf_counter()
    engine = SearchEngine(offline=args.offline)
    load_seconds = time.perf_counter() - start
    examples = []
    for number, query in enumerate(QUERIES, 1):
        start = time.perf_counter()
        results = engine.search(query)
        elapsed = time.perf_counter() - start
        repeated = engine.search(query)
        assert results == repeated, "Repeated identical query must return identical results"
        assert len(results) == 10 and len({row.image_id for row in results}) == 10
        assert all(results[i].score >= results[i + 1].score for i in range(9))
        assert all((ROOT / result.path).is_file() for result in results)
        payload = {
            "query": query,
            "ranking": engine.ranking_metadata,
            "elapsed_seconds": round(elapsed, 4),
            "results": [asdict(result) for result in results],
        }
        (output / f"query-{number}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        examples.append(payload)
        print(f"{query}: {elapsed:.3f}s; top result {results[0].image_id}", flush=True)
    memory = psutil.Process().memory_info()
    report = {
        "model": engine.metadata["signature"],
        "ranking": engine.ranking_metadata,
        "dataset_sha256": engine.metadata["dataset_sha256"],
        "image_count": len(engine.records),
        "device": "cpu",
        "python": platform.python_version(),
        "machine_ram_gb": round(psutil.virtual_memory().total / 1024**3, 2),
        "engine_load_seconds": round(load_seconds, 3),
        "peak_rss_mb": round(getattr(memory, "peak_wset", memory.rss) / 1024**2, 2),
        "peak_rss_source": "Windows peak working set" if hasattr(memory, "peak_wset") else "end-of-evaluation RSS (not a peak)",
        "build": engine.metadata["build"],
        "checks": "10 unique valid images, descending scores, repeat-query equality",
        "examples": examples,
    }
    (output / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

