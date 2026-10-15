"""Evaluate the selected production ranking, then run frozen unseen queries."""

import json
import time
from dataclasses import asdict

import numpy as np
import psutil

from scripts.run_quality_audit import OUTPUT, sheet
from scripts.summarize_quality_audit import scores
from src.dataset import file_sha256
from src.ranking import ranking_signature
from src.search import SearchEngine, rank_vectors


def baseline_search(engine, query):
    feature = engine.encoder.encode_text(query)
    order, values = rank_vectors(engine.vectors, feature, engine.id_array, 10)
    return [{"rank": rank, "image_id": engine.records[i].image_id,
             "path": f"materials/data/{engine.records[i].path}", "score": float(values[i])}
            for rank, i in enumerate(order, 1)]


def main():
    destination = OUTPUT / "optimization"
    selected = json.loads((destination / "selection.json").read_text(encoding="utf-8"))
    assert selected["selected"] == ranking_signature(), "Selected ranking changed before validation"
    assert selected["validation_queries_sha256_before_run"] == file_sha256(OUTPUT / "validation-queries.json")
    started = time.perf_counter()
    engine = SearchEngine(offline=True, ranking="optimized")
    load_seconds = time.perf_counter() - started
    dev = json.loads((OUTPUT / "baseline.json").read_text(encoding="utf-8"))
    review = json.loads((OUTPUT / "review-notes.json").read_text(encoding="utf-8"))
    views = json.loads((OUTPUT / "view_experiment.json").read_text(encoding="utf-8"))["queries"]
    development = []
    for query, view in zip(dev["queries"], views, strict=True):
        qid = query["id"]
        labels = {row["image_id"]: grade for row, grade in zip(query["results"], review["baseline_grades_by_rank"][qid], strict=True)}
        labels.update({row["image_id"]: grade for row, grade in zip(view["new_candidates"], review["additional_grades_by_review_order"][qid], strict=True)})
        started = time.perf_counter()
        rows = [asdict(row) for row in engine.search(query["query"])]
        elapsed = time.perf_counter() - started
        missing = [row for row in rows[:5] if row["image_id"] not in labels]
        if missing:
            sheet(query, missing, destination / "sheets" / f"{qid}-final-new.jpg", "New candidates requiring visual judgment")
        metrics = scores(rows, labels) if not missing else None
        development.append({"id": qid, "category": query["category"], "query": query["query"],
                            "results": rows, "metrics": metrics, "missing_judgments": missing, "elapsed_seconds": elapsed})
    (destination / "development-final.json").write_text(json.dumps(development, indent=2) + "\n", encoding="utf-8")
    assert not any(query["missing_judgments"] for query in development), "Review new development candidates before selecting this ranking"
    print("Development actual full-corpus P@5:", np.mean([query["metrics"]["strict_p5"] for query in development]), flush=True)
    validation = []
    queries = json.loads((OUTPUT / "validation-queries.json").read_text(encoding="utf-8"))
    for query in queries:
        started = time.perf_counter()
        baseline = baseline_search(engine, query["query"])
        baseline_seconds = time.perf_counter() - started
        started = time.perf_counter()
        optimized = [asdict(row) for row in engine.search(query["query"])]
        optimized_seconds = time.perf_counter() - started
        assert len({row["image_id"] for row in optimized}) == 10
        assert all(optimized[i]["score"] >= optimized[i + 1]["score"] for i in range(9))
        assert optimized == [asdict(row) for row in engine.search(query["query"])], "Results must repeat exactly"
        pooled = []
        for row in baseline[:5] + optimized[:5]:
            if not any(item["image_id"] == row["image_id"] for item in pooled):
                pooled.append({**row, "rank": len(pooled) + 1})
        sheet(query, pooled, destination / "sheets" / f"{query['id']}-review.jpg", "Baseline / optimized top-5 union; one shared relevance judgment per image")
        sheet(query, baseline, destination / "sheets" / f"{query['id']}-baseline.jpg", "Baseline top 10")
        sheet(query, optimized, destination / "sheets" / f"{query['id']}-optimized.jpg", "Optimized top 10; score includes text-side reranking")
        validation.append({**query, "baseline": baseline, "optimized": optimized, "review_candidates": pooled,
                           "baseline_seconds": baseline_seconds, "optimized_seconds": optimized_seconds})
        print(query["id"], "review candidates", len(pooled), "baseline", baseline[0]["image_id"], "optimized", optimized[0]["image_id"], flush=True)
    memory = psutil.Process().memory_info()
    (destination / "validation.json").write_text(json.dumps({
        "ranking": engine.ranking_metadata, "model": engine.metadata["signature"],
        "dataset_sha256": engine.metadata["dataset_sha256"],
        "validation_queries_sha256": selected["validation_queries_sha256_before_run"],
        "engine_load_seconds": load_seconds,
        "peak_working_set_mb": getattr(memory, "peak_wset", memory.rss) / 1024**2,
        "queries": validation,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
