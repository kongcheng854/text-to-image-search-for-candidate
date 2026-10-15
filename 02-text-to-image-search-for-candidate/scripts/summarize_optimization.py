"""Summarize shared visual labels for development and frozen validation queries."""

import json
import statistics
from collections import defaultdict

from scripts.run_quality_audit import OUTPUT
from scripts.summarize_quality_audit import scores
from src.dataset import file_sha256


def average(items):
    return {key: statistics.mean(item[key] for item in items) for key in items[0]}


def main():
    directory = OUTPUT / "optimization"
    development = json.loads((directory / "development-final.json").read_text(encoding="utf-8"))
    original = json.loads((OUTPUT / "metrics.json").read_text(encoding="utf-8"))
    validation = json.loads((directory / "validation.json").read_text(encoding="utf-8"))
    review = json.loads((directory / "validation-review.json").read_text(encoding="utf-8"))
    assert validation["validation_queries_sha256"] == file_sha256(OUTPUT / "validation-queries.json")
    baseline_by_id = {query["id"]: query for query in original["queries"]}
    dev_groups = defaultdict(lambda: {"baseline": [], "optimized": []})
    for query in development:
        before = baseline_by_id[query["id"]]["metrics"]["baseline"]
        before = {key: before[key] for key in query["metrics"]}
        for category in ("all", query["category"]):
            dev_groups[category]["baseline"].append(before)
            dev_groups[category]["optimized"].append(query["metrics"])
    dev_summary = {category: {mode: average(items) for mode, items in variants.items()} for category, variants in dev_groups.items()}
    comparisons = []
    for query in validation["queries"]:
        grades = review["grades_by_review_order"][query["id"]]
        assert len(grades) == len(query["review_candidates"])
        labels = {row["image_id"]: grade for row, grade in zip(query["review_candidates"], grades, strict=True)}
        comparisons.append({"id": query["id"], "query": query["query"],
                            "baseline": scores(query["baseline"], labels), "optimized": scores(query["optimized"], labels)})
    validation_summary = {mode: average([query[mode] for query in comparisons]) for mode in ("baseline", "optimized")}
    times = {mode: [query[mode + "_seconds"] for query in validation["queries"]] for mode in ("baseline", "optimized")}
    report = {
        "ranking": validation["ranking"],
        "development": dev_summary,
        "validation": validation_summary,
        "validation_queries": comparisons,
        "validation_judgment_count": sum(len(query["review_candidates"]) for query in validation["queries"]),
        "review_sha256": file_sha256(directory / "validation-review.json"),
        "validation_queries_sha256": validation["validation_queries_sha256"],
        "engine_load_seconds": validation["engine_load_seconds"],
        "peak_working_set_mb": validation["peak_working_set_mb"],
        "median_query_ms": {mode: statistics.median(values) * 1000 for mode, values in times.items()},
        "mean_query_ms": {mode: statistics.mean(values) * 1000 for mode, values in times.items()},
        "limitations": "Provisional assistant visual labels. Development was used for selection; 10 validation queries were frozen before selection and did not tune weights. Small same-gallery sample, no claim of statistical significance or complete-corpus recall.",
    }
    (directory / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
