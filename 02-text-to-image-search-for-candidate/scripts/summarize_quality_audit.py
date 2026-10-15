"""Aggregate provisional visual judgments, never infer labels from CLIP scores."""

import csv
import io
import json
import math
from collections import defaultdict
from pathlib import Path

from src.config import ROOT
from src.dataset import file_sha256

OUTPUT = ROOT / "evaluation"


def dcg(grades):
    return sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades))


def scores(rows, labels, k=5):
    grades = [labels[row["image_id"]] for row in rows[:k]]
    ideal = dcg(sorted(labels.values(), reverse=True)[:k])
    return {
        "strict_p5": sum(grade == 2 for grade in grades) / k,
        "related_p5": sum(grade >= 1 for grade in grades) / k,
        "pooled_ndcg5": dcg(grades) / ideal if ideal else 0.0,
        "strict_top1": int(grades[0] == 2),
    }


def run():
    baseline = json.loads((OUTPUT / "baseline.json").read_text(encoding="utf-8"))
    experiment = json.loads((OUTPUT / "view_experiment.json").read_text(encoding="utf-8"))
    review = json.loads((OUTPUT / "review-notes.json").read_text(encoding="utf-8"))
    assert baseline["queries_sha256"] == file_sha256(OUTPUT / "queries.json"), "Queries changed after the run"
    assert baseline["dataset_sha256"] == experiment["signature"]["dataset_sha256"]
    assert baseline["model"] == experiment["signature"]["model"]
    details, judgments, groups = [], [], defaultdict(list)
    for query, alt in zip(baseline["queries"], experiment["queries"], strict=True):
        qid = query["id"]
        assert qid == alt["id"] and query["query"] == alt["query"]
        grades = review["baseline_grades_by_rank"][qid]
        extras = review["additional_grades_by_review_order"][qid]
        assert len(grades) == 10 and len(extras) == len(alt["new_candidates"])
        labels = {row["image_id"]: grade for row, grade in zip(query["results"], grades, strict=True)}
        labels.update({row["image_id"]: grade for row, grade in zip(alt["new_candidates"], extras, strict=True)})
        assert all(grade in (0, 1, 2) for grade in labels.values())
        for image_id, grade in labels.items():
            judgments.append({"query_id": qid, "image_id": image_id, "grade": grade})
        item = {"id": qid, "category": query["category"], "query": query["query"],
                "judged_candidates": len(labels), "fully_matching_in_pool": sum(grade == 2 for grade in labels.values()),
                "metrics": {}, "variants": {}, "note": review["notes"].get(qid, "")}
        for name, rows in {"baseline": query["results"], **alt["variants"]}.items():
            metrics = scores(rows, labels)
            metrics["pool_attainable_strict_p5"] = min(item["fully_matching_in_pool"], 5) / 5
            if name == "baseline":
                metrics["strict_p10"] = sum(grade == 2 for grade in grades) / 10
            item["metrics"][name] = metrics
            item["variants"][name] = [{**row, "grade": labels[row["image_id"]]} for row in rows[:5]]
            groups[("all", name)].append(metrics)
            groups[(query["category"], name)].append(metrics)
        details.append(item)
    summary = {}
    for (category, variant), items in groups.items():
        summary.setdefault(category, {})[variant] = {
            key: sum(item[key] for item in items) / len(items) for key in items[0]
        }
    payload = {
        "method": "Provisional assistant visual judgments, 0/1/2; strict relevance requires grade 2. Pooled nDCG uses only judged candidates, not complete-corpus relevance.",
        "queries_sha256": baseline["queries_sha256"], "review_sha256": file_sha256(OUTPUT / "review-notes.json"),
        "query_count": len(details), "judgment_count": len(judgments),
        "queries_without_confirmed_full_match_in_pool": [item["id"] for item in details if not item["fully_matching_in_pool"]],
        "summary": summary, "queries": details,
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=["query_id", "image_id", "grade"])
    writer.writeheader()
    writer.writerows(judgments)
    (OUTPUT / "judgments.csv").write_text(handle.getvalue(), encoding="utf-8-sig")
    lines = ["严格匹配要求主体及显式条件均成立；结果数量少也可能来自图库匹配内容稀缺。候选池满分数量不是全库正例数量。", "", "| ID | Query | Baseline P@5 | Letterbox P@5 | Dual max P@5 | 池内满分数 | Review |", "|---|---|---:|---:|---:|---:|---|"]
    for item in details:
        values = [f"{item['metrics'][variant]['strict_p5']:.0%}" for variant in ("baseline", "letterbox", "dual_max")]
        lines.append(f"| {item['id']} | {item['query']} | {' | '.join(values)} | {item['fully_matching_in_pool']} | [Top 10](sheets/{item['id']}.jpg) |")
    (OUTPUT / "query-results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"judgments": len(judgments), "summary": summary,
                     "no_full_match_in_pool": payload["queries_without_confirmed_full_match_in_pool"]}, indent=2))


if __name__ == "__main__":
    run()
