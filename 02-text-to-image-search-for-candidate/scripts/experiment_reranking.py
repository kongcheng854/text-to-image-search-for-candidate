"""Compare fixed, general object-cue ranking formulas on the development set."""

import json
from pathlib import Path

import numpy as np

from scripts.run_quality_audit import OUTPUT, sheet
from src.dataset import file_sha256
from src.query import CONFUSIONS, OBJECTS, object_prompt, parse_query
from src.search import SearchEngine


def run():
    output = OUTPUT / "optimization"
    output.mkdir(exist_ok=True)
    (output / "sheets").mkdir(exist_ok=True)
    engine = SearchEngine(offline=True, ranking="baseline")
    queries = json.loads((OUTPUT / "baseline.json").read_text(encoding="utf-8"))["queries"]
    texts = np.load(OUTPUT / "text_features.npy", allow_pickle=False)
    full = np.load(OUTPUT / "letterbox.npy", allow_pickle=False)
    labels = json.loads((OUTPUT / "review-notes.json").read_text(encoding="utf-8"))
    views = json.loads((OUTPUT / "view_experiment.json").read_text(encoding="utf-8"))["queries"]
    vocabulary = sorted(OBJECTS)
    prototypes = np.concatenate([engine.encoder.encode_text(object_prompt(name)) for name in vocabulary])
    np.save(output / "object-prototypes.npy", prototypes, allow_pickle=False)
    (output / "object-prototypes.json").write_text(json.dumps({
        "objects": vocabulary, "model": engine.metadata["signature"],
        "sha256": file_sha256(output / "object-prototypes.npy"),
    }, indent=2), encoding="utf-8")
    center_cues = engine.vectors @ prototypes.T
    full_cues = full @ prototypes.T
    reports = []
    for query, feature, previous in zip(queries, texts, views, strict=True):
        qid = query["id"]
        plan = parse_query(query["query"])
        center = engine.vectors @ feature
        globally = full @ feature
        photo = engine.encoder.encode_text("a photo of " + query["query"][0].lower() + query["query"][1:])[0]
        ensemble = feature + photo
        ensemble = ensemble / np.linalg.norm(ensemble)
        prompted = engine.vectors @ ensemble
        if plan.primary and not plan.negated:
            positive = vocabulary.index(plan.primary)
            negatives = [vocabulary.index(name) for name in CONFUSIONS.get(plan.primary, ()) if name not in plan.objects]
            anchor = center_cues[:, positive]
            penalty = np.maximum(0, center_cues[:, negatives].max(axis=1) - anchor) if negatives else np.zeros_like(anchor)
            mix_cues = .75 * center_cues + .25 * full_cues
            mix_anchor = mix_cues[:, positive]
            mix_penalty = np.maximum(0, mix_cues[:, negatives].max(axis=1) - mix_anchor) if negatives else np.zeros_like(anchor)
            conservative_penalty = penalty if len([name for name in plan.objects if name != "person"]) == 1 else np.zeros_like(anchor)
            variants = {
                "anchor_025": center + .25 * anchor,
                "contrast_050": center - .5 * penalty,
                "anchor_contrast": center + .25 * anchor - .5 * penalty,
                "view_contrast": .75 * center + .25 * globally - .5 * mix_penalty,
                "conservative_contrast": center - .5 * conservative_penalty,
                "prompt_ensemble": prompted,
                "prompt_contrast": prompted - .5 * conservative_penalty,
            }
        else:
            variants = {name: center for name in ("anchor_025", "contrast_050", "anchor_contrast", "view_contrast", "conservative_contrast", "prompt_ensemble", "prompt_contrast")}
        known = {row["image_id"]: grade for row, grade in zip(query["results"], labels["baseline_grades_by_rank"][qid], strict=True)}
        known.update({row["image_id"]: grade for row, grade in zip(previous["new_candidates"], labels["additional_grades_by_review_order"][qid], strict=True)})
        added, rankings = [], {}
        for name, scores in variants.items():
            order = np.lexsort((np.asarray(engine.ids), -scores))[:10]
            rows = [{"rank": rank, "image_id": engine.records[i].image_id,
                     "path": f"materials/data/{engine.records[i].path}", "score": float(scores[i])}
                    for rank, i in enumerate(order, 1)]
            rankings[name] = rows
            for row in rows[:5]:
                if row["image_id"] not in known and not any(item["image_id"] == row["image_id"] for item in added):
                    added.append({**row, "rank": len(added) + 1})
        if added:
            # Multiple review pages if general formulas introduce >10 candidates.
            for offset in range(0, len(added), 10):
                sheet(query, added[offset:offset + 10], output / "sheets" / f"{qid}-{offset // 10 + 1}.jpg", "Previously unjudged candidates from object-cue experiments")
        # A within-known-pool comparison is a diagnostic, never reported as global P@5.
        indices = np.asarray([engine.ids.index(image_id) for image_id in known])
        ids = [engine.ids[i] for i in indices]
        pooled = {}
        for name, scores in {"baseline": center, **variants}.items():
            order = np.lexsort((np.asarray(ids), -scores[indices]))[:5]
            pooled[name] = sum(known[ids[i]] == 2 for i in order) / 5
        reports.append({"id": qid, "category": query["category"], "query": query["query"],
                        "plan": {"objects": plan.objects, "primary": plan.primary, "negated": plan.negated},
                        "variants": rankings, "new_candidates": added, "known_pool_p5": pooled})
        print(qid, "new", len(added), "known-pool", pooled, flush=True)
    result = {"model": engine.metadata["signature"], "dataset_sha256": engine.metadata["dataset_sha256"],
              "validation_queries_sha256_before_selection": file_sha256(OUTPUT / "validation-queries.json"),
              "queries": reports}
    (output / "development.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("New judgments:", sum(len(item["new_candidates"]) for item in reports), flush=True)
    print("Known-pool diagnostic means:", {name: np.mean([q["known_pool_p5"][name] for q in reports]) for name in reports[0]["known_pool_p5"]})


if __name__ == "__main__":
    run()
