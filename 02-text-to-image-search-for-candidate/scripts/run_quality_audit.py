"""Run frozen, self-written queries and render original-image review sheets.

python -m scripts.run_quality_audit [--views]
--views adds a full-image letterbox diagnostic index without changing production.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.config import ROOT
from src.dataset import file_sha256
from src.search import SearchEngine, rank_vectors

OUTPUT = ROOT / "evaluation"


def font(size):
    for name in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"):
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default(size=size)


def sheet(query, results, filename, title="Baseline / full original images"):
    width, height = 1500, 750
    canvas = Image.new("RGB", (width, height), "#f6f7f3")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 12), f"{query['id']} | {query['query']}", fill="#183c32", font=font(25))
    draw.text((18, 48), title, fill="#59695f", font=font(17))
    for number, row in enumerate(results):
        column, line = number % 5, number // 5
        x, y = column * 300 + 10, 86 + line * 325
        with Image.open(ROOT / row["path"]) as image:
            thumbnail = ImageOps.pad(image.convert("RGB"), (280, 250), color="#e6e9e1")
            canvas.paste(thumbnail, (x, y))
        draw.text((x, y + 255), f"#{row['rank']}  {row['image_id']}", fill="#172c24", font=font(16))
        draw.text((x, y + 279), f"score {row['score']:.4f}", fill="#526458", font=font(16))
    canvas.save(filename)


def render_additions(queries):
    for category in dict.fromkeys(item["category"] for item in queries):
        group = [item for item in queries if item["category"] == category]
        canvas = Image.new("RGB", (1500, 400 * len(group)), "#f6f7f3")
        draw = ImageDraw.Draw(canvas)
        for number, item in enumerate(group):
            path = OUTPUT / "sheets" / f"{item['id']}-additional.jpg"
            if item["new_candidates"]:
                with Image.open(path) as original:
                    canvas.paste(original.crop((0, 0, 1500, 400)), (0, number * 400))
            else:
                draw.text((18, number * 400 + 12), f"{item['id']} | {item['query']}", fill="#183c32", font=font(25))
                draw.text((18, number * 400 + 60), "No new top-5 candidates.", fill="#59695f", font=font(20))
        canvas.save(OUTPUT / "sheets" / f"{category}-additions.jpg")


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", action="store_true")
    args = parser.parse_args()
    queries_path = OUTPUT / "queries.json"
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    (OUTPUT / "sheets").mkdir(exist_ok=True)
    start = time.perf_counter()
    engine = SearchEngine(offline=True, ranking="baseline")
    load_seconds = time.perf_counter() - start
    features, results = [], []
    for query in queries:
        start = time.perf_counter()
        feature = engine.encoder.encode_text(query["query"])
        features.append(feature[0])
        order, scores = rank_vectors(engine.vectors, feature, engine.ids, 10)
        rows = [
            {"rank": rank, "image_id": engine.records[i].image_id,
             "path": f"materials/data/{engine.records[i].path}", "score": float(scores[i])}
            for rank, i in enumerate(order, 1)
        ]
        elapsed = time.perf_counter() - start
        results.append({**query, "elapsed_seconds": elapsed, "results": rows})
        sheet(query, rows, OUTPUT / "sheets" / f"{query['id']}.jpg")
        print(f"{query['id']} {query['query']} top1={rows[0]['image_id']}", flush=True)
    report = {
        "queries_sha256": file_sha256(queries_path),
        "model": engine.metadata["signature"],
        "dataset_sha256": engine.metadata["dataset_sha256"],
        "engine_load_seconds": load_seconds,
        "image_count": len(engine.records),
        "queries": results,
    }
    (OUTPUT / "baseline.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.save(OUTPUT / "text_features.npy", np.asarray(features, dtype=np.float32), allow_pickle=False)
    if args.views:
        # Diagnostic only: original aspect ratio retained and padded to 224x224.
        cache = OUTPUT / "letterbox.npy"
        cache_metadata = OUTPUT / "letterbox-metadata.json"
        signature = {"model": engine.metadata["signature"], "dataset_sha256": engine.metadata["dataset_sha256"],
                     "view": "PIL-ImageOps.pad-224-bicubic-fill-123-117-104-v1"}
        reuse = False
        if cache.exists() and cache_metadata.exists():
            metadata = json.loads(cache_metadata.read_text(encoding="utf-8"))
            reuse = metadata.get("signature") == signature and metadata.get("sha256") == file_sha256(cache)
        if reuse:
            full = np.load(cache, allow_pickle=False)
            view_seconds = None
        else:
            start = time.perf_counter()
            chunks = []
            for offset in range(0, len(engine.records), 8):
                images = []
                try:
                    for record in engine.records[offset:offset + 8]:
                        with Image.open(record.absolute_path()) as image:
                            images.append(ImageOps.pad(image.convert("RGB"), (224, 224),
                                method=Image.Resampling.BICUBIC, color=(123, 117, 104)))
                    chunks.append(engine.encoder.encode_images(images))
                finally:
                    for image in images:
                        image.close()
                if offset % 160 == 0:
                    print(f"Letterbox view {offset + len(images)}/{len(engine.records)}", flush=True)
            full = np.concatenate(chunks)
            view_seconds = time.perf_counter() - start
            np.save(cache, full, allow_pickle=False)
            cache_metadata.write_text(json.dumps({"signature": signature, "sha256": file_sha256(cache)}, indent=2), encoding="utf-8")
        alternative = []
        for query, feature, baseline in zip(queries, features, results):
            center_scores = engine.vectors @ feature
            full_scores = full @ feature
            variants = {}
            for name, scores in {"letterbox": full_scores, "dual_max": np.maximum(center_scores, full_scores)}.items():
                order = np.lexsort((np.asarray(engine.ids), -scores))[:10]
                variants[name] = [
                    {"rank": rank, "image_id": engine.records[i].image_id,
                     "path": f"materials/data/{engine.records[i].path}", "score": float(scores[i])}
                    for rank, i in enumerate(order, 1)
                ]
            # Review only candidates not already on the baseline sheet. One label
            # per (query, image) is then shared consistently by every variant.
            seen = {row["image_id"] for row in baseline["results"]}
            additions = []
            for name in ("letterbox", "dual_max"):
                for row in variants[name][:5]:
                    if row["image_id"] not in seen:
                        additions.append({**row, "rank": len(additions) + 1})
                        seen.add(row["image_id"])
            if additions:
                sheet(query, additions, OUTPUT / "sheets" / f"{query['id']}-additional.jpg", "Additional top-5 candidates from letterbox / dual-max")
            alternative.append({**query, "variants": variants, "new_candidates": additions})
        (OUTPUT / "view_experiment.json").write_text(json.dumps({
            "signature": signature, "encoding_seconds": view_seconds, "queries": alternative,
        }, indent=2) + "\n", encoding="utf-8")
        render_additions(alternative)


if __name__ == "__main__":
    run()
