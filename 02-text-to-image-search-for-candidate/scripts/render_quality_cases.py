"""Show model-visible crop regions and ranking changes for three observed cases."""

import json

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from transformers import CLIPImageProcessor

from scripts.run_quality_audit import OUTPUT, font
from src.config import ROOT
from src.index import read_index
from src.prepare_model import MODEL_DIR


def run():
    vectors, records, metadata = read_index()
    full = np.load(OUTPUT / "letterbox.npy", allow_pickle=False)
    texts = np.load(OUTPUT / "text_features.npy", allow_pickle=False)
    queries = json.loads((OUTPUT / "baseline.json").read_text(encoding="utf-8"))["queries"]
    ids = np.asarray([record.image_id for record in records])
    processor = CLIPImageProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    cases = [
        ("R05", "coco-000000473118", "Bicycle at the top is outside the center crop."),
        ("A04", "coco-000000256941", "Full view shows an umbrella, but no person holding it."),
        ("R01", "coco-000000213445", "Cat on a plate: the chair relation is not satisfied."),
    ]
    canvas = Image.new("RGB", (1100, 1290), "#f6f7f3")
    draw = ImageDraw.Draw(canvas)
    data = []
    for line, (qid, image_id, caption) in enumerate(cases):
        qindex = next(i for i, query in enumerate(queries) if query["id"] == qid)
        record_index = next(i for i, record in enumerate(records) if record.image_id == image_id)
        feature = texts[qindex]
        center_scores, full_scores = vectors @ feature, full @ feature
        ranks = {}
        for name, scores in {"center": center_scores, "letterbox": full_scores,
                             "dual_max": np.maximum(center_scores, full_scores)}.items():
            order = np.lexsort((ids, -scores))
            ranks[name] = int(np.flatnonzero(order == record_index)[0]) + 1
        y = line * 430
        draw.text((16, y + 8), f"{qid} | {queries[qindex]['query']}", fill="#183c32", font=font(23))
        draw.text((16, y + 40), caption, fill="#536659", font=font(18))
        with Image.open(records[record_index].absolute_path()) as original:
            rgb = original.convert("RGB")
            w, h = rgb.size
            side = min(w, h)
            box = ((w - side) / 2, (h - side) / 2, (w + side) / 2, (h + side) / 2)
            annotated = rgb.copy()
            ImageDraw.Draw(annotated).rectangle(box, outline="#ec3b39", width=max(3, round(side / 80)))
            canvas.paste(ImageOps.pad(annotated, (470, 300), color="#e4e9e1"), (16, y + 75))
            center = processor(images=rgb, do_normalize=False, do_rescale=False, return_tensors="np")["pixel_values"][0]
            center = Image.fromarray(np.moveaxis(center, 0, -1).astype(np.uint8))
            letterbox = ImageOps.pad(rgb, (224, 224), method=Image.Resampling.BICUBIC, color=(123, 117, 104))
            canvas.paste(center, (550, y + 110))
            canvas.paste(letterbox, (830, y + 110))
        draw.text((16, y + 379), "Original + approximate crop boundary", fill="#183c32", font=font(17))
        draw.text((550, y + 347), f"Center rank: {ranks['center']}", fill="#183c32", font=font(18))
        draw.text((830, y + 347), f"Letterbox rank: {ranks['letterbox']}", fill="#183c32", font=font(18))
        data.append({"query_id": qid, "image_id": image_id, "ranks": ranks,
                     "center_retained_area_fraction": side**2 / (w * h), "observation": caption})
    canvas.save(OUTPUT / "crop-cases.png")
    (OUTPUT / "case-studies.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    run()
