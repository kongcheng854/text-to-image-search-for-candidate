"""Tests exercise failures that can silently return the wrong image or ranking."""

import json
import numpy as np
import pytest
from PIL import Image

from src.config import DIMENSION
from src.dataset import DatasetError, ImageRecord, load_dataset
from src.encoder import normalize_vectors
from src.index import IndexError, read_index, write_index
from src.search import rank_vectors


def make_dataset(tmp_path):
    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    Image.new("RGB", (20, 10), "orange").save(data / "images" / "one.jpg")
    manifest = data / "images.jsonl"
    manifest.write_text(json.dumps({"image_id": "one", "path": "images/one.jpg"}), encoding="utf-8")
    return data, manifest


def test_dataset_change_invalidates_fingerprint(tmp_path):
    data, manifest = make_dataset(tmp_path)
    records, before = load_dataset(manifest, data)
    assert records[0].absolute_path(data).is_file()
    Image.new("RGB", (20, 10), "blue").save(data / "images" / "one.jpg")
    _, after = load_dataset(manifest, data)
    assert before != after


def test_manifest_cannot_escape_image_directory(tmp_path):
    data, manifest = make_dataset(tmp_path)
    Image.new("RGB", (20, 10)).save(data / "outside.jpg")
    manifest.write_text(json.dumps({"image_id": "one", "path": "images/../outside.jpg"}), encoding="utf-8")
    with pytest.raises(DatasetError, match="inside"):
        load_dataset(manifest, data)


def test_duplicate_manifest_images_are_rejected(tmp_path):
    data, manifest = make_dataset(tmp_path)
    row = manifest.read_text(encoding="utf-8")
    manifest.write_text(row + "\n" + row, encoding="utf-8")
    with pytest.raises(DatasetError, match="duplicate"):
        load_dataset(manifest, data)


def test_broken_image_is_rejected(tmp_path):
    data, manifest = make_dataset(tmp_path)
    (data / "images" / "one.jpg").write_bytes(b"broken")
    with pytest.raises(DatasetError, match="line 1"):
        load_dataset(manifest, data)


def test_cosine_ranking_and_deterministic_ties():
    vectors = normalize_vectors(np.array([[1, 0], [0, 1], [1, 0], [-1, 0]]))
    query = normalize_vectors(np.array([[4, 0]]))
    order, scores = rank_vectors(vectors, query, ["b", "c", "a", "d"], 3)
    assert order.tolist() == [2, 0, 1]
    assert scores.tolist() == [1.0, 0.0, 1.0, -1.0]


@pytest.mark.parametrize("invalid", [0, -1, 5, True, 1.5])
def test_invalid_top_k(invalid):
    with pytest.raises(ValueError, match="top_k"):
        rank_vectors(np.eye(2), np.array([[1.0, 0.0]]), ["a", "b"], invalid)


@pytest.mark.parametrize("values", [[[0.0, 0.0]], [[float("nan"), 1.0]]])
def test_invalid_embeddings(values):
    with pytest.raises(ValueError):
        normalize_vectors(np.array(values))


def test_index_roundtrip_and_corruption_detection(tmp_path):
    vectors = np.zeros((1, DIMENSION), dtype=np.float32)
    vectors[0, 0] = 1
    records = [ImageRecord("a", "images/a.jpg", "abc")]
    generation = write_index(vectors, records, "fingerprint", {}, tmp_path)
    loaded, mapped, metadata = read_index(tmp_path)
    assert np.array_equal(vectors, loaded)
    assert mapped == records
    assert metadata["dataset_sha256"] == "fingerprint"
    (generation / "images.json").write_text("[]", encoding="utf-8")
    with pytest.raises(IndexError, match="damaged"):
        read_index(tmp_path)


def test_model_mismatch_requires_rebuilding(tmp_path):
    vectors = np.zeros((1, DIMENSION), dtype=np.float32)
    vectors[0, 0] = 1
    generation = write_index(vectors, [ImageRecord("a", "images/a.jpg", "abc")], "fp", {}, tmp_path)
    path = generation / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["signature"]["model_revision"] = "different"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(IndexError, match="version changed"):
        read_index(tmp_path)

