"""Exact vector index with immutable generations and an atomic current pointer."""

import json
import os
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import INDEX_DIR, model_signature
from .dataset import ImageRecord, file_sha256


class IndexError(ValueError):
    pass


def write_index(
    embeddings: np.ndarray,
    records: list[ImageRecord],
    dataset_fingerprint: str,
    metrics: dict,
    directory: Path = INDEX_DIR,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    generation = uuid.uuid4().hex
    destination = directory / generation
    destination.mkdir()
    np.save(destination / "embeddings.npy", embeddings, allow_pickle=False)
    (destination / "images.json").write_text(
        json.dumps([asdict(record) for record in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "schema_version": 1,
        "generation": generation,
        "signature": model_signature(),
        "dataset_sha256": dataset_fingerprint,
        "count": len(records),
        "files": {
            name: file_sha256(destination / name)
            for name in ("embeddings.npy", "images.json")
        },
        "build": metrics,
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pointer_temp = directory / f"{generation}.lock.tmp"
    pointer_temp.write_text(json.dumps({"generation": generation}), encoding="utf-8")
    # Readers see either the old complete index or the new complete index.
    os.replace(pointer_temp, directory / "current.json")
    return destination


def read_index(directory: Path = INDEX_DIR):
    try:
        pointer = json.loads((directory / "current.json").read_text(encoding="utf-8"))
        generation = pointer["generation"]
        if not isinstance(generation, str) or len(generation) != 32:
            raise ValueError("invalid index generation")
        int(generation, 16)
        location = directory / generation
        metadata = json.loads((location / "metadata.json").read_text(encoding="utf-8"))
        if metadata["schema_version"] != 1 or metadata["generation"] != generation:
            raise ValueError("unsupported index metadata")
        if metadata["signature"] != model_signature():
            raise ValueError("model or preprocessing version changed; rebuild the index")
        for name in ("embeddings.npy", "images.json"):
            if file_sha256(location / name) != metadata["files"][name]:
                raise ValueError(f"index file is damaged: {name}")
        rows = json.loads((location / "images.json").read_text(encoding="utf-8"))
        records = [ImageRecord(**row) for row in rows]
        vectors = np.load(location / "embeddings.npy", allow_pickle=False)
        expected_shape = (len(records), metadata["signature"]["dimension"])
        if vectors.shape != expected_shape or len(records) != metadata["count"]:
            raise ValueError("image mapping and vector dimensions do not agree")
        if len({record.image_id for record in records}) != len(records):
            raise ValueError("duplicate image IDs in index")
        if vectors.dtype != np.float32 or not np.isfinite(vectors).all():
            raise ValueError("invalid embedding values")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5):
            raise ValueError("index embeddings must be normalized")
        vectors.flags.writeable = False
        return vectors, records, metadata
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise IndexError(
            f"Cannot use image index: {exc}. Run: python -m src.build_index"
        ) from exc

