"""A single pinned CLIP model, on CPU, with bounded batches and text length."""

import os
import json
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from .config import MODEL_ID, MODEL_REVISION
from .dataset import file_sha256


def normalize_vectors(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected a finite two-dimensional embedding matrix.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms < 1e-12).any():
        raise ValueError("Cannot normalize a zero embedding.")
    return np.ascontiguousarray(values / norms, dtype=np.float32)


class ClipEncoder:
    def __init__(self, offline: bool = False, threads: int | None = None):
        # Lazy imports let data validation and ranking tests run without loading a model.
        import torch
        from transformers import CLIPModel, CLIPProcessor

        threads = threads or min(4, os.cpu_count() or 1)
        if threads < 1:
            raise ValueError("CPU threads must be positive.")
        torch.set_num_threads(threads)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        self._torch = torch
        self._lock = threading.Lock()
        from .prepare_model import BPE_SHA256, MODEL_DIR, WEIGHTS_SHA256
        local_source = MODEL_DIR / "source.json"
        if local_source.exists():
            source = json.loads(local_source.read_text(encoding="utf-8"))
            if source.get("weights_sha256") != WEIGHTS_SHA256 or source.get("tokenizer_sha256") != BPE_SHA256:
                raise ValueError("Local model source does not match the pinned OpenAI checkpoint.")
            for name, digest in source["files"].items():
                if Path(name).name != name or file_sha256(MODEL_DIR / name) != digest:
                    raise ValueError("Local model is damaged. Run: python -m src.prepare_model")
            location = str(MODEL_DIR)
            options = {"local_files_only": True}
        else:
            location = MODEL_ID
            options = {"revision": MODEL_REVISION, "local_files_only": offline}
        self.processor = CLIPProcessor.from_pretrained(
            location, **options, use_fast=False
        )
        self.model = CLIPModel.from_pretrained(
            location, **options,
            use_safetensors=local_source.exists(),
            attn_implementation="eager",
        ).to("cpu").eval()

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=list(images), return_tensors="pt")
        with self._lock, self._torch.inference_mode():
            features = self.model.get_image_features(**inputs)
        return normalize_vectors(features.cpu().numpy())

    def encode_text(self, query: str) -> np.ndarray:
        return self.encode_texts([query])

    def encode_texts(self, queries: Sequence[str], batch_size: int = 16) -> np.ndarray:
        if not queries or batch_size < 1:
            raise ValueError("Text batch must be nonempty and batch_size must be positive.")
        cleaned = [self._validate_text(query) for query in queries]
        batches = []
        for offset in range(0, len(cleaned), batch_size):
            inputs = self.processor(text=cleaned[offset:offset + batch_size], return_tensors="pt", padding=True)
            with self._lock, self._torch.inference_mode():
                features = self.model.get_text_features(**inputs)
            batches.append(normalize_vectors(features.cpu().numpy()))
        return np.concatenate(batches, axis=0)

    def _validate_text(self, query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Please enter an English description.")
        query = query.strip()
        # Refuse overlong queries instead of silently dropping their final details.
        tokens = self.processor.tokenizer(query, truncation=False)["input_ids"]
        limit = self.model.config.text_config.max_position_embeddings
        if len(tokens) > limit:
            raise ValueError(
                f"Description is too long ({len(tokens)} tokens; maximum {limit}). "
                "Please use a shorter English description."
            )
        return query

