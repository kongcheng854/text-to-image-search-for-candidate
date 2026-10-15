"""Paths are anchored to the project, independent of the working directory."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "materials" / "data"
MANIFEST = DATA_DIR / "images.jsonl"
IMAGE_DIR = DATA_DIR / "images"
INDEX_DIR = ROOT / "artifacts" / "index"
MODEL_ID = "openai/clip-vit-base-patch32"
MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
DIMENSION = 512
PIPELINE_VERSION = "clip-rgb-center-crop-fp32-l2-v1"

# Keep downloaded files in the project; respect explicit user overrides.
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def model_signature() -> dict:
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dimension": DIMENSION,
        "pipeline": PIPELINE_VERSION,
        "metric": "cosine",
    }

