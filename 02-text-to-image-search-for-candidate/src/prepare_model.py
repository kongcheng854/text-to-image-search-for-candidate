"""Download the official OpenAI checkpoint and convert it for offline CLIP use.

This route works without contacting Hugging Face. The original checkpoint is
verified against the SHA256 embedded in OpenAI's published download URL.
"""

import argparse
import base64
import gzip
import json
import os
import time
import urllib.request
from pathlib import Path

from .config import ROOT
from .dataset import file_sha256

WEIGHTS_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
WEIGHTS_URL = f"https://openaipublic.azureedge.net/clip/models/{WEIGHTS_SHA256}/ViT-B-32.pt"
BPE_URL = "https://api.github.com/repos/openai/CLIP/git/blobs/7b5088a527f720063f044eb928eee315f63b2fc0"
BPE_SHA256 = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
MODEL_DIR = ROOT / ".cache" / "models" / "clip-vit-base-patch32"


def download(url: str, target: Path, expected_sha256: str | None = None):
    if target.exists() and (expected_sha256 is None or file_sha256(target) == expected_sha256):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    # A failed transfer never replaces an already downloaded file.
    for attempt in range(3):
        try:
            print(f"Downloading {target.name} (attempt {attempt + 1}/3)...", flush=True)
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as handle:
                total = int(response.headers.get("Content-Length", 0))
                count, last = 0, time.monotonic()
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
                    count += len(chunk)
                    if time.monotonic() - last > 15:
                        print(f"  {count / 1024**2:.1f} / {total / 1024**2:.1f} MB", flush=True)
                        last = time.monotonic()
            if expected_sha256 and file_sha256(partial) != expected_sha256:
                raise ValueError(f"Checksum mismatch for {target.name}")
            os.replace(partial, target)
            return
        except (OSError, ValueError):
            if attempt == 2:
                raise


def bytes_to_unicode():
    # Same reversible byte alphabet used by the official OpenAI CLIP tokenizer.
    values = list(range(ord("!"), ord("~") + 1))
    values += list(range(ord("¡"), ord("¬") + 1))
    values += list(range(ord("®"), ord("ÿ") + 1))
    characters = values[:]
    extra = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            characters.append(256 + extra)
            extra += 1
    return dict(zip(values, map(chr, characters)))


def download_tokenizer(target: Path):
    if target.exists() and file_sha256(target) == BPE_SHA256:
        return
    # The immutable Git blob API avoids slow raw.githubusercontent.com transfers.
    blob = target.with_suffix(".blob.json")
    download(BPE_URL, blob)
    payload = json.loads(blob.read_text(encoding="utf-8"))
    content = base64.b64decode(payload["content"])
    import hashlib
    if hashlib.sha256(content).hexdigest() != BPE_SHA256:
        raise ValueError("Official tokenizer checksum mismatch")
    partial = target.with_suffix(".part")
    partial.write_bytes(content)
    os.replace(partial, target)


def convert_state_dict(original):
    converted = {}
    direct = {
        "token_embedding.weight": "text_model.embeddings.token_embedding.weight",
        "positional_embedding": "text_model.embeddings.position_embedding.weight",
        "ln_final.weight": "text_model.final_layer_norm.weight",
        "ln_final.bias": "text_model.final_layer_norm.bias",
        "visual.conv1.weight": "vision_model.embeddings.patch_embedding.weight",
        "visual.class_embedding": "vision_model.embeddings.class_embedding",
        "visual.positional_embedding": "vision_model.embeddings.position_embedding.weight",
        "visual.ln_pre.weight": "vision_model.pre_layrnorm.weight",
        "visual.ln_pre.bias": "vision_model.pre_layrnorm.bias",
        "visual.ln_post.weight": "vision_model.post_layernorm.weight",
        "visual.ln_post.bias": "vision_model.post_layernorm.bias",
        "logit_scale": "logit_scale",
    }
    for name, tensor in original.items():
        if name in direct:
            converted[direct[name]] = tensor
        elif name in ("text_projection", "visual.proj"):
            destination = "text_projection.weight" if name == "text_projection" else "visual_projection.weight"
            converted[destination] = tensor.t()
        elif "transformer.resblocks." in name:
            vision = name.startswith("visual.")
            suffix = name.split("transformer.resblocks.", 1)[1]
            layer, parameter = suffix.split(".", 1)
            prefix = f"{'vision_model' if vision else 'text_model'}.encoder.layers.{layer}."
            if parameter.startswith("attn.in_proj_"):
                kind = parameter.rsplit("_", 1)[1]
                for projection, value in zip(("q", "k", "v"), tensor.chunk(3, dim=0)):
                    converted[f"{prefix}self_attn.{projection}_proj.{kind}"] = value
            else:
                for old, new in (
                    ("attn.out_proj", "self_attn.out_proj"),
                    ("ln_1", "layer_norm1"), ("ln_2", "layer_norm2"),
                    ("mlp.c_fc", "mlp.fc1"), ("mlp.c_proj", "mlp.fc2"),
                ):
                    parameter = parameter.replace(old, new)
                converted[prefix + parameter] = tensor
        elif name not in ("input_resolution", "context_length", "vocab_size"):
            raise ValueError(f"Unexpected checkpoint parameter: {name}")
    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    downloads = ROOT / ".cache" / "downloads"
    weights = downloads / "ViT-B-32.pt"
    bpe = downloads / "bpe_simple_vocab_16e6.txt.gz"
    download(WEIGHTS_URL, weights, WEIGHTS_SHA256)
    download_tokenizer(bpe)
    print(f"Official checkpoint verified: {WEIGHTS_SHA256}", flush=True)
    print(f"Tokenizer SHA256: {file_sha256(bpe)}", flush=True)
    if args.download_only:
        return
    import torch
    from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel, CLIPProcessor, CLIPTokenizer

    print("Converting official CLIP weights to a local offline model...", flush=True)
    torch.set_num_threads(4)
    # PyTorch's JIT filename loader mishandles some Windows Unicode paths.
    with weights.open("rb") as handle:
        original = torch.jit.load(handle, map_location="cpu").state_dict()
    # Defaults match OpenAI ViT-B/32: 224px, 12-layer vision/text, QuickGELU, 512-d projection.
    model = CLIPModel(CLIPConfig())
    model.load_state_dict(convert_state_dict(original), strict=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR, safe_serialization=True)
    merges = gzip.decompress(bpe.read_bytes()).decode("utf-8").split("\n")[1:49152 - 256 - 2 + 1]
    alphabet = list(bytes_to_unicode().values())
    vocab = alphabet + [character + "</w>" for character in alphabet]
    vocab += ["".join(merge.split()) for merge in merges]
    vocab += ["<|startoftext|>", "<|endoftext|>"]
    (MODEL_DIR / "vocab.json").write_text(json.dumps(dict(zip(vocab, range(len(vocab))))), encoding="utf-8")
    (MODEL_DIR / "merges.txt").write_text("#version: 0.2\n" + "\n".join(merges) + "\n", encoding="utf-8")
    tokenizer = CLIPTokenizer(
        vocab_file=str(MODEL_DIR / "vocab.json"), merges_file=str(MODEL_DIR / "merges.txt"),
        model_max_length=77,
    )
    processor = CLIPProcessor(image_processor=CLIPImageProcessor(), tokenizer=tokenizer)
    processor.save_pretrained(MODEL_DIR)
    source = {
        "weights_url": WEIGHTS_URL,
        "weights_sha256": WEIGHTS_SHA256,
        "tokenizer_url": BPE_URL,
        "tokenizer_sha256": file_sha256(bpe),
        "files": {path.name: file_sha256(path) for path in MODEL_DIR.iterdir() if path.is_file() and path.name != "source.json"},
    }
    (MODEL_DIR / "source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    print(f"Offline model ready: {MODEL_DIR}", flush=True)


if __name__ == "__main__":
    main()

