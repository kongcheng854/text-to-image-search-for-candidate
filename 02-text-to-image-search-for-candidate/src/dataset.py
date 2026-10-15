"""Read only the supplied image manifest; never load external annotations."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .config import DATA_DIR, MANIFEST


class DatasetError(ValueError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    path: str
    sha256: str

    def absolute_path(self, data_dir: Path = DATA_DIR) -> Path:
        return data_dir / self.path


def load_dataset(
    manifest: Path = MANIFEST, data_dir: Path = DATA_DIR, verify_images: bool = True
) -> tuple[list[ImageRecord], str]:
    records = []
    ids, paths = set(), set()
    allowed = (data_dir / "images").resolve()
    try:
        lines = manifest.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise DatasetError(f"Cannot read image manifest: {manifest}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            image_id, relative_path = row["image_id"], row["path"]
            if not isinstance(image_id, str) or not image_id.strip():
                raise ValueError("image_id must be a nonempty string")
            if not isinstance(relative_path, str):
                raise ValueError("path must be a string")
            if Path(relative_path).is_absolute():
                raise ValueError("absolute image paths are not allowed")
            absolute = (data_dir / relative_path).resolve()
            if not absolute.is_relative_to(allowed) or not absolute.is_file():
                raise ValueError("image must exist inside materials/data/images")
            canonical_path = absolute.relative_to(data_dir.resolve()).as_posix()
            if image_id in ids or canonical_path in paths:
                raise ValueError("duplicate image ID or image path")
            if verify_images:
                with Image.open(absolute) as image:
                    image.verify()
            records.append(ImageRecord(image_id, canonical_path, file_sha256(absolute)))
            ids.add(image_id)
            paths.add(canonical_path)
        except (OSError, ValueError, KeyError, TypeError, UnidentifiedImageError) as exc:
            raise DatasetError(f"Invalid manifest entry at line {line_number}: {exc}") from exc
    if not records:
        raise DatasetError("The image manifest is empty.")
    # Explicit ordering also makes ties and matrix-to-image mapping reproducible.
    records.sort(key=lambda record: record.image_id)
    return records, json_digest([asdict(record) for record in records])


def main() -> None:
    records, fingerprint = load_dataset()
    print(json.dumps({"images": len(records), "dataset_sha256": fingerprint}, indent=2))


if __name__ == "__main__":
    main()

