"""Search from the command line: python -m src.search 'A dog on a beach.'"""

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np

from .dataset import load_dataset
from .encoder import ClipEncoder
from .index import IndexError, read_index
from .ranking import ObjectGuard, ranking_signature


@dataclass(frozen=True)
class SearchResult:
    rank: int
    image_id: str
    path: str
    score: float
    clip_score: float | None = None


def rank_scores(scores: np.ndarray, ids, top_k: int):
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= len(ids):
        raise ValueError(f"top_k must be an integer between 1 and {len(ids)}")
    if scores.shape != (len(ids),) or not np.isfinite(scores).all():
        raise ValueError("Expected one finite ranking score per image.")
    return np.lexsort((np.asarray(ids), -scores))[:top_k]


def rank_vectors(vectors: np.ndarray, query: np.ndarray, ids, top_k: int):
    if vectors.ndim != 2 or query.shape != (1, vectors.shape[1]) or len(ids) != len(vectors):
        raise ValueError("Query and index dimensions do not agree.")
    if not np.isfinite(query).all():
        raise ValueError("Query contains non-finite values.")
    scores = vectors @ query[0]
    # Primary key: descending score; secondary key: ascending image ID.
    order = rank_scores(scores, ids, top_k)
    return order, scores


class SearchEngine:
    def __init__(self, offline: bool = False, ranking: str = "optimized"):
        if ranking not in ("baseline", "optimized"):
            raise ValueError("ranking must be baseline or optimized")
        self.vectors, self.records, self.metadata = read_index()
        current_records, fingerprint = load_dataset()
        if fingerprint != self.metadata["dataset_sha256"] or current_records != self.records:
            raise IndexError("The image dataset changed. Run: python -m src.build_index")
        self.encoder = ClipEncoder(offline=offline)
        self.ids = [record.image_id for record in self.records]
        self.id_array = np.asarray(self.ids)
        self.ranking = ranking
        self.guard = ObjectGuard(self.encoder, self.vectors) if ranking == "optimized" else None
        self.ranking_metadata = ranking_signature() if ranking == "optimized" else {"version": "whole-sentence-cosine-v1"}

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        feature = self.encoder.encode_text(query)
        clip_scores = self.vectors @ feature[0]
        if self.guard is not None:
            scores, clip_scores = self.guard.score(query, feature, self.vectors, self.encoder, clip_scores)
        else:
            scores = clip_scores
        order = rank_scores(scores, self.id_array, top_k)
        return [
            SearchResult(
                rank=rank,
                image_id=self.records[index].image_id,
                path=f"materials/data/{self.records[index].path}",
                score=float(scores[index]),
                clip_score=float(clip_scores[index]),
            )
            for rank, index in enumerate(order, 1)
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--ranking", choices=("baseline", "optimized"), default="optimized")
    parser.add_argument("--output", help="Optional JSON output file")
    args = parser.parse_args()
    engine = SearchEngine(offline=args.offline, ranking=args.ranking)
    payload = {
        "query": args.query,
        "model": engine.metadata["signature"],
        "ranking": engine.ranking_metadata,
        "results": [asdict(result) for result in engine.search(args.query, args.top_k)],
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

