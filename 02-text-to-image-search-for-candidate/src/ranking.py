"""General text-side reranking; independent of evaluation queries and image IDs."""

import hashlib
import json
import re

import numpy as np

from .query import CONFUSIONS, OBJECTS, object_prompt, parse_query

RANKING_VERSION = "object-guard-multiobject-prompt-v1"
CONTRAST_WEIGHT = 0.5
PROMPT_WEIGHT = 0.5


def ranking_signature() -> dict:
    vocabulary = json.dumps({"objects": OBJECTS, "confusions": CONFUSIONS}, sort_keys=True)
    return {
        "version": RANKING_VERSION,
        "contrast_weight": CONTRAST_WEIGHT,
        "prompt_weight": PROMPT_WEIGHT,
        "vocabulary_sha256": hashlib.sha256(vocabulary.encode()).hexdigest(),
        "multiobject_policy": "average normalized original/photo text features; no class penalty",
        "single_object_policy": "subtract positive part of competing-object advantage",
        "fallback": "original whole-sentence cosine for negation, alternatives or unknown objects",
    }


def photo_prompt(query: str) -> str | None:
    query = query.strip()
    if query.endswith("?") or re.match(r"(?i)^(please|find|show|search|what|where|which|how|can|could)\b", query):
        return None
    if re.match(r"(?i)^(a |an |the )?(photo|photograph|picture|image)\b", query):
        return None
    return "a photo of " + query[0].lower() + query[1:]


def guarded_scores(full_scores, positive_scores, negative_scores, weight=CONTRAST_WEIGHT):
    """A matching negative category only demotes; it never creates a hard filter."""
    if negative_scores.shape[1] == 0:
        return full_scores.copy()
    advantage = np.maximum(0.0, negative_scores.max(axis=1) - positive_scores)
    return full_scores - weight * advantage


class ObjectGuard:
    def __init__(self, encoder, image_vectors):
        self.vocabulary = sorted(OBJECTS)
        self.positions = {name: index for index, name in enumerate(self.vocabulary)}
        prototypes = encoder.encode_texts([object_prompt(name) for name in self.vocabulary])
        self.object_scores = image_vectors @ prototypes.T
        self.object_scores.flags.writeable = False

    def score(self, query, original_feature, image_vectors, encoder, raw_scores=None):
        raw = image_vectors @ original_feature[0] if raw_scores is None else raw_scores
        plan = parse_query(query)
        if plan.negated or plan.ambiguous or plan.primary is None:
            return raw, raw
        objects = [name for name in plan.objects if name != "person"]
        if len(objects) > 1:
            prompt = photo_prompt(query)
            if prompt:
                try:
                    expanded = encoder.encode_text(prompt)
                except ValueError:
                    # A valid 77-token description must not fail merely because
                    # the optional prefix exceeds the model's context length.
                    return raw, raw
                feature = (1 - PROMPT_WEIGHT) * original_feature + PROMPT_WEIGHT * expanded
                norm = np.linalg.norm(feature, axis=1, keepdims=True)
                if not np.isfinite(norm).all() or (norm < 1e-12).any():
                    return raw, raw
                feature = feature / norm
                return image_vectors @ feature[0], raw
            return raw, raw
        competitors = [name for name in CONFUSIONS.get(plan.primary, ()) if name not in plan.objects]
        positive = self.object_scores[:, self.positions[plan.primary]]
        negative = self.object_scores[:, [self.positions[name] for name in competitors]]
        return guarded_scores(raw, positive, negative), raw
