import numpy as np
import pytest

from src.query import parse_query
from src.ranking import ObjectGuard, guarded_scores, photo_prompt
from src.search import SearchEngine


@pytest.mark.parametrize("query,primary", [
    ("An orange cat on a sofa.", "cat"),
    ("People riding bikes.", "bicycle"),
    ("A motorbike.", "motorcycle"),
    ("A traffic light.", "traffic light"),
    ("A cathedral in the mountains.", None),
])
def test_word_boundaries_synonyms_and_color_ambiguity(query, primary):
    assert parse_query(query).primary == primary


@pytest.mark.parametrize("query", ["No dogs.", "A cat without a dog.", "A bus or a train."])
def test_negation_and_alternatives_disable_positive_object_guards(query):
    assert parse_query(query).negated


def test_competing_object_can_only_demote_and_does_not_hard_filter():
    original = np.array([.4, .4, .4])
    positive = np.array([.3, .2, .1])
    negatives = np.array([[.2], [.4], [.2]])
    result = guarded_scores(original, positive, negatives)
    assert np.allclose(result, [.4, .3, .35])
    assert np.all(result <= original)
    assert np.array_equal(guarded_scores(original, positive, np.empty((3, 0))), original)


def test_prompt_does_not_wrap_questions_or_existing_photo_descriptions():
    assert photo_prompt("Where is the cat?") is None
    assert photo_prompt("A photo of two cats.") is None
    assert photo_prompt("A cup beside a plate.") == "a photo of a cup beside a plate."


def test_unknown_and_negated_queries_keep_original_scores_exactly():
    guard = ObjectGuard.__new__(ObjectGuard)
    vectors = np.eye(2, dtype=np.float32)
    feature = np.array([[.6, .8]], dtype=np.float32)
    for query in ("A cathedral.", "A dog without a collar.", "A hot dog.", "A bus stop.", "A bird's eye view of a city."):
        actual, original = guard.score(query, feature, vectors, encoder=None)
        assert np.array_equal(actual, vectors @ feature[0])
        assert np.array_equal(actual, original)


def test_multiobject_query_skips_competing_category_penalties():
    class Encoder:
        def encode_text(self, text):
            return np.array([[0., 1.]], dtype=np.float32)

    guard = ObjectGuard.__new__(ObjectGuard)
    vectors = np.eye(2, dtype=np.float32)
    feature = np.array([[1., 0.]], dtype=np.float32)
    actual, original = guard.score("A cat next to a dog.", feature, vectors, Encoder())
    assert np.allclose(actual, np.sqrt(.5))
    assert np.array_equal(original, [1., 0.])


def test_optional_prompt_overflow_keeps_valid_original_query():
    class Encoder:
        def encode_text(self, text):
            raise ValueError("Description is too long")

    guard = ObjectGuard.__new__(ObjectGuard)
    vectors = np.eye(2, dtype=np.float32)
    feature = np.array([[1., 0.]], dtype=np.float32)
    actual, original = guard.score("A cat next to a dog.", feature, vectors, Encoder())
    assert np.array_equal(actual, original)


def test_degenerate_prompt_average_falls_back_without_nan_scores():
    class Encoder:
        def encode_text(self, text):
            return np.array([[-1., 0.]], dtype=np.float32)

    guard = ObjectGuard.__new__(ObjectGuard)
    actual, original = guard.score("A cat next to a dog.", np.array([[1., 0.]], dtype=np.float32), np.eye(2), Encoder())
    assert np.array_equal(actual, original)
    assert np.isfinite(actual).all()


def test_unknown_ranking_mode_fails_before_loading_a_model():
    with pytest.raises(ValueError, match="ranking"):
        SearchEngine(ranking="typo")
