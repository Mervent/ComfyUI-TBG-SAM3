import pytest

from lib.prompt_handler import aggregate_prompts, extract_by_mode, valid_block


def _sample_pipeline() -> dict:
    """Full pipeline with all four prompt types."""
    return {
        "positive_boxes": {"boxes": [[0.5, 0.5, 0.2, 0.2]], "labels": [1]},
        "negative_boxes": {"boxes": [[0.2, 0.2, 0.1, 0.1]], "labels": [0]},
        "positive_points": {"points": [[0.5, 0.5]], "labels": [1]},
        "negative_points": {"points": [[0.1, 0.1]], "labels": [0]},
    }


def test_valid_block_true() -> None:
    """Dict with a non-empty key returns True."""
    result = valid_block({"points": [[0.5, 0.5]]}, "points")

    assert result is True


def test_valid_block_false_missing_key() -> None:
    """Dict without the requested key returns False."""
    result = valid_block({"boxes": [[0.5, 0.5, 0.2, 0.2]]}, "points")

    assert result is False


def test_valid_block_false_none() -> None:
    """None input returns False."""
    result = valid_block(None, "points")

    assert result is False


def test_valid_block_false_empty_list() -> None:
    """Dict with empty list for key returns False."""
    result = valid_block({"points": []}, "points")

    assert result is False


def test_extract_by_mode_all() -> None:
    """'all' mode returns all four prompt blocks."""
    pos_b, neg_b, pos_p, neg_p = extract_by_mode(_sample_pipeline(), "all")

    assert pos_b is not None
    assert neg_b is not None
    assert pos_p is not None
    assert neg_p is not None


def test_extract_by_mode_boxes_only() -> None:
    """'boxes_only' returns boxes but not points."""
    pos_b, neg_b, pos_p, neg_p = extract_by_mode(_sample_pipeline(), "boxes_only")

    assert pos_b is not None
    assert neg_b is not None
    assert pos_p is None
    assert neg_p is None


def test_extract_by_mode_points_only() -> None:
    """'points_only' returns points but not boxes."""
    pos_b, neg_b, pos_p, neg_p = extract_by_mode(_sample_pipeline(), "points_only")

    assert pos_b is None
    assert neg_b is None
    assert pos_p is not None
    assert neg_p is not None


def test_extract_by_mode_positive_only() -> None:
    """'positive_only' returns positive boxes and points, no negatives."""
    pos_b, neg_b, pos_p, neg_p = extract_by_mode(_sample_pipeline(), "positive_only")

    assert pos_b is not None
    assert neg_b is None
    assert pos_p is not None
    assert neg_p is None


def test_extract_by_mode_negative_only() -> None:
    """'negative_only' returns negative boxes and points, no positives."""
    pos_b, neg_b, pos_p, neg_p = extract_by_mode(_sample_pipeline(), "negative_only")

    assert pos_b is None
    assert neg_b is not None
    assert pos_p is None
    assert neg_p is not None


def test_extract_by_mode_disabled() -> None:
    """'disabled' mode returns all None regardless of pipeline content."""
    pos_b, neg_b, pos_p, neg_p = extract_by_mode(_sample_pipeline(), "disabled")

    assert pos_b is None
    assert neg_b is None
    assert pos_p is None
    assert neg_p is None


def test_extract_by_mode_invalid_pipe() -> None:
    """Non-dict pipeline raises ValueError."""
    with pytest.raises(ValueError):
        extract_by_mode([], "all")


def test_aggregate_prompts_both() -> None:
    """Positive and negative prompts merge in order with correct labels."""
    positive = {"points": [[0.5, 0.5]], "labels": [1]}
    negative = {"points": [[0.1, 0.1]], "labels": [0]}

    prompts, labels = aggregate_prompts(
        positive_block=positive,
        negative_block=negative,
        prompt_type="points",
    )

    assert prompts == [[0.5, 0.5], [0.1, 0.1]]
    assert labels == [1, 0]


def test_aggregate_prompts_positive_only() -> None:
    """Only positive block contributes when negative is None."""
    positive = {"points": [[0.5, 0.5]], "labels": [1]}

    prompts, labels = aggregate_prompts(
        positive_block=positive,
        negative_block=None,
        prompt_type="points",
    )

    assert prompts == [[0.5, 0.5]]
    assert labels == [1]


def test_aggregate_prompts_empty() -> None:
    """Both None blocks produce empty lists."""
    prompts, labels = aggregate_prompts(
        positive_block=None,
        negative_block=None,
        prompt_type="points",
    )

    assert prompts == []
    assert labels == []


def test_aggregate_prompts_invalid_type() -> None:
    """Invalid prompt_type raises ValueError."""
    with pytest.raises(ValueError):
        aggregate_prompts(
            positive_block={"points": [[0.5, 0.5]], "labels": [1]},
            negative_block={"points": [[0.1, 0.1]], "labels": [0]},
            prompt_type="invalid",
        )
