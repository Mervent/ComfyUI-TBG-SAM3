import torch

from lib.mask_filters import (
    filter_by_density,
    filter_by_instances,
    filter_by_size,
    limit_detections,
)


def test_filter_by_size_keeps_large(masks_3d, sample_boxes, sample_scores) -> None:
    """Masks above min_size^2 area are kept."""
    out_masks, out_boxes, out_scores = filter_by_size(
        masks=masks_3d,
        boxes=sample_boxes,
        scores=sample_scores,
        min_size=3,
    )

    assert out_masks is not None
    assert out_masks.shape[0] >= 1
    assert out_boxes.shape[0] == out_masks.shape[0]
    assert out_scores.shape[0] == out_masks.shape[0]


def test_filter_by_size_removes_small(sample_boxes, sample_scores) -> None:
    """Single-pixel mask is removed, 8x8 block is kept."""
    masks = torch.zeros((2, 20, 20), dtype=torch.float32)
    masks[0, 10, 10] = 1.0
    masks[1, 2:10, 2:10] = 1.0

    out_masks, out_boxes, out_scores = filter_by_size(
        masks=masks,
        boxes=sample_boxes[:2],
        scores=sample_scores[:2],
        min_size=4,
    )

    assert out_masks is not None
    assert out_masks.shape[0] == 1


def test_filter_by_size_all_removed(sample_boxes, sample_scores) -> None:
    """Returns None triplet when all masks are too small."""
    masks = torch.zeros((2, 20, 20), dtype=torch.float32)
    masks[0, 1, 1] = 1.0
    masks[1, 2, 2] = 1.0

    out_masks, out_boxes, out_scores = filter_by_size(
        masks=masks,
        boxes=sample_boxes[:2],
        scores=sample_scores[:2],
        min_size=5,
    )

    assert out_masks is None
    assert out_boxes is None
    assert out_scores is None


def test_filter_by_size_min_size_1(masks_3d, sample_boxes, sample_scores) -> None:
    """min_size=1 is a no-op, all masks pass."""
    out_masks, out_boxes, out_scores = filter_by_size(
        masks=masks_3d,
        boxes=sample_boxes,
        scores=sample_scores,
        min_size=1,
    )

    assert torch.equal(out_masks, masks_3d)


def test_filter_by_density_keeps_dense(sample_boxes, sample_scores) -> None:
    """Dense 12x12 block (density=1.0) passes 0.8 threshold."""
    masks = torch.zeros((1, 20, 20), dtype=torch.float32)
    masks[0, 4:16, 4:16] = 1.0

    out_masks, out_boxes, out_scores = filter_by_density(
        masks=masks,
        boxes=sample_boxes[:1],
        scores=sample_scores[:1],
        min_density=0.8,
    )

    assert out_masks is not None
    assert out_masks.shape[0] == 1


def test_filter_by_density_removes_sparse(sample_boxes, sample_scores) -> None:
    """Four scattered pixels in a 10x10 bbox are too sparse for 0.1 threshold."""
    masks = torch.zeros((1, 20, 20), dtype=torch.float32)
    masks[0, 5, 5] = 1.0
    masks[0, 5, 14] = 1.0
    masks[0, 14, 5] = 1.0
    masks[0, 14, 14] = 1.0

    out_masks, out_boxes, out_scores = filter_by_density(
        masks=masks,
        boxes=sample_boxes[:1],
        scores=sample_scores[:1],
        min_density=0.1,
    )

    assert out_masks is None


def test_filter_by_density_disabled(masks_3d, sample_boxes, sample_scores) -> None:
    """min_density=0.0 is a no-op."""
    out_masks, out_boxes, out_scores = filter_by_density(
        masks=masks_3d,
        boxes=sample_boxes,
        scores=sample_scores,
        min_density=0.0,
    )

    assert torch.equal(out_masks, masks_3d)


def test_filter_by_instances_matching_boxes_keeps_detection(
    masks_3d, sample_boxes, sample_scores
) -> None:
    """Detections overlapping a positive box are kept."""
    out_masks, out_boxes, out_scores = filter_by_instances(
        masks=masks_3d,
        boxes=sample_boxes,
        scores=sample_scores,
        positive_boxes={"boxes": [[0.25, 0.25, 0.5, 0.5]], "labels": [1]},
        positive_points=None,
        width=20,
        height=20,
        iou_threshold=0.1,
    )

    assert out_masks is not None
    assert out_boxes is not None
    assert out_boxes.shape[0] >= 1


def test_filter_by_instances_nonmatching_boxes_removes_all(
    masks_3d, sample_boxes, sample_scores
) -> None:
    """Prompt box far from all detections removes everything."""
    out_masks, out_boxes, out_scores = filter_by_instances(
        masks=masks_3d,
        boxes=sample_boxes,
        scores=sample_scores,
        positive_boxes={"boxes": [[0.05, 0.95, 0.1, 0.1]], "labels": [1]},
        positive_points=None,
        width=20,
        height=20,
        iou_threshold=0.1,
    )

    assert out_masks is None
    assert out_boxes is None
    assert out_scores is None


def test_filter_by_instances_matching_points_keep_detection(
    masks_3d, sample_boxes, sample_scores
) -> None:
    """Detection containing a positive point is kept."""
    out_masks, out_boxes, out_scores = filter_by_instances(
        masks=masks_3d,
        boxes=sample_boxes,
        scores=sample_scores,
        positive_boxes=None,
        positive_points={"points": [[0.75, 0.75]], "labels": [1]},
        width=20,
        height=20,
    )

    assert out_masks is not None
    assert out_boxes.shape[0] >= 1


def test_filter_by_instances_nonmatching_points_removes_all(
    masks_3d, sample_boxes, sample_scores
) -> None:
    """Point outside all detection boxes removes everything."""
    out_masks, out_boxes, out_scores = filter_by_instances(
        masks=masks_3d,
        boxes=sample_boxes,
        scores=sample_scores,
        positive_boxes=None,
        positive_points={"points": [[0.95, 0.05]], "labels": [1]},
        width=20,
        height=20,
    )

    assert out_masks is None


def test_filter_by_instances_boxes_and_points_or_logic(
    masks_3d, sample_boxes, sample_scores
) -> None:
    """OR logic: detection kept if it matches box OR contains point."""
    out_masks, out_boxes, out_scores = filter_by_instances(
        masks=masks_3d,
        boxes=sample_boxes,
        scores=sample_scores,
        positive_boxes={"boxes": [[0.25, 0.25, 0.5, 0.5]], "labels": [1]},
        positive_points={"points": [[0.95, 0.95]], "labels": [1]},
        width=20,
        height=20,
        iou_threshold=0.8,
    )

    assert out_masks is not None
    assert out_boxes.shape[0] == 2


def test_filter_by_instances_no_prompts_returns_none(
    masks_3d, sample_boxes, sample_scores
) -> None:
    """No positive prompts at all results in None triplet."""
    out_masks, out_boxes, out_scores = filter_by_instances(
        masks=masks_3d,
        boxes=sample_boxes,
        scores=sample_scores,
        positive_boxes=None,
        positive_points=None,
        width=20,
        height=20,
    )

    assert out_masks is None


def test_filter_by_instances_none_boxes_input_returns_unchanged(
    masks_3d, sample_scores
) -> None:
    """When detection boxes is None, masks are returned unchanged."""
    out_masks, out_boxes, out_scores = filter_by_instances(
        masks=masks_3d,
        boxes=None,
        scores=sample_scores,
        positive_boxes={"boxes": [[0.25, 0.25, 0.5, 0.5]], "labels": [1]},
        positive_points=None,
        width=20,
        height=20,
    )

    assert torch.equal(out_masks, masks_3d)
    assert out_boxes is None


def test_limit_detections_trims(masks_3d, sample_boxes, sample_scores) -> None:
    """Keeps top-2 by score from 3 detections."""
    out_masks, out_boxes, out_scores = limit_detections(
        masks=masks_3d,
        boxes=sample_boxes,
        scores=sample_scores,
        max_detections=2,
    )

    assert out_masks.shape[0] == 2
    assert torch.allclose(out_scores, torch.tensor([0.9, 0.7]))


def test_limit_detections_no_limit(masks_3d, sample_boxes, sample_scores) -> None:
    """Negative max_detections is a no-op."""
    out_masks, out_boxes, out_scores = limit_detections(
        masks=masks_3d,
        boxes=sample_boxes,
        scores=sample_scores,
        max_detections=-1,
    )

    assert torch.equal(out_masks, masks_3d)


def test_filter_handles_none_boxes(masks_3d, sample_scores) -> None:
    """None boxes pass through filter_by_size unchanged."""
    out_masks, out_boxes, out_scores = filter_by_size(
        masks=masks_3d,
        boxes=None,
        scores=sample_scores,
        min_size=3,
    )

    assert out_masks is not None
    assert out_boxes is None
