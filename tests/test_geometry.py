import pytest

from lib.geometry import (
    UnionFind,
    compute_iou,
    denormalize_boxes,
    denormalize_points,
    max_iou_with_boxes,
    point_inside_box,
)


def test_iou_identical_boxes() -> None:
    """Identical boxes have IoU of exactly 1.0."""
    box = [0, 0, 10, 10]

    assert compute_iou(box, box) == pytest.approx(1.0)


def test_iou_no_overlap() -> None:
    """Disjoint boxes have IoU of 0.0."""
    result = compute_iou([0, 0, 10, 10], [20, 20, 30, 30])

    assert result == pytest.approx(0.0)


def test_iou_partial_overlap() -> None:
    """Two partially overlapping boxes produce expected IoU."""
    result = compute_iou([0, 0, 10, 10], [5, 5, 15, 15])

    assert result == pytest.approx(25.0 / 175.0)


def test_iou_contained_box() -> None:
    """A small box inside a large box produces area_small / area_large."""
    result = compute_iou([0, 0, 10, 10], [2, 2, 6, 6])

    assert result == pytest.approx(16.0 / 100.0)


def test_denormalize_boxes() -> None:
    """Center-format normalized box converts to correct pixel coords."""
    result = denormalize_boxes(
        normalized_boxes=[[0.5, 0.5, 0.5, 0.5]],
        width=100,
        height=200,
    )

    assert result == [[25.0, 50.0, 75.0, 150.0]]


def test_denormalize_points() -> None:
    """Normalized 0-1 point converts to correct pixel coords."""
    result = denormalize_points(
        normalized_points=[[0.5, 0.5]],
        width=100,
        height=200,
    )

    assert result == [[50.0, 100.0]]


def test_point_inside_box_true() -> None:
    """Point at center of box is detected as inside."""
    result = point_inside_box([0, 0, 10, 10], [[5, 5]])

    assert result is True


def test_point_inside_box_false() -> None:
    """Point outside box is detected as outside."""
    result = point_inside_box([0, 0, 10, 10], [[15, 15]])

    assert result is False


def test_point_inside_box_empty_points() -> None:
    """Empty points list returns False."""
    result = point_inside_box([0, 0, 10, 10], [])

    assert result is False


def test_union_find_basic() -> None:
    """Two separate unions create two distinct groups."""
    uf = UnionFind(4)
    uf.union(0, 1)
    uf.union(2, 3)

    assert uf.find(0) == uf.find(1)
    assert uf.find(2) == uf.find(3)
    assert uf.find(0) != uf.find(2)


def test_union_find_transitive() -> None:
    """Chained unions merge all elements into one group."""
    uf = UnionFind(3)
    uf.union(0, 1)
    uf.union(1, 2)

    roots = [uf.find(0), uf.find(1), uf.find(2)]

    assert roots[0] == roots[1] == roots[2]


def test_max_iou_with_boxes() -> None:
    """Returns highest IoU across all prompt boxes."""
    detection_box = [2, 2, 8, 8]
    prompt_boxes = [[0, 0, 6, 6], [10, 10, 20, 20]]
    expected = compute_iou(detection_box, prompt_boxes[0])

    result = max_iou_with_boxes(detection_box, prompt_boxes)

    assert result == pytest.approx(expected)
