from lib.segs_builder import (
    build_combined_segs,
    build_contour_segs,
    build_detection_segs,
    build_overlapping_segs,
    make_label,
)


def test_make_label_with_prompt() -> None:
    """Prompt text is prefixed to the index."""
    assert make_label("cat", index=0) == "cat_0"


def test_make_label_without_prompt() -> None:
    """Empty prompt falls back to 'detection_N'."""
    assert make_label("", index=0) == "detection_0"


def test_build_contour_segs_format(masks_3d) -> None:
    """Returns ((H,W), [SEG, ...]) with valid SEG fields."""
    result = build_contour_segs(
        masks=masks_3d,
        text_prompt="cat",
        width=20,
        height=20,
        crop_factor=1.0,
    )

    assert isinstance(result, tuple)
    assert result[0] == (20, 20)
    assert isinstance(result[1], list)

    for seg in result[1]:
        assert seg.cropped_mask is not None
        assert len(seg.crop_region) == 4
        assert isinstance(seg.label, str)


def test_build_contour_segs_empty_masks() -> None:
    """None masks produce empty seg list."""
    result = build_contour_segs(
        masks=None,
        text_prompt="",
        width=20,
        height=20,
        crop_factor=1.0,
    )

    assert result == ((20, 20), [])


def test_build_detection_segs_one_per_mask(masks_3d) -> None:
    """Each detection mask produces exactly one SEG with correct label."""
    result = build_detection_segs(
        masks=masks_3d,
        text_prompt="cat",
        width=20,
        height=20,
        crop_factor=1.5,
    )

    assert result[0] == (20, 20)
    assert len(result[1]) == masks_3d.shape[0]

    for seg in result[1]:
        assert seg.cropped_mask is not None
        assert "cat" in seg.label


def test_build_combined_segs_single_seg(masks_3d) -> None:
    """Combined mask produces a single SEG labelled 'combined'."""
    combined = (masks_3d > 0.5).any(dim=0, keepdim=True).float()

    result = build_combined_segs(
        combined_tensor=combined,
        text_prompt="",
        width=20,
        height=20,
        crop_factor=1.5,
    )

    assert result[0] == (20, 20)
    assert len(result[1]) == 1
    assert result[1][0].label == "combined"
    assert result[1][0].cropped_mask is not None


def test_build_overlapping_segs_groups_overlapping(masks_3d) -> None:
    """Two overlapping masks merge into one group; disjoint mask stays separate."""
    masks = masks_3d.clone()
    masks[:, :, :] = 0.0
    masks[0, 2:10, 2:10] = 1.0
    masks[1, 6:14, 6:14] = 1.0
    masks[2, 15:19, 15:19] = 1.0

    result = build_overlapping_segs(
        masks=masks,
        text_prompt="cat",
        width=20,
        height=20,
        crop_factor=1.5,
    )

    assert result[0] == (20, 20)
    assert len(result[1]) == 2


def test_build_overlapping_segs_separates_disjoint(masks_3d) -> None:
    """Three non-overlapping masks produce three separate groups."""
    masks = masks_3d.clone()
    masks[:, :, :] = 0.0
    masks[0, 1:5, 1:5] = 1.0
    masks[1, 8:12, 8:12] = 1.0
    masks[2, 14:18, 14:18] = 1.0

    result = build_overlapping_segs(
        masks=masks,
        text_prompt="",
        width=20,
        height=20,
        crop_factor=1.5,
    )

    assert result[0] == (20, 20)
    assert len(result[1]) == 3
