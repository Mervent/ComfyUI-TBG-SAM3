import numpy as np
import pytest
import torch

from lib.mask_ops import (
    apply_per_mask,
    build_combined_mask,
    dilate_erode,
    fill_holes,
    normalize_masks,
    restore_mask_shape,
)


def test_normalize_masks_3d_passthrough(masks_3d) -> None:
    """3D tensor passes through unchanged."""
    result = normalize_masks(masks_3d)

    assert result.shape == masks_3d.shape
    assert torch.equal(result, masks_3d)


def test_normalize_masks_4d_squeeze(masks_4d) -> None:
    """4D single-channel tensor squeezes to 3D."""
    result = normalize_masks(masks_4d)

    assert result.shape == (masks_4d.shape[0], masks_4d.shape[2], masks_4d.shape[3])


def test_normalize_masks_4d_multichannel() -> None:
    """4D multi-channel tensor averages across channels."""
    masks = torch.zeros((2, 3, 4, 4), dtype=torch.float32)
    masks[:, 0, :, :] = 1.0

    result = normalize_masks(masks)

    assert result.shape == (2, 4, 4)
    assert torch.allclose(result, torch.full((2, 4, 4), 1.0 / 3.0))


def test_normalize_masks_to_cpu(masks_3d) -> None:
    """to_cpu=True detaches and moves to CPU."""
    result = normalize_masks(masks_3d, to_cpu=True)

    assert result.device.type == "cpu"


def test_normalize_masks_invalid_shape() -> None:
    """2D tensor raises ValueError."""
    with pytest.raises(ValueError):
        normalize_masks(torch.zeros(10, 10))


def test_restore_mask_shape_4d(masks_4d) -> None:
    """Restores channel dim for 4D originals."""
    processed = normalize_masks(masks_4d)
    restored = restore_mask_shape(processed, masks_4d)

    assert restored.shape == masks_4d.shape


def test_restore_mask_shape_3d(masks_3d) -> None:
    """3D originals are returned unchanged."""
    processed = normalize_masks(masks_3d)
    restored = restore_mask_shape(processed, masks_3d)

    assert restored.shape == masks_3d.shape
    assert torch.equal(restored, processed)


def test_fill_holes_fills_internal(mask_with_hole_np) -> None:
    """Interior hole pixels become foreground."""
    result = fill_holes(mask_with_hole_np)

    assert result[9:11, 9:11].min() == 1.0


def test_fill_holes_preserves_background(mask_with_hole_np) -> None:
    """Background corners stay zero after filling."""
    result = fill_holes(mask_with_hole_np)

    assert result[0, 0] == 0.0
    assert result[-1, -1] == 0.0


def test_fill_holes_empty_mask() -> None:
    """All-zero mask returns all-zero."""
    mask = np.zeros((20, 20), dtype=np.float32)
    result = fill_holes(mask)

    assert np.array_equal(result, mask)


def test_dilate_erode_zero_noop() -> None:
    """dilation=0 returns binarized mask unchanged."""
    mask = np.zeros((20, 20), dtype=np.float32)
    result = dilate_erode(mask, dilation=0)

    assert np.array_equal(result, mask)


def test_dilate_expands(square_mask_np) -> None:
    """Positive dilation increases foreground area."""
    original_area = int((square_mask_np > 0.5).sum())
    result = dilate_erode(square_mask_np, dilation=3)

    assert int((result > 0.5).sum()) > original_area


def test_erode_shrinks() -> None:
    """Negative dilation decreases foreground area."""
    mask = np.zeros((20, 20), dtype=np.float32)
    mask[4:16, 4:16] = 1.0
    original_area = int((mask > 0.5).sum())

    result = dilate_erode(mask, dilation=-3)

    assert int((result > 0.5).sum()) < original_area


def test_apply_per_mask_processes_all(masks_3d) -> None:
    """Function is applied to every mask in the batch."""

    def invert(m: np.ndarray) -> np.ndarray:
        return (1.0 - (m > 0.5).astype(np.float32)).astype(np.float32)

    result = apply_per_mask(masks_3d, invert)

    assert result.shape == masks_3d.shape

    for i in range(3):
        expected = torch.from_numpy(invert(masks_3d[i].numpy()))
        assert torch.equal(result[i], expected)


def test_apply_per_mask_preserves_shape_4d(masks_4d) -> None:
    """4D input shape is preserved after per-mask processing."""

    def passthrough(m: np.ndarray) -> np.ndarray:
        return (m > 0.5).astype(np.float32)

    result = apply_per_mask(masks_4d, passthrough)

    assert result.shape == masks_4d.shape


def test_build_combined_mask_union(masks_3d) -> None:
    """Combined mask is the logical OR of all individual masks."""
    expected = (masks_3d > 0.5).any(dim=0, keepdim=True).float()
    result = build_combined_mask(masks_3d)

    assert result.shape == expected.shape
    assert torch.equal(result, expected)
