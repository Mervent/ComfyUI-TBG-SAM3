import torch

from lib.types import empty_masks, empty_segmentation_result, empty_segs


def test_empty_masks_shape() -> None:
    """Returns a (1, H, W) zero tensor."""
    result = empty_masks(height=20, width=30)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 20, 30)
    assert torch.count_nonzero(result).item() == 0


def test_empty_masks_device() -> None:
    """Respects the device parameter."""
    result = empty_masks(height=10, width=12, device="cpu")

    assert result.device.type == "cpu"


def test_empty_segs_format() -> None:
    """Returns ((H, W), []) tuple."""
    result = empty_segs(height=20, width=30)
    assert result == ((20, 30), [])


def test_empty_segmentation_result_length() -> None:
    """Returns a 9-element tuple matching TBGSam3Segmentation outputs."""
    pil_image = object()
    result = empty_segmentation_result(
        height=20,
        width=30,
        vis_image_fn=lambda img: img,
        pil_image=pil_image,
    )

    assert isinstance(result, tuple)
    assert len(result) == 9


def test_empty_segmentation_result_masks_are_zero() -> None:
    """The mask outputs (index 0 and 5) are zero tensors of correct shape."""
    pil_image = object()
    result = empty_segmentation_result(
        height=20,
        width=30,
        vis_image_fn=lambda img: img,
        pil_image=pil_image,
    )

    first_mask = result[0]
    sixth_mask = result[5]

    assert first_mask.shape == (1, 20, 30)
    assert sixth_mask.shape == (1, 20, 30)
    assert torch.count_nonzero(first_mask).item() == 0
    assert torch.count_nonzero(sixth_mask).item() == 0
