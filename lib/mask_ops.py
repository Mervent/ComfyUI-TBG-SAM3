from collections.abc import Callable

import cv2
import numpy as np
import torch


def normalize_masks(masks: torch.Tensor, to_cpu: bool = False) -> torch.Tensor:
    """Flatten mask tensor to 3D (N, H, W) regardless of input shape."""
    if masks.dim() == 4 and masks.shape[1] == 1:
        result = masks[:, 0, :, :]
    elif masks.dim() == 3:
        result = masks
    elif masks.dim() == 4:
        result = masks.mean(dim=1)
    else:
        raise ValueError(f"[SAM3] Unexpected masks shape: {masks.shape}")

    if to_cpu:
        result = result.detach().cpu()

    return result


def restore_mask_shape(processed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    """Re-add channel dim if the original was 4D single-channel."""
    if original.dim() == 4 and original.shape[1] == 1:
        return processed.unsqueeze(1)
    return processed


def apply_per_mask(
    masks: torch.Tensor,
    fn: Callable[[np.ndarray], np.ndarray],
    device: str | None = None,
) -> torch.Tensor:
    """Apply a numpy function to each mask individually, preserving shape and device."""
    target_device = device if device is not None else masks.device
    masks_flat = normalize_masks(masks, to_cpu=True)

    processed = []
    for idx in range(masks_flat.shape[0]):
        processed.append(fn(masks_flat[idx].numpy()).astype(np.float32))

    processed_stack = torch.from_numpy(np.stack(processed, axis=0)).to(target_device)
    return restore_mask_shape(processed_stack, masks)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes in a binary mask using flood-fill from corners."""
    fg = (mask > 0.5).astype(np.uint8)
    if fg.sum() == 0:
        return fg.astype(np.float32)

    ys, xs = np.where(fg == 1)
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    crop_fg = fg[y1 : y2 + 1, x1 : x2 + 1]
    ch, cw = crop_fg.shape

    inv = 1 - crop_fg
    inv_ff = inv.copy()
    mask_ff = np.zeros((ch + 2, cw + 2), np.uint8)

    cv2.floodFill(inv_ff, mask_ff, (0, 0), 2)
    cv2.floodFill(inv_ff, mask_ff, (cw - 1, 0), 2)
    cv2.floodFill(inv_ff, mask_ff, (0, ch - 1), 2)
    cv2.floodFill(inv_ff, mask_ff, (cw - 1, ch - 1), 2)

    outer_bg = (inv_ff == 2).astype(np.uint8)
    holes = inv - outer_bg
    holes[holes < 0] = 0

    filled_crop = crop_fg + holes
    filled_crop = np.clip(filled_crop, 0, 1).astype(np.uint8)

    filled_full = fg.copy()
    filled_full[y1 : y2 + 1, x1 : x2 + 1] = filled_crop
    return filled_full.astype(np.float32)


def dilate_erode(mask: np.ndarray, dilation: int) -> np.ndarray:
    """Dilate (positive) or erode (negative) a binary mask. Zero is a no-op."""
    if dilation == 0:
        return mask
    binary = (mask > 0.5).astype(np.uint8)
    kernel = np.ones((abs(dilation), abs(dilation)), np.uint8)
    if dilation > 0:
        binary = cv2.dilate(binary, kernel, iterations=1)
    else:
        binary = cv2.erode(binary, kernel, iterations=1)
    return binary.astype(np.float32)


def build_combined_mask(masks: torch.Tensor) -> torch.Tensor:
    """OR-merge all masks into a single (1, H, W) binary mask."""
    masks_flat = normalize_masks(masks)
    return (masks_flat > 0.5).any(dim=0, keepdim=True).float()
