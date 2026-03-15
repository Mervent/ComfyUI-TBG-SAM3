from typing import TypeAlias

import torch

from .geometry import (
    denormalize_boxes,
    denormalize_points,
    max_iou_with_boxes,
    point_inside_box,
)
from .mask_ops import normalize_masks
from .prompt_handler import valid_block

_Triplet: TypeAlias = tuple[
    torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
]


def _index(tensor: torch.Tensor | None, indices: torch.Tensor) -> torch.Tensor | None:
    """Index a tensor if non-None, else return None."""
    if tensor is None:
        return None
    return tensor[indices]


def filter_by_size(
    masks: torch.Tensor | None,
    boxes: torch.Tensor | None,
    scores: torch.Tensor | None,
    min_size: int,
) -> _Triplet:
    """Remove masks whose foreground area is below min_size^2 pixels."""
    if masks is not None and masks.numel() > 0 and min_size > 1:
        min_area = float(min_size * min_size)
        masks_flat = normalize_masks(masks)
        binary = (masks_flat > 0.5).float()
        areas = binary.view(binary.shape[0], -1).sum(dim=1)

        keep_indices = (areas >= min_area).nonzero(as_tuple=False).view(-1)
        if keep_indices.numel() > 0:
            masks = masks[keep_indices]
            boxes = _index(boxes, keep_indices)
            scores = _index(scores, keep_indices)
        else:
            return None, None, None

    return masks, boxes, scores


def filter_by_density(
    masks: torch.Tensor | None,
    boxes: torch.Tensor | None,
    scores: torch.Tensor | None,
    min_density: float,
) -> _Triplet:
    """Remove masks where foreground_pixels / bbox_area is below min_density."""
    if min_density > 0.0 and masks is not None and masks.numel() > 0:
        masks_flat = normalize_masks(masks)
        kept: list[int] = []

        for i in range(masks_flat.shape[0]):
            ys, xs = torch.where(masks_flat[i] > 0.5)
            if len(ys) == 0:
                continue
            fg = len(ys)
            bbox_area = (xs.max() - xs.min() + 1).item() * (
                ys.max() - ys.min() + 1
            ).item()
            density = fg / bbox_area
            if density >= min_density:
                kept.append(i)

        if kept:
            keep_indices = torch.tensor(kept, dtype=torch.long, device=masks.device)
            masks = masks[keep_indices]
            boxes = _index(boxes, keep_indices)
            scores = _index(scores, keep_indices)
        else:
            return None, None, None

    return masks, boxes, scores


def filter_by_instances(
    masks: torch.Tensor | None,
    boxes: torch.Tensor | None,
    scores: torch.Tensor | None,
    positive_boxes: dict | None,
    positive_points: dict | None,
    width: int,
    height: int,
    iou_threshold: float = 0.1,
) -> _Triplet:
    """Keep only detections that overlap a positive box or contain a positive point."""
    if boxes is None:
        return masks, boxes, scores

    boxes_cpu = boxes.detach().cpu()

    prompt_boxes: list[list[float]] = []
    if valid_block(positive_boxes, "boxes"):
        prompt_boxes = denormalize_boxes(
            normalized_boxes=positive_boxes["boxes"],
            width=width,
            height=height,
        )

    prompt_points: list[list[float]] = []
    if valid_block(positive_points, "points"):
        prompt_points = denormalize_points(
            normalized_points=positive_points["points"],
            width=width,
            height=height,
        )

    kept: list[int] = []
    for i, det_box in enumerate(boxes_cpu):
        box = det_box.tolist()
        best_iou = max_iou_with_boxes(box, prompt_boxes) if prompt_boxes else 0.0
        has_point = point_inside_box(box, prompt_points) if prompt_points else False

        match = (prompt_boxes and best_iou >= iou_threshold) or (
            prompt_points and has_point
        )
        if match:
            kept.append(i)

    if kept:
        keep_indices = torch.tensor(kept, dtype=torch.long, device=boxes.device)
        masks = _index(masks, keep_indices)
        boxes = boxes[keep_indices]
        scores = _index(scores, keep_indices)
        return masks, boxes, scores

    return None, None, None


def limit_detections(
    masks: torch.Tensor | None,
    boxes: torch.Tensor | None,
    scores: torch.Tensor | None,
    max_detections: int,
) -> _Triplet:
    """Keep only the top-N detections by score. Negative max means no limit."""
    if masks is not None and max_detections > 0 and len(masks) > max_detections:
        if scores is not None:
            top_indices = torch.argsort(scores, descending=True)[:max_detections]
            masks = masks[top_indices]
            boxes = _index(boxes, top_indices)
            scores = _index(scores, top_indices)

    return masks, boxes, scores
