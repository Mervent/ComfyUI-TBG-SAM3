# TBG SAM3 — Refactoring Design

## Current State

| Metric | Value | Verdict |
|--------|-------|---------|
| `nodes.py` | 1,616 lines, 6 classes | Monolithic |
| God class | `TBGSam3Segmentation` — 723 lines, 13 responsibilities | Critical |
| God method | `segment()` — 560 lines | Critical |
| Duplication | 80% between `Segmentation` and `SegmentationBatch` | Critical |
| Mask normalize pattern | Repeated **8 times** | DRY violation |
| Distinct duplicated patterns | **14 patterns** cataloged | DRY violation |
| Tests | Zero | No safety net |
| Testability score | 15/100 | Untestable |

### What `segment()` Does Today (13 responsibilities in one method)

```
1. Pipeline mode selection        (39 lines)   — prompt routing
2. Model setup                    (13 lines)   — device, processor init
3. Text prompt application        (3 lines)
4. Box prompt aggregation         (11 lines)
5. Point prompt aggregation       (12 lines)
6. Mask prompt application        (6 lines)
7. Min-size filtering             (31 lines)   — mask normalization + area check
8. Min-density filtering          (34 lines)   — mask normalization + density check
9. Instance filtering             (82 lines)   — IoU, point-in-box, box denormalization
10. Max-detections limiting       (6 lines)
11. Fill holes                    (70 lines)   — mask normalization + per-mask numpy loop
12. Dilation/erosion              (28 lines)   — mask normalization + per-mask numpy loop
13. Output building               (160 lines)  — 9 return values, SEGS conversion, union-find
```

### Duplicated Patterns Catalog

| # | Pattern | Occurrences | Impact |
|---|---------|-------------|--------|
| 1 | Mask normalization `dim==4` → `[N,H,W]` | 8 | ~56 lines wasted |
| 2 | Per-mask numpy loop (iterate, convert, process, collect) | 3 | ~90 lines wasted |
| 3 | Restore original mask shape (unsqueeze) | 3 | ~12 lines wasted |
| 4 | Empty return 9-tuple | 4 | ~20 lines wasted |
| 5 | `make_2d_mask` + `mask_to_segs` conversion | 5 | ~50 lines wasted |
| 6 | `_valid_block` helper (defined twice as nested fn) | 2 | Fragile |
| 7 | `_build_segs` method (near-identical in both classes) | 2 | ~60 lines wasted |
| 8 | Pipeline mode branching | 2 | ~34 lines wasted |
| 9 | Prompt aggregation (boxes + points) | 2 | ~30 lines wasted |
| 10 | Box denormalization (center,w,h → x1,y1,x2,y2) | 2 | ~14 lines wasted |
| 11 | Point denormalization | 2 | ~8 lines wasted |
| 12 | IoU calculation | 2 | ~30 lines wasted |
| 13 | Point-inside-box check | 2 | ~8 lines wasted |
| 14 | Flood fill from 4 corners | 2 | ~8 lines wasted |

---

## ComfyUI Constraints

What **must** stay on the node class:

- `INPUT_TYPES` classmethod
- `RETURN_TYPES`, `RETURN_NAMES`, `FUNCTION`, `CATEGORY`
- The method named by `FUNCTION` (can be a one-liner delegating to `lib/`)
- `IS_CHANGED`, `VALIDATE_INPUTS` if needed

What **can** be extracted:

- All ML logic, filtering, morphology, SEGS construction → `lib/`
- Model loading/caching → already in `model_manager.py`
- Node classes can live in separate files — `__init__.py` only needs `NODE_CLASS_MAPPINGS`

---

## Proposed Architecture

### Directory Structure

```
ComfyUI-TBG-SAM3/
├── __init__.py                     # Registry only — merges NODE_CLASS_MAPPINGS
├── nodes/
│   ├── __init__.py                 # Re-exports all node mappings
│   ├── segmentation.py             # TBGSam3Segmentation, TBGSam3SegmentationBatch
│   ├── model_loaders.py            # TBGLoadSAM3Model, TBGSAM3ModelLoaderAndDownloader
│   ├── prompt_collector.py         # TBGSAM3PromptCollector
│   └── depth_map.py                # TBGSAM3DepthMap
├── lib/
│   ├── __init__.py
│   ├── mask_ops.py                 # MaskProcessor: normalize, fill_holes, dilate, erode
│   ├── mask_filters.py             # MaskFilter: min_size, min_density, instance_filter, max_detections
│   ├── segs_builder.py             # SegsBuilder: all SEGS construction (detection, overlapping, combined)
│   ├── prompt_handler.py           # PromptHandler: mode selection, aggregation, denormalization
│   ├── geometry.py                 # IoU, point-in-box, box denormalization, union-find
│   └── types.py                    # Type aliases, constants, empty result factory
├── masktosegs.py                   # Unchanged — Impact Pack SEGS format
├── sam3_utils.py                   # Unchanged — tensor/image conversions
├── model_manager.py                # Unchanged — model paths/download
├── sam3_lib/                       # Unchanged — vendored Meta SAM3
├── web/                            # Unchanged — frontend JS
├── tests/
│   ├── test_mask_ops.py
│   ├── test_mask_filters.py
│   ├── test_segs_builder.py
│   ├── test_prompt_handler.py
│   ├── test_geometry.py
│   └── conftest.py                 # Fixtures: fake masks, fake boxes, etc.
└── pyproject.toml
```

### The Golden Rule

`lib/` has **zero ComfyUI imports**. Pure Python + torch + numpy + cv2.
`nodes/` are **thin adapters** — INPUT_TYPES definition + one-liner delegation to `lib/`.
`__init__.py` is a **pure registry**.

---

## Module Designs

### `lib/mask_ops.py` — MaskProcessor

Eliminates patterns #1, #2, #3, #11, #14 (mask normalization, per-mask loops, shape restoration, flood fill).

```python
"""Mask morphological operations. Zero ComfyUI imports."""

import cv2
import numpy as np
import torch


def normalize_masks(masks: torch.Tensor, to_cpu: bool = False) -> torch.Tensor:
    """Normalize masks from [N,H,W] or [N,1,H,W] or [N,C,H,W] to [N,H,W].

    Args:
        masks: Input mask tensor.
        to_cpu: If True, detach and move to CPU.

    Returns:
        [N,H,W] float tensor.
    """
    if masks.dim() == 4 and masks.shape[1] == 1:
        out = masks[:, 0, :, :]
    elif masks.dim() == 3:
        out = masks
    elif masks.dim() == 4:
        out = masks.mean(dim=1)
    else:
        raise ValueError(f"Unexpected masks shape: {masks.shape}")
    return out.detach().cpu() if to_cpu else out


def restore_mask_shape(processed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    """Restore processed [N,H,W] masks to original tensor shape."""
    if original.dim() == 4 and original.shape[1] == 1:
        return processed.unsqueeze(1)
    return processed


def apply_per_mask(masks: torch.Tensor, fn, device=None):
    """Normalize → per-mask numpy fn → restore shape.

    Args:
        masks: [N,H,W] or [N,1,H,W] tensor.
        fn: Callable(np.ndarray[H,W] float32) -> np.ndarray[H,W] float32.
        device: Target device for output. Defaults to masks.device.

    Returns:
        Tensor with same shape as input.
    """
    if device is None:
        device = masks.device
    flat = normalize_masks(masks, to_cpu=True)
    results = [fn(flat[i].numpy()) for i in range(flat.shape[0])]
    stack = torch.from_numpy(np.stack(results, axis=0)).to(device)
    return restore_mask_shape(stack, masks)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill internal holes in a single binary mask.

    Args:
        mask: [H,W] float32 array.

    Returns:
        [H,W] float32 with holes filled.
    """
    fg = (mask > 0.5).astype(np.uint8)
    if fg.sum() == 0:
        return fg.astype(np.float32)

    ys, xs = np.where(fg == 1)
    y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
    crop = fg[y1:y2 + 1, x1:x2 + 1]
    ch, cw = crop.shape

    inv = (1 - crop).copy()
    ff_mask = np.zeros((ch + 2, cw + 2), np.uint8)
    for seed in [(0, 0), (cw - 1, 0), (0, ch - 1), (cw - 1, ch - 1)]:
        cv2.floodFill(inv, ff_mask, seed, 2)

    holes = np.clip((1 - crop) - (inv == 2).astype(np.uint8), 0, 1)
    filled_crop = np.clip(crop + holes, 0, 1).astype(np.uint8)

    result = fg.copy()
    result[y1:y2 + 1, x1:x2 + 1] = filled_crop
    return result.astype(np.float32)


def dilate_erode(mask: np.ndarray, dilation: int) -> np.ndarray:
    """Dilate (positive) or erode (negative) a single binary mask.

    Args:
        mask: [H,W] float32 array.
        dilation: Kernel size. Positive=dilate, negative=erode, 0=no-op.

    Returns:
        [H,W] float32 processed mask.
    """
    if dilation == 0:
        return mask
    m = (mask > 0.5).astype(np.uint8)
    kernel = np.ones((abs(dilation), abs(dilation)), np.uint8)
    if dilation > 0:
        m = cv2.dilate(m, kernel, iterations=1)
    else:
        m = cv2.erode(m, kernel, iterations=1)
    return m.astype(np.float32)
```

**Usage in node (before vs after):**

```python
# BEFORE: 70 lines of fill_holes + 28 lines of dilation inline in segment()

# AFTER: 4 lines
if fill_holes:
    masks = apply_per_mask(masks, mask_ops.fill_holes)
if dilation != 0:
    masks = apply_per_mask(masks, lambda m: mask_ops.dilate_erode(m, dilation))
```

---

### `lib/mask_filters.py` — MaskFilter

Eliminates patterns #7 (min_size), #8 (min_density), #9-13 (instance filtering).

```python
"""Mask filtering operations. Zero ComfyUI imports."""

import torch
from .mask_ops import normalize_masks
from .geometry import compute_max_iou, point_inside_box, denormalize_boxes, denormalize_points


def filter_by_size(masks, boxes, scores, min_size):
    """Remove masks smaller than min_size². Returns filtered (masks, boxes, scores)."""
    if min_size <= 1:
        return masks, boxes, scores
    flat = normalize_masks(masks)
    binary = (flat > 0.5).float()
    areas = binary.view(binary.shape[0], -1).sum(dim=1)
    keep = (areas >= min_size * min_size).nonzero(as_tuple=False).view(-1)
    if len(keep) == 0:
        return None, None, None
    return masks[keep], _index(boxes, keep), _index(scores, keep)


def filter_by_density(masks, boxes, scores, min_density):
    """Remove masks where foreground_pixels / bbox_area < min_density."""
    if min_density <= 0.0:
        return masks, boxes, scores
    flat = normalize_masks(masks)
    keep = []
    for i in range(flat.shape[0]):
        ys, xs = torch.where(flat[i] > 0.5)
        if len(ys) == 0:
            continue
        fg = len(ys)
        bbox_area = (xs.max() - xs.min() + 1).item() * (ys.max() - ys.min() + 1).item()
        if fg / bbox_area >= min_density:
            keep.append(i)
    if not keep:
        return None, None, None
    idx = torch.tensor(keep, device=masks.device)
    return masks[idx], _index(boxes, idx), _index(scores, idx)


def filter_by_instances(masks, boxes, scores, positive_boxes, positive_points,
                        width, height, iou_threshold=0.1):
    """Keep only detections overlapping positive boxes or containing positive points."""
    # ... denormalize, IoU, point-in-box logic extracted from segment()
    pass


def limit_detections(masks, boxes, scores, max_detections):
    """Keep top-k detections by confidence score."""
    if max_detections <= 0 or len(masks) <= max_detections:
        return masks, boxes, scores
    top = torch.argsort(scores, descending=True)[:max_detections]
    return masks[top], _index(boxes, top), _index(scores, top)


def _index(tensor, indices):
    """Safe indexing that handles None tensors."""
    return tensor[indices] if tensor is not None else None
```

---

### `lib/segs_builder.py` — SegsBuilder

Eliminates patterns #5, #7 (_build_segs duplication, mask_to_segs calls).

```python
"""SEGS construction from masks. Zero ComfyUI imports."""

import numpy as np
import torch
from masktosegs import mask_to_segs, make_2d_mask
from .mask_ops import normalize_masks
from .geometry import UnionFind


def build_contour_segs(masks, text_prompt, width, height, crop_factor=1.0):
    """Per-instance SEGS via contour splitting. Replaces _build_segs()."""
    shape = (height, width)
    if masks is None or len(masks) == 0:
        return (shape, [])

    flat = normalize_masks(masks, to_cpu=True)
    seg_list = []
    for i in range(flat.shape[0]):
        label = _make_label(text_prompt, i)
        mask_2d = make_2d_mask(flat[i])
        _, segs = mask_to_segs(mask_2d, combined=False, crop_factor=crop_factor,
                               bbox_fill=False, drop_size=1, label=label,
                               crop_min_size=None, detailer_hook=None, is_contour=True)
        if segs:
            seg_list.extend(segs)
    return (shape, seg_list)


def build_detection_segs(masks, text_prompt, width, height, crop_factor):
    """1 SEG per detection, whole mask (combined=True), no contour splitting."""
    shape = (height, width)
    if masks is None or len(masks) == 0:
        return (shape, [])

    flat = normalize_masks(masks, to_cpu=True)
    seg_list = []
    for i in range(flat.shape[0]):
        label = _make_label(text_prompt, i)
        mask_2d = make_2d_mask(flat[i])
        _, segs = mask_to_segs(mask_2d, combined=True, crop_factor=crop_factor,
                               bbox_fill=False, drop_size=1, label=label)
        if segs:
            seg_list.extend(segs)
    return (shape, seg_list)


def build_combined_segs(combined_tensor, text_prompt, width, height, crop_factor):
    """Single SEG from union of all masks."""
    shape = (height, width)
    label = text_prompt.strip() or "combined"
    combined_2d = make_2d_mask(combined_tensor.detach().cpu())
    return mask_to_segs(combined_2d, combined=True, crop_factor=crop_factor,
                        bbox_fill=False, drop_size=1, label=label,
                        crop_min_size=None, detailer_hook=None, is_contour=True)


def build_overlapping_segs(masks, boxes, text_prompt, width, height, crop_factor):
    """Merge overlapping masks via union-find pixel overlap, 1 SEG per group."""
    shape = (height, width)
    if masks is None or len(masks) == 0:
        return (shape, [])

    flat = normalize_masks(masks, to_cpu=True)
    n = flat.shape[0]

    # Union-find with bbox pre-filter + pixel overlap
    uf = UnionFind(n)
    # ... existing overlap logic from segment() lines 791-849
    # ... extracted into clean methods on UnionFind

    # Group masks, merge, create SEGs
    # ...
    return (shape, seg_list)


def _make_label(text_prompt, index):
    if text_prompt and text_prompt.strip():
        return f"{text_prompt.strip()}_{index}"
    return f"detection_{index}"
```

---

### `lib/prompt_handler.py` — PromptHandler

Eliminates patterns #8, #9, #10, #6 (pipeline mode, prompt aggregation, _valid_block).

```python
"""Prompt handling for SAM3 segmentation. Zero ComfyUI imports."""


def valid_block(block, key):
    """Check if a prompt block has a valid non-empty key."""
    return isinstance(block, dict) and key in block and bool(block[key])


def extract_by_mode(pipeline_data, mode):
    """Route pipeline data based on mode selection.

    Returns:
        (positive_boxes, negative_boxes, positive_points, negative_points)
    """
    pos_boxes = pipeline_data.get("positive_boxes")
    neg_boxes = pipeline_data.get("negative_boxes")
    pos_points = pipeline_data.get("positive_points")
    neg_points = pipeline_data.get("negative_points")

    routes = {
        "all":           (pos_boxes, neg_boxes, pos_points, neg_points),
        "boxes_only":    (pos_boxes, neg_boxes, None, None),
        "points_only":   (None, None, pos_points, neg_points),
        "positive_only": (pos_boxes, None, pos_points, None),
        "negative_only": (None, neg_boxes, None, neg_points),
    }
    return routes.get(mode, (None, None, None, None))


def aggregate_prompts(positive_block, negative_block, prompt_type):
    """Merge positive + negative prompt blocks into flat lists.

    Args:
        prompt_type: "boxes" or "points"

    Returns:
        (all_prompts, all_labels) — empty lists if none.
    """
    all_prompts = []
    all_labels = []

    if valid_block(positive_block, prompt_type):
        items = positive_block[prompt_type]
        all_prompts.extend(items)
        all_labels.extend(positive_block.get("labels", [1] * len(items)))

    if valid_block(negative_block, prompt_type):
        items = negative_block[prompt_type]
        all_prompts.extend(items)
        all_labels.extend(negative_block.get("labels", [0] * len(items)))

    return all_prompts, all_labels
```

---

### `lib/geometry.py` — Geometric Utilities

Eliminates patterns #10, #11, #12, #13 (box/point denormalization, IoU, point-in-box, union-find).

```python
"""Geometric operations. Zero ComfyUI imports."""


def denormalize_boxes(normalized_boxes, width, height):
    """Convert (cx, cy, w_norm, h_norm) → (x1, y1, x2, y2) pixel coords."""
    result = []
    for cx, cy, w_norm, h_norm in normalized_boxes:
        x1 = (cx - w_norm / 2.0) * width
        y1 = (cy - h_norm / 2.0) * height
        x2 = (cx + w_norm / 2.0) * width
        y2 = (cy + h_norm / 2.0) * height
        result.append([x1, y1, x2, y2])
    return result


def denormalize_points(normalized_points, width, height):
    """Convert (px_norm, py_norm) → (px, py) pixel coords."""
    return [[px * width, py * height] for px, py in normalized_points]


def compute_iou(box_a, box_b):
    """Compute IoU between two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def max_iou_with_boxes(detection_box, prompt_boxes):
    """Compute max IoU between one detection box and a list of prompt boxes."""
    return max((compute_iou(detection_box, pb) for pb in prompt_boxes), default=0.0)


def point_inside_box(box, points):
    """Check if any point falls inside the box."""
    x1, y1, x2, y2 = box
    return any(x1 <= px <= x2 and y1 <= py <= y2 for px, py in points)


class UnionFind:
    """Union-Find for grouping overlapping masks."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
```

---

### `lib/types.py` — Shared Types & Factories

Eliminates pattern #4 (empty return tuple).

```python
"""Shared types and result factories. Zero ComfyUI imports."""

import torch


def empty_masks(height, width, device="cpu"):
    """Create an empty mask tensor."""
    return torch.zeros(1, height, width, device=device)


def empty_segs(height, width):
    """Create empty SEGS tuple."""
    return ((height, width), [])


def empty_segmentation_result(height, width, vis_image_fn, pil_image, device="cpu"):
    """Build the full 9-tuple empty return for segmentation nodes.

    Args:
        vis_image_fn: Callable(pil_image) -> tensor (e.g., pil_to_comfy_image).
    """
    mask = empty_masks(height, width, device)
    segs = empty_segs(height, width)
    return (mask, vis_image_fn(pil_image), "[]", "[]", segs, mask, segs, segs, segs)
```

---

### `nodes/segmentation.py` — Thin Node Adapter

After extraction, the god method becomes a coordinator:

```python
"""SAM3 segmentation nodes — thin ComfyUI adapters."""

from lib import mask_ops, mask_filters, segs_builder, prompt_handler, types
from sam3_utils import (
    SAM3ImageSegmenter, masks_to_comfy_mask, pil_to_comfy_image,
    visualize_masks_on_image, tensor_to_list, comfy_image_to_pil,
    offload_model_if_needed,
)
from masktosegs import make_2d_mask, mask_to_segs


class TBGSam3Segmentation:
    # ... INPUT_TYPES, RETURN_TYPES unchanged ...

    def segment(self, sam3_model, image, confidence_threshold=0.2, detect_all=True,
                pipeline_mode="all", instances=False, crop_factor=1.5, min_size=32,
                min_density=0.0, fill_holes=False, dilation=0, text_prompt="",
                sam3_selectors_pipe=None, mask_prompt=None, **kwargs):

        pil_image = comfy_image_to_pil(image)
        width, height = pil_image.size
        empty = lambda: types.empty_segmentation_result(
            height, width, pil_to_comfy_image, pil_image)

        # --- Prompt handling ---
        pos_boxes, neg_boxes, pos_points, neg_points = (
            prompt_handler.extract_prompts(sam3_selectors_pipe, pipeline_mode)
        )
        processor, state = self._run_inference(
            sam3_model, pil_image, confidence_threshold,
            text_prompt, pos_boxes, neg_boxes, pos_points, neg_points, mask_prompt
        )
        masks, boxes, scores = state.get("masks"), state.get("boxes"), state.get("scores")

        if masks is None or len(masks) == 0:
            offload_model_if_needed(sam3_model)
            return empty()

        # --- Filtering (density checked on original mask, before dilation) ---
        masks, boxes, scores = mask_filters.filter_by_size(masks, boxes, scores, min_size)
        if masks is None: return empty()

        masks, boxes, scores = mask_filters.filter_by_density(masks, boxes, scores, min_density)
        if masks is None: return empty()

        if instances:
            masks, boxes, scores = mask_filters.filter_by_instances(
                masks, boxes, scores, pos_boxes, pos_points, width, height)
            if masks is None: return empty()

        masks, boxes, scores = mask_filters.limit_detections(
            masks, boxes, scores, -1 if detect_all else kwargs.get("max_detections", 10))

        # --- Post-processing ---
        if fill_holes:
            masks = mask_ops.apply_per_mask(masks, mask_ops.fill_holes)
        if dilation != 0:
            masks = mask_ops.apply_per_mask(masks, lambda m: mask_ops.dilate_erode(m, dilation))

        # --- Build all outputs ---
        comfy_masks = masks_to_comfy_mask(masks)
        combined_tensor = mask_ops.build_combined_mask(masks)
        combined_mask = masks_to_comfy_mask(combined_tensor)
        vis_tensor = pil_to_comfy_image(
            visualize_masks_on_image(pil_image, masks, boxes, scores))

        segs = segs_builder.build_contour_segs(masks, text_prompt, width, height)
        combined_segs = segs_builder.build_combined_segs(
            combined_tensor, text_prompt, width, height, crop_factor)
        detection_segs = segs_builder.build_detection_segs(
            masks, text_prompt, width, height, crop_factor)
        overlapping_segs = segs_builder.build_overlapping_segs(
            masks, boxes, text_prompt, width, height, crop_factor)

        offload_model_if_needed(sam3_model)
        return (comfy_masks, vis_tensor,
                json.dumps(tensor_to_list(boxes) or []),
                json.dumps(tensor_to_list(scores) or []),
                segs, combined_mask, combined_segs, detection_segs, overlapping_segs)
```

**~560 lines → ~50 lines.** Each `lib/` module is independently testable.

---

### Shared Base for Segmentation & SegmentationBatch

The 80% duplication between the two classes is eliminated by extracting all logic to `lib/`. Both node classes become thin adapters calling the same `lib/` functions. No inheritance needed — composition over inheritance.

The only differences between single and batch:
- Batch iterates over images in the batch dimension
- Batch doesn't produce `combined_segs`, `detection_segs`, `overlapping_segs`
- Batch uses different `crop_factor` default

These differences are handled by the adapter, not the logic.

---

## Testability

### What Becomes Testable

| Module | Testable Without | Test Type |
|--------|-----------------|-----------|
| `lib/mask_ops.py` | SAM3, ComfyUI | Unit — synthetic numpy masks |
| `lib/mask_filters.py` | SAM3, ComfyUI | Unit — synthetic torch tensors |
| `lib/segs_builder.py` | SAM3, ComfyUI | Unit — synthetic masks → verify SEGS format |
| `lib/prompt_handler.py` | SAM3, ComfyUI, torch | Unit — pure dict manipulation |
| `lib/geometry.py` | SAM3, ComfyUI, torch | Unit — pure math |
| `lib/types.py` | SAM3, ComfyUI | Unit — factory outputs |
| `nodes/segmentation.py` | SAM3 model | Integration — mock processor |

### Example Tests

```python
# tests/test_mask_ops.py
import numpy as np
from lib.mask_ops import fill_holes, dilate_erode, normalize_masks

def test_fill_holes_fills_internal_hole():
    mask = np.zeros((10, 10), dtype=np.float32)
    mask[2:8, 2:8] = 1.0   # solid square
    mask[4:6, 4:6] = 0.0   # internal hole
    result = fill_holes(mask)
    assert result[5, 5] == 1.0  # hole filled

def test_fill_holes_preserves_external_background():
    mask = np.zeros((10, 10), dtype=np.float32)
    mask[2:8, 2:8] = 1.0
    result = fill_holes(mask)
    assert result[0, 0] == 0.0  # background preserved

def test_dilate_expands_mask():
    mask = np.zeros((20, 20), dtype=np.float32)
    mask[8:12, 8:12] = 1.0  # 4x4 square
    result = dilate_erode(mask, dilation=3)
    assert result[7, 7] == 1.0  # expanded

def test_erode_shrinks_mask():
    mask = np.zeros((20, 20), dtype=np.float32)
    mask[5:15, 5:15] = 1.0  # 10x10 square
    result = dilate_erode(mask, dilation=-3)
    assert result[5, 5] == 0.0  # eroded away
    assert result[10, 10] == 1.0  # center preserved

def test_dilation_zero_is_noop():
    mask = np.random.rand(10, 10).astype(np.float32)
    result = dilate_erode(mask, dilation=0)
    np.testing.assert_array_equal(result, mask)


# tests/test_geometry.py
from lib.geometry import compute_iou, point_inside_box, UnionFind

def test_iou_identical_boxes():
    assert compute_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0

def test_iou_no_overlap():
    assert compute_iou([0, 0, 5, 5], [10, 10, 20, 20]) == 0.0

def test_iou_partial_overlap():
    iou = compute_iou([0, 0, 10, 10], [5, 5, 15, 15])
    assert 0.1 < iou < 0.2  # 25/175 ≈ 0.143

def test_point_inside_box():
    assert point_inside_box([0, 0, 10, 10], [[5, 5]]) is True
    assert point_inside_box([0, 0, 10, 10], [[15, 15]]) is False

def test_union_find_groups():
    uf = UnionFind(4)
    uf.union(0, 1)
    uf.union(2, 3)
    assert uf.find(0) == uf.find(1)
    assert uf.find(2) == uf.find(3)
    assert uf.find(0) != uf.find(2)
```

---

## Execution Plan

### Phase 1: Extract `lib/` modules (no behavior change)

| Step | Module | Eliminates | Risk |
|------|--------|-----------|------|
| 1 | `lib/geometry.py` | IoU, point-in-box, union-find, denormalization | None — pure math |
| 2 | `lib/types.py` | Empty result factory | None — pure factories |
| 3 | `lib/mask_ops.py` | Normalize, fill_holes, dilate, apply_per_mask | Low — well-defined I/O |
| 4 | `lib/prompt_handler.py` | Mode selection, aggregation, valid_block | None — pure dicts |
| 5 | `lib/mask_filters.py` | Size, density, instance, max_detections | Low — uses geometry.py |
| 6 | `lib/segs_builder.py` | All SEGS construction | Medium — uses masktosegs.py |

**Each step**: Extract → write tests → replace calls in nodes.py → verify node still works.

### Phase 2: Split `nodes.py` into `nodes/` directory

| Step | File | Contents |
|------|------|----------|
| 7 | `nodes/model_loaders.py` | `TBGLoadSAM3Model`, `TBGSAM3ModelLoaderAndDownloader` |
| 8 | `nodes/segmentation.py` | `TBGSam3Segmentation`, `TBGSam3SegmentationBatch` |
| 9 | `nodes/prompt_collector.py` | `TBGSAM3PromptCollector` |
| 10 | `nodes/depth_map.py` | `TBGSAM3DepthMap` |
| 11 | Update `__init__.py` | Merge all `NODE_CLASS_MAPPINGS` |

### Phase 3: Eliminate remaining duplication

| Step | Action |
|------|--------|
| 12 | Rewrite `TBGSam3SegmentationBatch.segment()` to use `lib/` modules |
| 13 | Remove all duplicated code paths |
| 14 | Replace `print()` with `logging` |

---

## Expected Outcome

| Metric | Before | After |
|--------|--------|-------|
| `nodes.py` | 1,616 lines | **0** (split into 4 files, ~80 lines each) |
| `segment()` | 560 lines | ~50 lines |
| Duplicated patterns | 14 | 0 |
| Duplicated lines (estimate) | ~400 | 0 |
| Test coverage | 0% | ~80% of `lib/` |
| Testability score | 15/100 | ~75/100 |
| Files | 4 root .py | 4 `nodes/` + 6 `lib/` + tests |

`lib/` modules are reusable outside ComfyUI. Node classes are thin adapters. Every extracted function has unit tests. Zero behavior changes.
