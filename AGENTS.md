# AGENTS.md — ComfyUI-TBG-SAM3

## What This Project Is

A ComfyUI custom-node extension integrating Meta's SAM3 for image segmentation. Provides ComfyUI nodes for text-prompt segmentation, point/box-guided masks, and mask refinement. Outputs Impact Pack SEGS format.

## Architecture

```
ComfyUI-TBG-SAM3/
├── nodes.py              # ComfyUI node classes (thin adapters → lib/)
├── lib/                  # Pure logic — zero ComfyUI imports
│   ├── geometry.py       # IoU, point-in-box, box/point denormalization, UnionFind
│   ├── mask_ops.py       # normalize, fill_holes, dilate_erode, apply_per_mask, build_combined_mask
│   ├── mask_filters.py   # filter_by_size, filter_by_density, filter_by_instances, limit_detections
│   ├── segs_builder.py   # build_contour_segs, build_detection_segs, build_combined_segs, build_overlapping_segs
│   ├── prompt_handler.py # extract_by_mode, aggregate_prompts, valid_block
│   ├── masktosegs.py     # mask_to_segs, make_2d_mask (Impact Pack SEGS conversion)
│   └── types.py          # empty_masks, empty_segs, empty_segmentation_result
├── sam3_utils.py          # Tensor/image conversions, SAM3ImageSegmenter, DepthEstimator
├── model_manager.py       # Model download/path resolution
├── sam3_lib/              # Vendored Meta SAM3
├── web/                   # Frontend JS (ComfyUI widget)
└── tests/                 # pytest suite — unit tests for lib/ modules
```

### The Golden Rule

`lib/` has **zero ComfyUI imports**. Pure Python + torch + numpy + cv2.
`nodes.py` is a **thin adapter** — INPUT_TYPES + delegation to `lib/`.

### Data Flow (single image segmentation)

```
nodes.py TBGSam3Segmentation.segment()
  ├─ prompt_handler.extract_by_mode()     → route pipeline prompts by mode
  ├─ SAM3 inference (sam3_utils)          → raw masks, boxes, scores
  ├─ mask_filters.filter_by_size()        → remove small masks
  ├─ mask_filters.filter_by_density()     → remove sparse masks
  ├─ mask_filters.filter_by_instances()   → keep only prompt-matched detections
  ├─ mask_filters.limit_detections()      → top-k by score
  ├─ mask_ops.apply_per_mask(fill_holes)  → morphology
  ├─ mask_ops.apply_per_mask(dilate_erode)→ morphology
  ├─ mask_ops.build_combined_mask()       → OR-merge all masks
  └─ segs_builder.build_*_segs()          → 4 SEGS output variants
```

## Key Types

- **Masks**: `torch.Tensor` shaped `[N, H, W]` (float32, 0-1). 4D `[N, 1, H, W]` normalized via `mask_ops.normalize_masks()`.
- **SEGS**: `tuple[tuple[int, int], list[SEG]]` where shape is `(height, width)`. Each SEG has `cropped_mask`, `crop_region`, `bbox`, `label`, `confidence`.
- **Boxes**: `torch.Tensor` shaped `[N, 4]` as `[x1, y1, x2, y2]` pixel coords.
- **Scores**: `torch.Tensor` shaped `[N]` confidence values.
- **Pipeline prompts**: `dict` with keys `positive_boxes`, `negative_boxes`, `positive_points`, `negative_points`. Each is `{"boxes"|"points": [[...]], "labels": [int]}` or `None`.

## Module Reference

### `lib/geometry.py`
Pure math. No torch dependency for core functions.
- `compute_iou(box_a, box_b)` — IoU between two `[x1,y1,x2,y2]` boxes
- `max_iou_with_boxes(detection_box, prompt_boxes)` — max IoU against list
- `point_inside_box(box, points)` — any point inside box?
- `denormalize_boxes(normalized, w, h)` — `(cx,cy,w,h)` normalized → `[x1,y1,x2,y2]` pixel
- `denormalize_points(normalized, w, h)` — `(px,py)` normalized → pixel
- `UnionFind(n)` — union-find for grouping overlapping masks

### `lib/mask_ops.py`
Tensor ↔ numpy mask operations.
- `normalize_masks(masks, to_cpu=False)` — `[N,1,H,W]` or `[N,C,H,W]` → `[N,H,W]`
- `restore_mask_shape(processed, original)` — undo normalize for 4D originals
- `apply_per_mask(masks, fn)` — normalize → per-mask numpy fn → restore shape
- `fill_holes(mask)` — flood-fill internal holes in single `[H,W]` numpy mask
- `dilate_erode(mask, dilation)` — positive=dilate, negative=erode, zero=no-op
- `build_combined_mask(masks)` — logical OR across all masks → `[1,H,W]`

### `lib/mask_filters.py`
All filters return `(masks, boxes, scores)` triplet. Returns `(None, None, None)` when all filtered out.
- `filter_by_size(masks, boxes, scores, min_size)` — remove masks < `min_size²` pixels
- `filter_by_density(masks, boxes, scores, min_density)` — remove masks where `fg/bbox_area < min_density`
- `filter_by_instances(masks, boxes, scores, pos_boxes, pos_points, w, h, iou_threshold)` — keep detections matching positive prompts
- `limit_detections(masks, boxes, scores, max_detections)` — top-k by score

### `lib/segs_builder.py`
All builders return `tuple[tuple[int, int], list]` — `((height, width), seg_list)`.
- `build_contour_segs(masks, text_prompt, w, h, crop_factor)` — split each mask into contour-based SEGs
- `build_detection_segs(masks, text_prompt, w, h, crop_factor)` — one SEG per detection (no contour split)
- `build_combined_segs(combined_tensor, text_prompt, w, h, crop_factor)` — single SEG from OR-merged mask
- `build_overlapping_segs(masks, text_prompt, w, h, crop_factor)` — group overlapping masks via UnionFind, one SEG per group
- `make_label(text_prompt, index)` — `"prompt_N"` or `"detection_N"`

### `lib/prompt_handler.py`
Pure dict manipulation — no torch.
- `valid_block(block, key)` — check if prompt block has non-empty key
- `extract_by_mode(pipeline_data, mode)` — route prompts by mode (`all`, `boxes_only`, `points_only`, `positive_only`, `negative_only`, `disabled`)
- `aggregate_prompts(positive_block, negative_block, prompt_type)` — merge pos+neg into flat `(prompts, labels)` lists

### `lib/masktosegs.py`
Impact Pack SEGS conversion (legacy, ported from Impact Pack).
- `mask_to_segs(mask, combined, crop_factor, ...)` — numpy mask → SEGS tuple
- `make_2d_mask(mask)` — ensure mask is 2D numpy array

### `lib/types.py`
Factory functions for empty/default values.
- `empty_masks(h, w, device)` — `[1,H,W]` zero tensor
- `empty_segs(h, w)` — `((H,W), [])`
- `empty_segmentation_result(h, w, vis_fn, pil_image, device)` — full 9-tuple empty return

## Node Classes (in `nodes.py`)

| Class | Purpose | Outputs |
|---|---|---|
| `TBGSam3Segmentation` | Single-image segmentation | 9: masks, vis, boxes_json, scores_json, segs, combined_mask, combined_segs, detection_segs, overlapping_segs |
| `TBGSam3SegmentationBatch` | Batch segmentation | 4: segs, vis, combined_mask, batch_masks |
| `TBGLoadSAM3Model` | Load model from local path | model dict |
| `TBGSAM3ModelLoaderAndDownloader` | Download + load model | model dict |
| `TBGSAM3PromptCollector` | Collect points/boxes from UI | pipeline dict |
| `TBGSAM3DepthMap` | Depth estimation via MiDaS | depth map, visualization |

## Development

```bash
make venv      # create .venv with Python 3.10 via uv
make test      # run pytest
make lint      # ruff check
make format    # ruff format
```

- Tests cover `lib/` modules only (pure logic, no SAM3 model needed)
- Fixtures in `tests/conftest.py`: synthetic masks, boxes, scores
- See `STYLE-GUIDE.md` for test conventions (AAA, type hints, kwargs)

## Style

See `STYLE-GUIDE.md` for coding conventions.
