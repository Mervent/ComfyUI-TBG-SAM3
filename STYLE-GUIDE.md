# Style Guide

## Python Version

Target **Python 3.10+**. Use modern syntax:

```python
# YES
x: torch.Tensor | None = None
def foo() -> tuple[int, int]: ...
_Triplet: TypeAlias = tuple[...]

# NO
x: Optional[torch.Tensor] = None
def foo() -> Tuple[int, int]: ...
type _Triplet = tuple[...]  # 3.12-only
```

## Package Structure

- `lib/` — Pure logic. **Zero ComfyUI imports.** Only torch, numpy, cv2, stdlib.
- `nodes/` — Thin ComfyUI adapters. INPUT_TYPES + one-liner delegation to `lib/`.
- All imports within the package use **relative imports** (`from .module import ...`).

## Functions

- **Use all parameters.** No dead params in signatures.
- **Consistent return types.** Same shape on all code paths (no conditional tuple sizes).
- **No-op means no-op.** Identity operations must return input unchanged, not a transformed version.
- **Trust the lib.** Callers should not re-check preconditions that the called function already handles.
- **No unused imports.**

## Function Calls

Use keyword arguments when passing more than 2 parameters:

```python
# YES — 2 params, positional is fine
compute_iou(box_a, box_b)
make_label("cat", index=0)

# YES — 3+ params, use kwargs
masks, boxes, scores = filter_by_size(
    masks=masks,
    boxes=boxes,
    scores=scores,
    min_size=32,
)

# NO — 3+ positional args are hard to read
masks, boxes, scores = filter_by_size(masks, boxes, scores, 32)
```

## Null Handling

One pattern — early return guard, then proceed:

```python
if masks is None or len(masks) == 0:
    return shape_info, []

masks_cpu = masks.detach().cpu()
# ... work with masks_cpu
```

Do not inline the guard into a ternary assignment.

## Tests

- All test functions annotated `-> None`.
- AAA separation with blank lines:

```python
def test_filter_removes_small(sample_boxes, sample_scores) -> None:
    """Single-pixel mask is removed, 8x8 block is kept."""
    masks = torch.zeros((2, 20, 20), dtype=torch.float32)
    masks[0, 10, 10] = 1.0
    masks[1, 2:10, 2:10] = 1.0

    out_masks, out_boxes, out_scores = filter_by_size(
        masks=masks, boxes=sample_boxes[:2], scores=sample_scores[:2], min_size=4,
    )

    assert out_masks is not None
    assert out_masks.shape[0] == 1
```

- Trivial one-liner asserts (no arrange/act) need no blank lines.
- Fixtures live in `conftest.py` with return type hints.
