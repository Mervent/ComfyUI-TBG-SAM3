from collections.abc import Sequence


def denormalize_boxes(
    normalized_boxes: Sequence[Sequence[float]],
    width: int,
    height: int,
) -> list[list[float]]:
    """Convert center-format normalized boxes to absolute pixel [x1,y1,x2,y2]."""
    boxes = []
    for cx, cy, w_norm, h_norm in normalized_boxes:
        x1 = (cx - w_norm / 2.0) * width
        y1 = (cy - h_norm / 2.0) * height
        x2 = (cx + w_norm / 2.0) * width
        y2 = (cy + h_norm / 2.0) * height
        boxes.append([x1, y1, x2, y2])
    return boxes


def denormalize_points(
    normalized_points: Sequence[Sequence[float]],
    width: int,
    height: int,
) -> list[list[float]]:
    """Convert normalized 0-1 points to absolute pixel coordinates."""
    points = []
    for px_norm, py_norm in normalized_points:
        px = px_norm * width
        py = py_norm * height
        points.append([px, py])
    return points


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute intersection-over-union between two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    if inter > 0:
        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        union = area_a + area_b - inter
        if union > 0:
            return inter / union

    return 0.0


def max_iou_with_boxes(
    detection_box: Sequence[float],
    prompt_boxes: Sequence[Sequence[float]],
) -> float:
    """Return the highest IoU between a detection box and a list of prompt boxes."""
    best = 0.0
    for prompt_box in prompt_boxes:
        iou_val = compute_iou(detection_box, prompt_box)
        if iou_val > best:
            best = iou_val
    return best


def point_inside_box(box: Sequence[float], points: Sequence[Sequence[float]]) -> bool:
    """Return True if any point falls inside the box (inclusive boundaries)."""
    ax1, ay1, ax2, ay2 = box
    for px, py in points:
        if ax1 <= px <= ax2 and ay1 <= py <= ay2:
            return True
    return False


class UnionFind:
    """Disjoint-set with path compression for grouping overlapping masks."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, x: int) -> int:
        """Find root with path-halving compression."""
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        """Merge the sets containing a and b."""
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb
