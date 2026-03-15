import numpy as np
import torch

from .masktosegs import make_2d_mask, mask_to_segs

from .geometry import UnionFind


def make_label(text_prompt: str, index: int) -> str:
    """Build a SEG label from prompt text and detection index."""
    if text_prompt and text_prompt.strip():
        return f"{text_prompt}_{index}"
    return f"detection_{index}"


def build_contour_segs(
    masks: torch.Tensor | None,
    text_prompt: str,
    width: int,
    height: int,
    crop_factor: float = 1.0,
) -> tuple[tuple[int, int], list]:
    """Split each mask into contour-based SEGs via cv2.findContours."""
    shape_info = (height, width)
    seg_list: list = []

    if masks is None or len(masks) == 0:
        return shape_info, seg_list

    masks_cpu = masks.detach().cpu() if isinstance(masks, torch.Tensor) else masks
    for i in range(len(masks_cpu)):
        mask_2d = make_2d_mask(masks_cpu[i])
        label = make_label(text_prompt, index=i)

        _, segs_inst = mask_to_segs(
            mask_2d,
            combined=False,
            crop_factor=crop_factor,
            bbox_fill=False,
            drop_size=1,
            label=label,
            crop_min_size=None,
            detailer_hook=None,
            is_contour=True,
        )
        if segs_inst:
            seg_list.extend(segs_inst)

    return shape_info, seg_list


def build_detection_segs(
    masks: torch.Tensor | None,
    text_prompt: str,
    width: int,
    height: int,
    crop_factor: float,
) -> tuple[tuple[int, int], list]:
    """Build one SEG per detection using combined mode (no contour splitting)."""
    shape_info = (height, width)
    detection_seg_list: list = []

    if masks is None or len(masks) == 0:
        return shape_info, detection_seg_list

    masks_cpu = masks.detach().cpu()
    for i in range(len(masks_cpu)):
        mask_2d = make_2d_mask(masks_cpu[i])
        dlabel = make_label(text_prompt, index=i)
        _, segs_inst = mask_to_segs(
            mask_2d,
            combined=True,
            crop_factor=crop_factor,
            bbox_fill=False,
            drop_size=1,
            label=dlabel,
        )
        if segs_inst:
            detection_seg_list.extend(segs_inst)

    return shape_info, detection_seg_list


def build_combined_segs(
    combined_tensor: torch.Tensor,
    text_prompt: str,
    width: int,
    height: int,
    crop_factor: float,
) -> tuple[tuple[int, int], list]:
    """Build a single SEG from the OR-merged combined mask."""
    combined_label = text_prompt.strip() or "combined"
    combined_cpu = combined_tensor.detach().cpu()
    combined_2d = make_2d_mask(combined_cpu)
    _, seg_list = mask_to_segs(
        combined_2d,
        combined=True,
        crop_factor=crop_factor,
        bbox_fill=False,
        drop_size=1,
        label=combined_label,
        crop_min_size=None,
        detailer_hook=None,
        is_contour=True,
    )
    return (height, width), seg_list


def build_overlapping_segs(
    masks: torch.Tensor | None,
    text_prompt: str,
    width: int,
    height: int,
    crop_factor: float,
) -> tuple[tuple[int, int], list]:
    """Group masks by pixel overlap (union-find) and merge each group into one SEG."""
    shape_info = (height, width)
    overlapping_seg_list: list = []

    if masks is None or len(masks) == 0:
        return shape_info, overlapping_seg_list

    masks_cpu = masks.detach().cpu()
    n = len(masks_cpu)
    binary_masks = [(make_2d_mask(masks_cpu[i]) > 0.5) for i in range(n)]

    uf = UnionFind(n)

    bboxes_ol: list[tuple[int, int, int, int] | None] = []
    for bm in binary_masks:
        rows, cols = np.nonzero(bm)
        if len(rows) > 0:
            bboxes_ol.append((cols.min(), rows.min(), cols.max(), rows.max()))
        else:
            bboxes_ol.append(None)

    for i in range(n):
        for j in range(i + 1, n):
            if bboxes_ol[i] is None or bboxes_ol[j] is None:
                continue
            if (
                bboxes_ol[i][0] <= bboxes_ol[j][2]
                and bboxes_ol[i][2] >= bboxes_ol[j][0]
                and bboxes_ol[i][1] <= bboxes_ol[j][3]
                and bboxes_ol[i][3] >= bboxes_ol[j][1]
            ):
                if np.any(binary_masks[i] & binary_masks[j]):
                    uf.union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = uf.find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    for _, indices in groups.items():
        merged = binary_masks[indices[0]].astype(np.float32)
        for idx in indices[1:]:
            merged = np.maximum(merged, binary_masks[idx].astype(np.float32))

        if text_prompt and text_prompt.strip():
            glabel = f"{text_prompt}_group_{'_'.join(str(i) for i in sorted(indices))}"
        else:
            glabel = f"group_{'_'.join(str(i) for i in sorted(indices))}"

        _, segs_group = mask_to_segs(
            merged,
            combined=True,
            crop_factor=crop_factor,
            bbox_fill=False,
            drop_size=1,
            label=glabel,
        )
        if segs_group:
            overlapping_seg_list.extend(segs_group)

    return shape_info, overlapping_seg_list
