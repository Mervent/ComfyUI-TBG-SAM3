"""
ComfyUI SAM3 Nodes, unified model loader for both image and video using official Meta sam3_lib.
All class names and functions prefixed with TBG for uniqueness.
"""

import base64
import io
import json
import os

import numpy as np
import torch
from PIL import Image

from sam3_utils import (
    DepthEstimator,
    comfy_image_to_pil,
    ensure_model_on_device,
    masks_to_comfy_mask,
    offload_model_if_needed,
    pil_to_comfy_image,
    tensor_to_list,
    visualize_masks_on_image,
)

# Impact-Pack style MASK -> SEGS helper (your file in same folder)
from .lib import mask_filters, mask_ops, prompt_handler
from .lib import types as lib_types
from .lib.conditioning_wrapper import ConditioningOverrideWrapper
from .lib.florence2_captioner import (
    TASK_LIST,
    bbox_crops_to_tensor,
    build_caption,
    caption_image,
    crop_seg_bbox,
)
from .lib.masktosegs import SEG
from .lib.segs_builder import (
    build_combined_segs,
    build_contour_segs,
    build_detection_segs,
    build_overlapping_segs,
)
from .model_manager import download_sam3_model, get_available_models, get_model_path
from .sam3_lib.model.sam3_image_processor import Sam3Processor
from .sam3_lib.model_builder import build_sam3_image_model

try:
    import folder_paths

    base_models_folder = folder_paths.models_dir
except ImportError:
    base_models_folder = "models"


_MODEL_CACHE = {}


class TBGSAM3ModelLoaderAndDownloader:
    """
    Advanced SAM3 model loader that:
    - Can use the official API (auto configuration)
    - Can auto-download a local checkpoint if missing
    - Can load a specific local checkpoint under models/sam3
    Returns the same SAM3_MODEL dict as TBGLoadSAM3Model.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # List known local models from model_manager
        # get_available_models() returns ["auto (download from HuggingFace)", <files...>]
        available = get_available_models()
        # Present clearer choices in UI
        model_sources = (
            [
                "auto (API to cache)",  # build default model (no fixed ckpt path)
                "local (auto-download)",  # download sam3.pt into models/sam3 if missing
            ]
            + available[1:]
        )  # additional discovered checkpoint files

        return {
            "required": {
                "model_source": (model_sources, {"default": "local (auto-download)"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            },
        }

    RETURN_TYPES = ("SAM3_MODEL",)
    RETURN_NAMES = ("sam3_model",)
    FUNCTION = "load_model"
    CATEGORY = "TBG/SAM3"

    def load_model(self, model_source: str, device: str):
        hf_repo = "facebook/sam3"

        """
        Build and return a SAM3_MODEL dict:
          {model, processor, device, original_device}
        """
        # Resolve checkpoint path if needed
        checkpoint_path = None

        if model_source == "auto (API to cache)":
            # Let builder construct its default weights / config
            print("[TBGSAM3ModelLoaderAdvanced] Using API/default SAM3 image model.")
            checkpoint_path = None

        elif model_source == "local (auto-download)":
            # Download only sam3.pt into models/sam3
            sam3_dir = download_sam3_model(hf_repo)  # returns models/sam3
            checkpoint_path = os.path.join(sam3_dir, "sam3.pt")
            if not os.path.isfile(checkpoint_path):
                raise RuntimeError(
                    f"[TBGSAM3ModelLoaderAdvanced] Downloaded model file not found at: {checkpoint_path}"
                )
            print(
                f"[TBGSAM3ModelLoaderAdvanced] Using downloaded local checkpoint: {checkpoint_path}"
            )

        else:
            # Specific local checkpoint chosen from list under models/sam3
            checkpoint_path = get_model_path(model_source)
            if not checkpoint_path or not os.path.isfile(checkpoint_path):
                raise RuntimeError(
                    f"[TBGSAM3ModelLoaderAdvanced] Local model file not found: {model_source} -> {checkpoint_path}"
                )
            print(
                f"[TBGSAM3ModelLoaderAdvanced] Using selected local checkpoint: {checkpoint_path}"
            )

        # --- Build SAM3 image model + processor, mirroring TBGLoadSAM3Model ---

        if checkpoint_path:
            sam3_model = build_sam3_image_model(checkpoint_path=checkpoint_path)
        else:
            sam3_model = build_sam3_image_model()

        processor = Sam3Processor(sam3_model)

        sam3_model.to(device)
        sam3_model.processor = processor
        sam3_model.eval()

        model_dict = {
            "model": sam3_model,
            "processor": processor,
            "device": device,
            "original_device": device,
        }

        print("[TBGSAM3ModelLoaderAdvanced] SAM3 model ready on device:", device)
        return (model_dict,)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    arr = tensor.permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    arr = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


class TBGLoadSAM3Model:
    """
    Simple SAM3 loader using the new models/sam3 folder.

    Currently supports image mode only (no video).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("SAM3_MODEL",)
    FUNCTION = "tbg_load_model"
    CATEGORY = "TBG/SAM3"

    def tbg_load_model(self, device: str):
        # Ensure base folder exists (models/sam3)
        _ = get_available_models()  # implicitly creates models/sam3 via model_manager

        model = build_sam3_image_model()
        processor = Sam3Processor(model)
        model.to(device)
        model.processor = processor
        model.eval()

        model_dict = {
            "model": model,
            "processor": processor,
            "device": device,
            "original_device": device,
        }

        return (model_dict,)


class TBGSam3Segmentation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_model": (
                    "SAM3_MODEL",
                    {"tooltip": "SAM3 model loaded from LoadSAM3Model node"},
                ),
                "image": (
                    "IMAGE",
                    {"tooltip": "Input image to perform segmentation on"},
                ),
                "confidence_threshold": (
                    "FLOAT",
                    {
                        "default": 0.4,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "display": "slider",
                        "tooltip": "Minimum confidence score to keep detections. Lower threshold (0.2) works better with SAM3's presence scoring",
                    },
                ),
                "pipeline_mode": (
                    [
                        "all",
                        "boxes_only",
                        "points_only",
                        "positive_only",
                        "negative_only",
                        "disabled",
                    ],
                    {
                        "default": "all",
                        "tooltip": "Which prompts from pipeline to use.",
                    },
                ),
                "detect_all": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "Detect All",
                        "label_off": "Limit Detections to max_detection",
                        "tooltip": "When enabled, detects all objects. When disabled, uses max_detections value.",
                    },
                ),
                "max_detections": (
                    "INT",
                    {
                        "default": 50,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Maximum detections when detect_all is disabled.",
                    },
                ),
                "instances": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "No Instances",
                        "label_off": "All Instances",
                        "tooltip": (
                            "When ON: keep only detections whose boxes overlap a positive box or contain a positive point.\n"
                            "When OFF: return all SAM3 detections including instances."
                        ),
                    },
                ),
                "crop_factor": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 1.0,
                        "max": 4.0,
                        "step": 0.1,
                        "tooltip": "Crop factor used when building combined SEGS (Impact Pack style). 1.0 = tight bbox.",
                    },
                ),
                "min_size": (
                    "INT",
                    {
                        "default": 100,
                        "min": 1,
                        "max": 500,
                        "step": 1,
                        "display": "slider",
                        "tooltip": "Minimum segment size in pixels as a square side. 1=1x1, 200=200x200; smaller masks are discarded.",
                    },
                ),
                "min_density": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": "Minimum mask density (foreground pixels / bbox area). 0=disabled, 0.1=at least 10% of bbox must be filled. Filters sparse/feathery masks.",
                    },
                ),
                "fill_holes": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "Fill Holes",
                        "label_off": "Keep Holes",
                        "tooltip": "When enabled, fills holes inside each mask (solid segments).",
                    },
                ),
                "dilation": (
                    "INT",
                    {
                        "default": 0,
                        "min": -512,
                        "max": 512,
                        "step": 1,
                        "tooltip": "Expand (positive) or shrink (negative) the mask boundary. 0 = no change.",
                    },
                ),
            },
            "optional": {
                "text_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "e.g., 'cat', 'person in red', 'car'",
                        "tooltip": "Text to guide segmentation (optional).",
                    },
                ),
                "sam3_selectors_pipe": (
                    "SAM3_PROMPT_PIPELINE",
                    {"tooltip": "Unified pipeline containing boxes/points)."},
                ),
                "mask_prompt": (
                    "MASK",
                    {"tooltip": "Optional mask to refine the segmentation."},
                ),
            },
        }

    RETURN_TYPES = (
        "MASK",
        "IMAGE",
        "STRING",
        "STRING",
        "SEGS",
        "MASK",
        "SEGS",
        "SEGS",
        "SEGS",
    )
    RETURN_NAMES = (
        "masks",
        "visualization",
        "boxes",
        "scores",
        "segs",
        "combined_mask",
        "combined_segs",
        "detection_segs",
        "overlapping_segs",
    )
    FUNCTION = "segment"
    CATEGORY = "TBG/SAM3"

    def segment(
        self,
        sam3_model,
        image,
        confidence_threshold=0.2,
        detect_all=True,
        pipeline_mode="all",
        instances=False,
        crop_factor=1.5,
        min_size=32,
        min_density=0.0,
        fill_holes=False,
        dilation=0,
        text_prompt="",
        sam3_selectors_pipe=None,
        mask_prompt=None,
        exemplar_box=None,
        exemplar_mask=None,
        max_detections=10,
    ):
        actual_max_detections = -1 if detect_all else max_detections

        positive_boxes, negative_boxes, positive_points, negative_points = (
            prompt_handler.extract_by_mode(
                sam3_selectors_pipe,
                pipeline_mode,
            )
        )
        print(
            f"[SAM3] pipeline_mode='{pipeline_mode}', instances={instances} | "
            f"pos_boxes={prompt_handler.valid_block(positive_boxes, 'boxes')}, "
            f"neg_boxes={prompt_handler.valid_block(negative_boxes, 'boxes')}, "
            f"pos_points={prompt_handler.valid_block(positive_points, 'points')}, "
            f"neg_points={prompt_handler.valid_block(negative_points, 'points')}"
        )

        ensure_model_on_device(sam3_model)
        processor = sam3_model["processor"]
        print("[SAM3] Running segmentation")
        print(f"[SAM3] Confidence threshold: {confidence_threshold}")

        pil_image = comfy_image_to_pil(image)
        print(f"[SAM3] Image size: {pil_image.size}")

        _, height, width, _ = image.shape
        processor.set_confidence_threshold(confidence_threshold)
        state = processor.set_image(pil_image)

        if text_prompt and text_prompt.strip():
            print(f"[SAM3] Using text_prompt='{text_prompt.strip()}'")
            state = processor.set_text_prompt(text_prompt.strip(), state)

        all_boxes, all_box_labels = prompt_handler.aggregate_prompts(
            positive_boxes, negative_boxes, "boxes"
        )
        print(f"[SAM3] total box prompts={len(all_boxes)}")
        if all_boxes:
            state = processor.add_multiple_box_prompts(all_boxes, all_box_labels, state)

        all_points, all_point_labels = prompt_handler.aggregate_prompts(
            positive_points, negative_points, "points"
        )
        print(f"[SAM3] total point prompts={len(all_points)}")
        if all_points:
            state = processor.add_point_prompt(all_points, all_point_labels, state)

        if mask_prompt is not None:
            if not isinstance(mask_prompt, torch.Tensor):
                mask_prompt = torch.from_numpy(mask_prompt)
            mask_prompt = mask_prompt.to(sam3_model["device"])
            print("[SAM3] Adding external mask_prompt")
            state = processor.add_mask_prompt(mask_prompt, state)

        masks = state.get("masks", None)
        boxes = state.get("boxes", None)
        scores = state.get("scores", None)

        total_scores = len(scores) if scores is not None else 0
        print(f"[SAM3 DEBUG] RAW PREDICTIONS: total {total_scores}")
        if boxes is not None:
            print(f"[SAM3 DEBUG] Output boxes shape: {boxes.shape}")

        device_before = masks.device if masks is not None else "cpu"

        before_count = len(masks) if masks is not None else 0
        masks, boxes, scores = mask_filters.filter_by_size(
            masks, boxes, scores, min_size
        )
        if masks is None:
            print(f"[SAM3] All {before_count} detections removed by min_size filter")
            offload_model_if_needed(sam3_model)
            return lib_types.empty_segmentation_result(
                height, width, pil_to_comfy_image, pil_image, device=device_before
            )
        if len(masks) < before_count:
            print(
                f"[SAM3] min_size={min_size}px removed {before_count - len(masks)} masks, keeping {len(masks)}"
            )

        before_count = len(masks)
        masks, boxes, scores = mask_filters.filter_by_density(
            masks, boxes, scores, min_density
        )
        if masks is None:
            print(f"[SAM3] All {before_count} detections removed by min_density filter")
            offload_model_if_needed(sam3_model)
            return lib_types.empty_segmentation_result(
                height, width, pil_to_comfy_image, pil_image, device=device_before
            )
        if len(masks) < before_count:
            print(
                f"[SAM3] min_density={min_density:.2f} removed {before_count - len(masks)} sparse masks, keeping {len(masks)}"
            )

        if masks is None or len(masks) == 0:
            print(f"[SAM3] No detections found at threshold {confidence_threshold}")
            offload_model_if_needed(sam3_model)
            return lib_types.empty_segmentation_result(
                height, width, pil_to_comfy_image, pil_image, device=device_before
            )

        if instances and boxes is not None:
            print(
                "[SAM3] Instances filter: keep only detections overlapping positive boxes / containing positive points"
            )
            before_instances = len(boxes)
            print(
                f"[SAM3] Instances filter: total detections before filter={before_instances}"
            )
            boxes_device = boxes.device
            masks, boxes, scores = mask_filters.filter_by_instances(
                masks,
                boxes,
                scores,
                positive_boxes,
                positive_points,
                width,
                height,
                iou_threshold=0.1,
            )
            if masks is None:
                print(
                    "[SAM3] Instances filter removed all detections; returning empty result"
                )
                offload_model_if_needed(sam3_model)
                return lib_types.empty_segmentation_result(
                    height, width, pil_to_comfy_image, pil_image, device=boxes_device
                )
            print(
                f"[SAM3] Instances filter kept {len(masks)} of {before_instances} detections"
            )

        masks, boxes, scores = mask_filters.limit_detections(
            masks, boxes, scores, actual_max_detections
        )

        if fill_holes:
            masks = mask_ops.apply_per_mask(masks, mask_ops.fill_holes)

        if dilation != 0:
            masks = mask_ops.apply_per_mask(
                masks, lambda m: mask_ops.dilate_erode(m, dilation)
            )

        comfy_masks = masks_to_comfy_mask(masks)
        combined_tensor = mask_ops.build_combined_mask(masks)
        combined_mask = masks_to_comfy_mask(combined_tensor)
        vis_image = visualize_masks_on_image(pil_image, masks, boxes, scores, alpha=0.5)
        vis_tensor = pil_to_comfy_image(vis_image)

        def tensor_to_list_safe(t):
            if t is None:
                return []
            return tensor_to_list(t)

        boxes_json = json.dumps(tensor_to_list_safe(boxes), indent=2)
        scores_json = json.dumps(tensor_to_list_safe(scores), indent=2)

        segs = build_contour_segs(
            masks, text_prompt, width, height, crop_factor=crop_factor
        )
        combined_segs = build_combined_segs(
            combined_tensor, text_prompt, width, height, crop_factor
        )
        detection_segs = build_detection_segs(
            masks, text_prompt, width, height, crop_factor
        )
        overlapping_segs = build_overlapping_segs(
            masks, text_prompt, width, height, crop_factor
        )
        combined_count = (
            len(combined_segs[1])
            if isinstance(combined_segs, tuple) and len(combined_segs) > 1
            else 0
        )
        detection_count = (
            len(detection_segs[1])
            if isinstance(detection_segs, tuple) and len(detection_segs) > 1
            else 0
        )
        overlapping_count = (
            len(overlapping_segs[1])
            if isinstance(overlapping_segs, tuple) and len(overlapping_segs) > 1
            else 0
        )

        print(
            f"[SAM3] Segmentation complete. {len(comfy_masks)} masks, {len(segs[1])} SEGS, "
            f"combined_segs has {combined_count} elements, "
            f"detection_segs has {detection_count} elements, "
            f"overlapping_segs has {overlapping_count} groups."
        )

        offload_model_if_needed(sam3_model)
        return (
            comfy_masks,
            vis_tensor,
            boxes_json,
            scores_json,
            segs,
            combined_mask,
            combined_segs,
            detection_segs,
            overlapping_segs,
        )


class TBGSam3SegmentationBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_model": (
                    "SAM3_MODEL",
                    {"tooltip": "SAM3 model loaded from LoadSAM3Model node"},
                ),
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Input image batch to perform segmentation on (B,H,W,C)"
                    },
                ),
                "confidence_threshold": (
                    "FLOAT",
                    {
                        "default": 0.4,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "display": "slider",
                        "tooltip": "Minimum confidence score to keep detections",
                    },
                ),
                "pipeline_mode": (
                    [
                        "all",
                        "boxes_only",
                        "points_only",
                        "positive_only",
                        "negative_only",
                        "disabled",
                    ],
                    {"default": "all", "tooltip": "Which prompts from pipeline to use"},
                ),
                "detect_all": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "Detect All",
                        "label_off": "Limit Detections",
                        "tooltip": "When enabled, detects all objects. When disabled, uses max_detections",
                    },
                ),
                "max_detections": (
                    "INT",
                    {
                        "default": 50,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Maximum detections when detect_all is disabled",
                    },
                ),
                "instances": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "Filter Instances",
                        "label_off": "All Instances",
                        "tooltip": "When ON: keep only detections overlapping positive prompts",
                    },
                ),
                "crop_factor": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 1.0,
                        "max": 4.0,
                        "step": 0.1,
                        "tooltip": "Crop factor for SEGS. 1.0 = tight bbox",
                    },
                ),
                "min_size": (
                    "INT",
                    {
                        "default": 100,
                        "min": 1,
                        "max": 500,
                        "step": 1,
                        "display": "slider",
                        "tooltip": "Minimum segment size (square side in pixels)",
                    },
                ),
                "fill_holes": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "Fill Holes",
                        "label_off": "Keep Holes",
                        "tooltip": "Fill holes inside masks",
                    },
                ),
            },
            "optional": {
                "text_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "e.g., 'cat', 'person in red', 'car'",
                        "tooltip": "Text to guide segmentation",
                    },
                ),
                "sam3_selectors_pipe": (
                    "SAM3_PROMPT_PIPELINE",
                    {"tooltip": "Unified pipeline containing boxes/points"},
                ),
                "mask_prompt": (
                    "MASK",
                    {"tooltip": "Optional batch mask (B,H,W or 1,H,W)"},
                ),
            },
        }

    # Impact Pack compatible outputs
    RETURN_TYPES = ("SEGS", "IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("segs", "visualization", "combined_mask", "batch_masks")
    FUNCTION = "segment"
    CATEGORY = "TBG/SAM3"

    def segment(
        self,
        sam3_model,
        image,
        confidence_threshold=0.4,
        detect_all=True,
        pipeline_mode="all",
        instances=False,
        crop_factor=1.5,
        min_size=100,
        fill_holes=False,
        text_prompt="",
        sam3_selectors_pipe=None,
        mask_prompt=None,
        max_detections=50,
    ):
        actual_max_detections = -1 if detect_all else max_detections
        batch_size, height, width, _ = image.shape
        positive_boxes, negative_boxes, positive_points, negative_points = (
            prompt_handler.extract_by_mode(
                sam3_selectors_pipe,
                pipeline_mode,
            )
        )

        print(
            f"[SAM3] Batch size: {batch_size}, pipeline_mode: {pipeline_mode}, instances: {instances}"
        )

        all_detection_masks = []
        all_combined_masks = []
        all_vis_tensors = []
        all_segs = []

        ensure_model_on_device(sam3_model)
        processor = sam3_model["processor"]
        processor.set_confidence_threshold(confidence_threshold)

        all_boxes, all_box_labels = prompt_handler.aggregate_prompts(
            positive_boxes, negative_boxes, "boxes"
        )
        all_points, all_point_labels = prompt_handler.aggregate_prompts(
            positive_points, negative_points, "points"
        )

        for b in range(batch_size):
            print(f"[SAM3] Frame {b + 1}/{batch_size}")
            img_tensor = image[b : b + 1]
            pil_image = comfy_image_to_pil(img_tensor)
            pil_image_single = (
                pil_image[0] if isinstance(pil_image, list) else pil_image
            )
            state = processor.set_image(pil_image_single)

            if text_prompt and text_prompt.strip():
                state = processor.set_text_prompt(text_prompt.strip(), state)
            if all_boxes:
                state = processor.add_multiple_box_prompts(
                    all_boxes, all_box_labels, state
                )
            if all_points:
                state = processor.add_point_prompt(all_points, all_point_labels, state)

            current_mask_prompt = None
            if mask_prompt is not None:
                if isinstance(mask_prompt, torch.Tensor):
                    if mask_prompt.dim() == 3 and mask_prompt.shape[0] == batch_size:
                        current_mask_prompt = mask_prompt[b]
                    else:
                        current_mask_prompt = mask_prompt
                else:
                    current_mask_prompt = mask_prompt
                if current_mask_prompt is not None:
                    if not isinstance(current_mask_prompt, torch.Tensor):
                        current_mask_prompt = torch.from_numpy(current_mask_prompt)
                    state = processor.add_mask_prompt(
                        current_mask_prompt.to(sam3_model["device"]), state
                    )

            masks = state.get("masks", None)
            boxes = state.get("boxes", None)
            scores = state.get("scores", None)

            if (
                masks is not None
                and isinstance(masks, torch.Tensor)
                and masks.numel() > 0
                and min_size > 1
            ):
                masks, boxes, scores = mask_filters.filter_by_size(
                    masks, boxes, scores, min_size
                )

            if masks is None or (
                isinstance(masks, torch.Tensor) and masks.numel() == 0
            ):
                print(f"[SAM3] No detections for frame {b}")
                all_combined_masks.append(torch.zeros(1, height, width, device="cpu"))
                all_vis_tensors.append(pil_to_comfy_image(pil_image_single))
                continue

            if instances and boxes is not None and isinstance(boxes, torch.Tensor):
                masks, boxes, scores = mask_filters.filter_by_instances(
                    masks,
                    boxes,
                    scores,
                    positive_boxes,
                    positive_points,
                    width,
                    height,
                    iou_threshold=0.1,
                )

            if masks is None or (
                isinstance(masks, torch.Tensor) and masks.numel() == 0
            ):
                print(f"[SAM3] No detections for frame {b}")
                all_combined_masks.append(torch.zeros(1, height, width, device="cpu"))
                all_vis_tensors.append(pil_to_comfy_image(pil_image_single))
                continue

            masks, boxes, scores = mask_filters.limit_detections(
                masks, boxes, scores, actual_max_detections
            )

            if fill_holes and isinstance(masks, torch.Tensor) and masks.numel() > 0:
                masks = mask_ops.apply_per_mask(masks, mask_ops.fill_holes)

            frame_detection_masks = masks_to_comfy_mask(masks)
            all_detection_masks.append(frame_detection_masks)

            combined_tensor = mask_ops.build_combined_mask(masks)
            all_combined_masks.append(combined_tensor)

            segs = build_contour_segs(
                masks, text_prompt, width, height, crop_factor=crop_factor
            )
            all_segs.append(segs)

            vis_image = visualize_masks_on_image(
                pil_image_single, masks, boxes, scores, alpha=0.5
            )
            all_vis_tensors.append(pil_to_comfy_image(vis_image))

        batch_masks = (
            torch.cat(all_detection_masks, dim=0)
            if all_detection_masks
            else torch.zeros(0, height, width, device="cpu")
        )
        combined_mask = torch.cat(all_combined_masks, dim=0)
        visualization = torch.cat(all_vis_tensors, dim=0)

        if all_segs:
            shape_info = all_segs[0][0]
            merged_seg_list = []
            for _, seg_list_f in all_segs:
                merged_seg_list.extend(seg_list_f)
            final_segs = (shape_info, merged_seg_list)
        else:
            final_segs = ((height, width), [])

        print(
            f"[SAM3] Complete: {len(final_segs[1])} SEGS, {batch_masks.shape[0]} detection masks, "
            f"{combined_mask.shape[0]} frames"
        )

        offload_model_if_needed(sam3_model)
        return (final_segs, visualization, combined_mask, batch_masks)


class TBGSAM3PromptCollector:
    """
    Unified SAM3 Prompt Collector - collects points and boxes in single node
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Image for interactive selection. Use B to toggle Point/Box. Left=Positive, Right/Shift=Negative."
                    },
                ),
                "positive_points": ("STRING", {"default": "[]", "multiline": False}),
                "negative_points": ("STRING", {"default": "[]", "multiline": False}),
                "positive_boxes": ("STRING", {"default": "[]", "multiline": False}),
                "negative_boxes": ("STRING", {"default": "[]", "multiline": False}),
            }
        }

    RETURN_TYPES = ("SAM3_PROMPT_PIPELINE",)
    RETURN_NAMES = ("sam3_selectors_pipe",)
    FUNCTION = "collect_pipeline"
    CATEGORY = "TBG/SAM3"
    OUTPUT_NODE = True

    def collect_pipeline(
        self, image, positive_points, negative_points, positive_boxes, negative_boxes
    ):
        # Parse JSON inputs
        try:
            pos_pts = json.loads(positive_points) if positive_points else []
            neg_pts = json.loads(negative_points) if negative_points else []
            pos_bxs = json.loads(positive_boxes) if positive_boxes else []
            neg_bxs = json.loads(negative_boxes) if negative_boxes else []
        except Exception:
            pos_pts, neg_pts, pos_bxs, neg_bxs = [], [], [], []

        print(
            f"[TBGSAM3PromptCollector] Points: +{len(pos_pts)} -{len(neg_pts)}, Boxes: +{len(pos_bxs)} -{len(neg_bxs)}"
        )

        # Get image dimensions
        img_height, img_width = image.shape[1], image.shape[2]

        pipeline = {
            "positive_points": None,
            "negative_points": None,
            "positive_boxes": None,
            "negative_boxes": None,
        }

        def normalize_points(pts):
            return [[p["x"] / img_width, p["y"] / img_height] for p in pts]

        if pos_pts:
            pipeline["positive_points"] = {
                "points": normalize_points(pos_pts),
                "labels": [1] * len(pos_pts),
            }

        if neg_pts:
            pipeline["negative_points"] = {
                "points": normalize_points(neg_pts),
                "labels": [0] * len(neg_pts),
            }

        def convert_boxes(boxes):
            converted = []
            for b in boxes:
                x1 = b["x1"] / img_width
                y1 = b["y1"] / img_height
                x2 = b["x2"] / img_width
                y2 = b["y2"] / img_height
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                converted.append([cx, cy, w, h])
            return converted

        if pos_bxs:
            pipeline["positive_boxes"] = {
                "boxes": convert_boxes(pos_bxs),
                "labels": [True] * len(pos_bxs),
            }

        if neg_bxs:
            pipeline["negative_boxes"] = {
                "boxes": convert_boxes(neg_bxs),
                "labels": [False] * len(neg_bxs),
            }

        # Convert image to base64 string for widget background
        img_tensor = image[0]
        if isinstance(img_tensor, torch.Tensor):
            img_array = img_tensor.detach().cpu().numpy()
        else:
            img_array = np.asarray(img_tensor)
        img_array = np.clip(img_array, 0.0, 1.0)
        img_array = (img_array * 255).astype(np.uint8)

        pil_img = Image.fromarray(img_array)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=75)
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "ui": {"bg_image": [img_base64]},
            "bg_image": [img_base64],
            "result": (pipeline,),
        }


class TBGSAM3DepthMap:
    """Generate depth maps for images or segments"""

    def __init__(self):
        self.depth_estimator = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["full_image", "per_segment"], {"default": "full_image"}),
                "normalize": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "segs": ("SEGS",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("depth_image", "depth_mask")
    FUNCTION = "generate_depth"
    CATEGORY = "SAM3"
    DESCRIPTION = "Generate depth maps. per_segment mode requires SEGS input."

    def generate_depth(
        self,
        image: torch.Tensor,
        mode: str,
        normalize: bool = True,
        segs: tuple | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate depth map"""
        try:
            # Initialize depth estimator
            if self.depth_estimator is None:
                self.depth_estimator = DepthEstimator()

            # Handle batch
            if len(image.shape) == 4:
                batch_size = image.shape[0]
                images = image
            else:
                batch_size = 1
                images = image.unsqueeze(0)

            all_depth_images = []
            all_depth_masks = []

            # Process each image in batch
            for batch_idx in range(batch_size):
                img = images[batch_idx]
                h, w, c = img.shape

                if mode == "full_image":
                    depth_map = self.depth_estimator.estimate_depth(img)

                else:  # per_segment
                    if segs is None:
                        raise ValueError("SEGS required for per_segment mode")

                    (img_h, img_w), segs_list = segs

                    if len(segs_list) == 0:
                        depth_map = self.depth_estimator.estimate_depth(img)
                    else:
                        depth_map = torch.zeros((h, w), dtype=torch.float32)

                        for seg in segs_list:
                            x, y, crop_w, crop_h = seg.crop_region
                            cropped_mask = seg.cropped_mask
                            bbox = seg.bbox
                            label = seg.label
                            confidence = seg.confidence

                            # Create full mask
                            full_mask = torch.zeros((h, w), dtype=torch.float32)

                            # Resize cropped mask
                            if cropped_mask.shape != (crop_h, crop_w):
                                resized_mask = torch.nn.functional.interpolate(
                                    cropped_mask.unsqueeze(0).unsqueeze(0),
                                    size=(crop_h, crop_w),
                                    mode="nearest",
                                ).squeeze()
                            else:
                                resized_mask = cropped_mask

                            # Place mask
                            end_y = min(y + crop_h, h)
                            end_x = min(x + crop_w, w)
                            actual_h = end_y - y
                            actual_w = end_x - x

                            full_mask[y:end_y, x:end_x] = resized_mask[
                                :actual_h, :actual_w
                            ]

                            # Generate depth for segment
                            seg_depth = self.depth_estimator.estimate_depth(
                                img, full_mask
                            )
                            depth_map = torch.maximum(depth_map, seg_depth)

                # Normalize
                if normalize:
                    depth_min = depth_map.min()
                    depth_max = depth_map.max()
                    if depth_max > depth_min:
                        depth_map = (depth_map - depth_min) / (depth_max - depth_min)

                # Convert to image [H, W, C]
                depth_image = depth_map.unsqueeze(-1).repeat(1, 1, 3)
                all_depth_images.append(depth_image)

                # Also as mask [H, W]
                all_depth_masks.append(depth_map)

            # Stack results
            if batch_size == 1:
                final_depth_image = all_depth_images[0].unsqueeze(0)  # [1, H, W, C]
                final_depth_mask = all_depth_masks[0].unsqueeze(0)  # [1, H, W]
            else:
                final_depth_image = torch.stack(all_depth_images, dim=0)  # [B, H, W, C]
                final_depth_mask = torch.stack(all_depth_masks, dim=0)  # [B, H, W]

            return (final_depth_image, final_depth_mask)

        except Exception as e:
            error_msg = f"Depth Generation Error:\n{str(e)}"
            print(f"[SAM3] ERROR: {error_msg}")
            raise RuntimeError(error_msg)


class TBGAttachConditioningToSEGS:
    """Attach per-SEG positive conditioning via control_net_wrapper.

    Accepts prompts as either:
    - A multiline string (one line per SEG, manual input)
    - A list of strings (one per SEG, from Florence2 pipeline via OUTPUT_IS_LIST)

    INPUT_IS_LIST absorbs upstream list cascades (e.g. SEGSPreview →
    Florence2Run) so this node and the Detailer downstream run once,
    not once-per-list-item.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segs": ("SEGS",),
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "mode": (["replace", "concat"],),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"
    CATEGORY = "TBG-SAM3"

    def doit(self, segs, clip, prompt, mode):
        # INPUT_IS_LIST wraps every input in a list.
        # Non-list inputs arrive as [value]; list-cascaded inputs as [v1, v2, …].
        actual_segs = segs[0]
        actual_clip = clip[0]
        actual_mode = mode[0]

        # prompt is [str1, str2, …] from Florence pipeline,
        # or ["line1\nline2\n…"] from manual multiline input.
        if len(prompt) == 1:
            lines = [l.strip() for l in prompt[0].split("\n") if l.strip()]
        else:
            lines = [p.strip() for p in prompt if p.strip()]

        shape, seg_list = actual_segs
        new_segs = []

        for i, seg in enumerate(seg_list):
            if i < len(lines):
                tokens = actual_clip.tokenize(lines[i])
                cond, pooled = actual_clip.encode_from_tokens(
                    tokens,
                    return_pooled=True,
                )
                conditioning = [[cond, {"pooled_output": pooled}]]

                wrapper = ConditioningOverrideWrapper(
                    conditioning=conditioning,
                    mode=actual_mode,
                    original_wrapper=seg.control_net_wrapper,
                )

                new_seg = SEG(
                    seg.cropped_image,
                    seg.cropped_mask,
                    seg.confidence,
                    seg.crop_region,
                    seg.bbox,
                    seg.label,
                    wrapper,
                )
            else:
                new_seg = seg

            new_segs.append(new_seg)

        return ((shape, new_segs),)


class TBGFlorence2SEGSCaptioner:
    """Caption each SEG with Florence2, CLIP-encode, and bake conditioning in.

    Replaces the multi-node chain:
        SEGSPreview → Florence2Run → StringListToString → AttachConditioning

    For each SEG the node crops the original image to the SEG's region,
    masks out non-segment pixels (so Florence2 sees only the object),
    generates a caption, optionally combines it with a user prompt,
    CLIP-encodes the result, and attaches it as per-SEG conditioning
    via ``control_net_wrapper``.

    Requires kijai's ComfyUI-Florence2 extension for the FL2MODEL type.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "segs": ("SEGS",),
                "florence2_model": ("FL2MODEL",),
                "clip": ("CLIP",),
                "task": (TASK_LIST, {"default": "caption"}),
                "prompt_mode": (
                    ["prepend", "append", "generated only", "user only"],
                    {"default": "prepend"},
                ),
                "conditioning_mode": (["replace", "concat"],),
            },
            "optional": {
                "text_input": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": False},
                ),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "num_beams": ("INT", {"default": 3, "min": 1, "max": 64}),
                "do_sample": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("SEGS", "STRING", "IMAGE")
    RETURN_NAMES = ("segs", "captions", "bbox_preview")
    FUNCTION = "doit"
    CATEGORY = "TBG-SAM3"

    def doit(
        self,
        image,
        segs,
        florence2_model,
        clip,
        task,
        prompt_mode,
        conditioning_mode,
        text_input="",
        keep_model_loaded=False,
        max_new_tokens=1024,
        num_beams=3,
        do_sample=True,
        seed=None,
    ):
        import comfy.model_management as mm

        shape, seg_list = segs

        if not seg_list:
            empty_img = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return ((shape, []), "", empty_img)

        # --- Device management (matches kijai's Florence2Run pattern) ---
        fl2_model = florence2_model["model"]
        processor = florence2_model["processor"]
        dtype = florence2_model["dtype"]
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        fl2_model.to(device)

        new_segs: list = []
        captions: list[str] = []
        bbox_crops: list = []

        try:
            for seg in seg_list:
                # --- Crop original image to tight bbox ---
                pil_crop = crop_seg_bbox(image=image, bbox=seg.bbox)
                bbox_crops.append(pil_crop)

                # --- Florence2 captioning ---
                generated = caption_image(
                    model=fl2_model,
                    processor=processor,
                    dtype=dtype,
                    pil_image=pil_crop,
                    task=task,
                    device=device,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    seed=seed,
                )

                # --- Combine with user prompt ---
                final_prompt = build_caption(
                    generated=generated,
                    user_prompt=text_input,
                    prompt_mode=prompt_mode,
                )
                captions.append(final_prompt)

                # --- CLIP encode → conditioning ---
                tokens = clip.tokenize(final_prompt)
                cond, pooled = clip.encode_from_tokens(
                    tokens,
                    return_pooled=True,
                )
                conditioning = [[cond, {"pooled_output": pooled}]]

                # --- Wrap and attach to SEG ---
                wrapper = ConditioningOverrideWrapper(
                    conditioning=conditioning,
                    mode=conditioning_mode,
                    original_wrapper=seg.control_net_wrapper,
                )

                new_segs.append(
                    SEG(
                        seg.cropped_image,
                        seg.cropped_mask,
                        seg.confidence,
                        seg.crop_region,
                        seg.bbox,
                        seg.label,
                        wrapper,
                    )
                )
        finally:
            # --- Offload Florence2 model ---
            if not keep_model_loaded:
                fl2_model.to(offload_device)
                mm.soft_empty_cache()

        all_captions = "\n".join(captions)
        preview_tensor = bbox_crops_to_tensor(bbox_crops)
        return ((shape, new_segs), all_captions, preview_tensor)


NODE_CLASS_MAPPINGS = {
    "TBGLoadSAM3Model": TBGLoadSAM3Model,  # your simple loader
    "TBGSAM3ModelLoaderAdvanced": TBGSAM3ModelLoaderAndDownloader,  # new advanced loader
    "TBGSam3Segmentation": TBGSam3Segmentation,
    "TBGSam3SegmentationBatch": TBGSam3SegmentationBatch,
    "TBGSAM3PromptCollector": TBGSAM3PromptCollector,
    "TBGSAM3DepthMap": TBGSAM3DepthMap,
    "TBGAttachConditioningToSEGS": TBGAttachConditioningToSEGS,
    "TBGFlorence2SEGSCaptioner": TBGFlorence2SEGSCaptioner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TBGLoadSAM3Model": "TBG SAM3 Model Loader",
    "TBGSAM3ModelLoaderAdvanced": "TBG SAM3 Model Loader and Downloader",
    "TBGSam3Segmentation": "TBG SAM3 Segmentation",
    "TBGSAM3PromptCollector": "TBG SAM3 Selector",
    "TBGSam3SegmentationBatch": "TBG SAM3 Batch Selector",
    "TBGSAM3DepthMap": "TBG SAM3 Depth Map",
    "TBGAttachConditioningToSEGS": "TBG Attach Conditioning to SEGS",
    "TBGFlorence2SEGSCaptioner": "TBG Florence2 SEGS Captioner",
}
