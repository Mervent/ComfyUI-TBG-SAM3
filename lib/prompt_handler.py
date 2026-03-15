from typing import Any


def valid_block(block: Any, key: str) -> bool:
    """Check if block is a dict containing a truthy value for key."""
    return isinstance(block, dict) and key in block and bool(block[key])


def extract_by_mode(
    pipeline_data: dict | None,
    mode: str,
) -> tuple[Any, Any, Any, Any]:
    """Extract prompt blocks from pipeline dict filtered by mode."""
    positive_boxes = None
    negative_boxes = None
    positive_points = None
    negative_points = None

    if pipeline_data is not None and mode != "disabled":
        if not isinstance(pipeline_data, dict):
            raise ValueError(
                f"sam3_selectors_pipe must be a dictionary, got {type(pipeline_data)}"
            )

        pipeline_positive_boxes = pipeline_data.get("positive_boxes", None)
        pipeline_negative_boxes = pipeline_data.get("negative_boxes", None)
        pipeline_positive_points = pipeline_data.get("positive_points", None)
        pipeline_negative_points = pipeline_data.get("negative_points", None)

        if mode == "all":
            positive_boxes = pipeline_positive_boxes
            negative_boxes = pipeline_negative_boxes
            positive_points = pipeline_positive_points
            negative_points = pipeline_negative_points
        elif mode == "boxes_only":
            positive_boxes = pipeline_positive_boxes
            negative_boxes = pipeline_negative_boxes
        elif mode == "points_only":
            positive_points = pipeline_positive_points
            negative_points = pipeline_negative_points
        elif mode == "positive_only":
            positive_boxes = pipeline_positive_boxes
            positive_points = pipeline_positive_points
        elif mode == "negative_only":
            negative_boxes = pipeline_negative_boxes
            negative_points = pipeline_negative_points

    return positive_boxes, negative_boxes, positive_points, negative_points


def aggregate_prompts(
    positive_block: dict | None,
    negative_block: dict | None,
    prompt_type: str,
) -> tuple[list, list]:
    """Merge positive and negative prompt blocks into flat lists of prompts and labels."""
    if prompt_type not in ("boxes", "points"):
        raise ValueError(f"Unsupported prompt_type: {prompt_type}")

    all_prompts: list = []
    all_labels: list = []

    if valid_block(positive_block, prompt_type):
        all_prompts.extend(positive_block[prompt_type])
        all_labels.extend(
            positive_block.get("labels", [1] * len(positive_block[prompt_type]))
        )

    if valid_block(negative_block, prompt_type):
        all_prompts.extend(negative_block[prompt_type])
        all_labels.extend(
            negative_block.get("labels", [0] * len(negative_block[prompt_type]))
        )

    return all_prompts, all_labels
