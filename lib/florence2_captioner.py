"""Florence2 captioning helpers for per-SEG image crops.

Pure logic — zero ComfyUI imports.  Only torch, numpy, PIL, stdlib.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch
from PIL import Image

# Only captioning / prompt-gen tasks — detection/segmentation/OCR tasks
# return structured data (bboxes, polygons) rather than useful text.
TASK_PROMPTS: dict[str, str] = {
    "caption": "<CAPTION>",
    "detailed_caption": "<DETAILED_CAPTION>",
    "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
    "prompt_gen_tags": "<GENERATE_TAGS>",
    "prompt_gen_mixed_caption": "<MIXED_CAPTION>",
    "prompt_gen_analyze": "<ANALYZE>",
    "prompt_gen_mixed_caption_plus": "<MIXED_CAPTION_PLUS>",
}

TASK_LIST: list[str] = list(TASK_PROMPTS.keys())


def hash_seed(seed: int) -> int:
    """SHA-256 hash → 32-bit int.  Matches kijai's Florence2Run approach."""
    seed_bytes = str(seed).encode("utf-8")
    return int(hashlib.sha256(seed_bytes).hexdigest(), 16) % (2**32)


def crop_and_mask_seg_image(
    image: torch.Tensor,
    crop_region: tuple[int, int, int, int],
    cropped_mask: np.ndarray | torch.Tensor,
) -> Image.Image:
    """Crop original image to *crop_region*, mask out non-segment pixels.

    Parameters
    ----------
    image:
        Full image tensor ``[B, H, W, C]`` or ``[H, W, C]``, float32 0-1.
    crop_region:
        ``(x1, y1, x2, y2)`` pixel coordinates (same space as the full image).
    cropped_mask:
        2-D mask aligned to *crop_region*, values in 0-1.
        Shape ``[crop_h, crop_w]`` or broadcastable.

    Returns
    -------
    PIL.Image.Image
        RGB crop with non-segment pixels set to black.
    """
    if image.ndim == 4:
        image = image[0]  # take first batch element

    x1, y1, x2, y2 = crop_region
    crop = image[y1:y2, x1:x2, :]  # [crop_h, crop_w, C]

    # Ensure mask is a numpy array, float32, 2-D
    if isinstance(cropped_mask, torch.Tensor):
        mask_np = cropped_mask.detach().cpu().float().numpy()
    else:
        mask_np = np.asarray(cropped_mask, dtype=np.float32)

    if mask_np.ndim == 3:
        mask_np = mask_np.squeeze(0) if mask_np.shape[0] == 1 else mask_np[:, :, 0]

    crop_np = crop.detach().cpu().float().numpy()  # [crop_h, crop_w, C], 0-1

    # Resize mask to match crop if dimensions differ (safety net)
    ch, cw = crop_np.shape[:2]
    mh, mw = mask_np.shape[:2]
    if (mh, mw) != (ch, cw):
        from PIL import Image as _PILImage

        mask_pil = _PILImage.fromarray((mask_np * 255).astype(np.uint8), mode="L")
        mask_pil = mask_pil.resize((cw, ch), _PILImage.Resampling.BILINEAR)
        mask_np = np.asarray(mask_pil, dtype=np.float32) / 255.0

    # Apply mask: object pixels only, black elsewhere
    masked = crop_np * mask_np[:, :, np.newaxis]

    # Convert to uint8 PIL
    masked_uint8 = (np.clip(masked, 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(masked_uint8, mode="RGB")


def caption_image(
    *,
    model: Any,
    processor: Any,
    dtype: torch.dtype,
    pil_image: Image.Image,
    task: str,
    device: torch.device | str,
    max_new_tokens: int = 1024,
    num_beams: int = 3,
    do_sample: bool = True,
    seed: int | None = None,
) -> str:
    """Run Florence2 captioning on a single PIL image.

    Parameters
    ----------
    model / processor / dtype:
        From the FL2MODEL dict (``model['model']``, etc.).
    pil_image:
        The cropped+masked segment image.
    task:
        One of the keys in :data:`TASK_PROMPTS`.
    device:
        Torch device for inference.

    Returns
    -------
    str
        Generated caption text.
    """
    if seed is not None:
        from transformers import set_seed as _set_seed

        _set_seed(hash_seed(seed))

    task_prompt = TASK_PROMPTS[task]

    inputs = (
        processor(
            text=task_prompt,
            images=pil_image,
            return_tensors="pt",
            do_rescale=False,
        )
        .to(dtype)
        .to(device)
    )

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_beams=num_beams,
        use_cache=False,
    )

    result = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    # Strip special tokens
    clean = result.replace("</s>", "").replace("<s>", "").strip()

    # Post-process to extract the actual caption text from Florence2's
    # structured output (e.g. ``<CAPTION>the caption</CAPTION>``).
    W, H = pil_image.size
    parsed = processor.post_process_generation(
        result,
        task=task_prompt,
        image_size=(W, H),
    )
    # For captioning tasks the parsed dict has the task_prompt as key
    # mapping to a plain string.
    if task_prompt in parsed and isinstance(parsed[task_prompt], str):
        clean = parsed[task_prompt].strip()

    return clean


def build_caption(
    *,
    generated: str,
    user_prompt: str,
    prompt_mode: str,
) -> str:
    """Combine Florence2 caption with optional user prompt.

    Parameters
    ----------
    generated:
        Caption from Florence2.
    user_prompt:
        Optional text from the user.
    prompt_mode:
        ``"prepend"`` — ``"{user}, {generated}"``
        ``"append"``  — ``"{generated}, {user}"``
        ``"generated only"`` — ignore user text
        ``"user only"`` — ignore generated text (skip Florence2 result)

    Returns
    -------
    str
        Final prompt string for CLIP encoding.
    """
    gen = generated.strip()
    usr = user_prompt.strip()

    if prompt_mode == "generated only" or not usr:
        return gen
    if prompt_mode == "user only":
        return usr if usr else gen
    if prompt_mode == "prepend":
        return f"{usr}, {gen}" if gen else usr
    # append
    return f"{gen}, {usr}" if gen else usr
