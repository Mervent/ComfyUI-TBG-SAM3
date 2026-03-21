"""Wrapper that overrides positive conditioning per-SEG via control_net_wrapper."""

from __future__ import annotations

from typing import Any

import torch


class ConditioningOverrideWrapper:
    """Drop-in for Impact Pack's control_net_wrapper interface.

    When the Detailer calls ``wrapper.apply(positive, negative, ...)``,
    this either replaces or concatenates the stored per-SEG conditioning
    with the global positive, while optionally chaining with an existing
    ControlNet wrapper.
    """

    def __init__(
        self,
        conditioning: Any,
        mode: str = "replace",
        original_wrapper: Any | None = None,
    ) -> None:
        self.conditioning = conditioning
        self.mode = mode
        self.original_wrapper = original_wrapper
        self.control_image: Any | None = (
            getattr(original_wrapper, "control_image", None)
            if original_wrapper is not None
            else None
        )

    def _concat_conditioning(
        self,
        base: list,
        added: list,
    ) -> list:
        """Concatenate added conditioning tokens onto base (like ConditioningConcat)."""
        cond_from = added[0][0]
        out = []
        for t in base:
            tw = torch.cat((t[0], cond_from), 1)
            n = [tw, t[1].copy()]
            out.append(n)
        return out

    def apply(
        self,
        positive: Any,
        negative: Any,
        image: Any,
        noise_mask: Any = None,
    ) -> tuple[Any, Any, list]:
        """Replace or concat positive, then chain original wrapper if present."""
        if self.mode == "concat":
            positive = self._concat_conditioning(
                base=positive,
                added=self.conditioning,
            )
        else:
            positive = self.conditioning

        if self.original_wrapper is not None:
            return self.original_wrapper.apply(positive, negative, image, noise_mask)

        return positive, negative, []

    def doit_ipadapter(self, model: Any) -> tuple[Any, list]:
        """Delegate IPAdapter to original wrapper if present."""
        if self.original_wrapper is not None:
            return self.original_wrapper.doit_ipadapter(model)
        return model, []
