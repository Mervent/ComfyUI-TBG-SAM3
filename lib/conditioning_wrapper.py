"""Wrapper that overrides positive conditioning per-SEG via control_net_wrapper."""

from __future__ import annotations

from typing import Any


class ConditioningOverrideWrapper:
    """Drop-in for Impact Pack's control_net_wrapper interface.

    When the Detailer calls ``wrapper.apply(positive, negative, ...)``,
    this returns the stored per-SEG conditioning instead of the global
    positive, while optionally chaining with an existing ControlNet wrapper.
    """

    def __init__(
        self,
        conditioning: Any,
        original_wrapper: Any | None = None,
    ) -> None:
        self.conditioning = conditioning
        self.original_wrapper = original_wrapper
        self.control_image: Any | None = (
            getattr(original_wrapper, "control_image", None)
            if original_wrapper is not None
            else None
        )

    def apply(
        self,
        positive: Any,
        negative: Any,
        image: Any,
        noise_mask: Any = None,
    ) -> tuple[Any, Any, list]:
        """Replace positive, then chain original wrapper if present."""
        positive = self.conditioning

        if self.original_wrapper is not None:
            return self.original_wrapper.apply(positive, negative, image, noise_mask)

        return positive, negative, []

    def doit_ipadapter(self, model: Any) -> tuple[Any, list]:
        """Delegate IPAdapter to original wrapper if present."""
        if self.original_wrapper is not None:
            return self.original_wrapper.doit_ipadapter(model)
        return model, []
