import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture
def square_mask_np() -> np.ndarray:
    """4x4 square mask centered in a 20x20 image."""
    mask = np.zeros((20, 20), dtype=np.float32)
    mask[8:12, 8:12] = 1.0
    return mask


@pytest.fixture
def mask_with_hole_np() -> np.ndarray:
    """10x10 filled square with a 2x2 hole in the center."""
    mask = np.zeros((20, 20), dtype=np.float32)
    mask[5:15, 5:15] = 1.0
    mask[9:11, 9:11] = 0.0
    return mask


@pytest.fixture
def masks_3d(square_mask_np, mask_with_hole_np) -> torch.Tensor:
    """Three stacked 20x20 masks as a (3, 20, 20) tensor."""
    mask_third = np.zeros((20, 20), dtype=np.float32)
    mask_third[1:6, 13:19] = 1.0
    stacked = np.stack([square_mask_np, mask_with_hole_np, mask_third], axis=0)
    return torch.from_numpy(stacked)


@pytest.fixture
def masks_4d(masks_3d) -> torch.Tensor:
    """Same masks as masks_3d but with channel dim: (3, 1, 20, 20)."""
    return masks_3d.unsqueeze(1)


@pytest.fixture
def sample_boxes() -> torch.Tensor:
    """Three detection boxes in absolute pixel coords [x1, y1, x2, y2]."""
    return torch.tensor(
        [[2.0, 2.0, 18.0, 18.0], [0.0, 0.0, 10.0, 10.0], [10.0, 10.0, 20.0, 20.0]],
        dtype=torch.float32,
    )


@pytest.fixture
def sample_scores() -> torch.Tensor:
    """Three confidence scores matching sample_boxes."""
    return torch.tensor([0.9, 0.7, 0.5], dtype=torch.float32)
