import torch
import numpy as np
from typing import Tuple, Optional


class EventSurfaceGenerator:
    """
    Converts asynchronous event streams (x, y, t, p) into continuous 
    time-surface representations or discretized voxel grids for SNN processing.
    """

    def __init__(self, sensor_size: Tuple[int, int] = (128, 128), tau: float = 20000.0):
        self.height, self.width = sensor_size
        self.tau = tau

    def generate_time_surface(
        self, 
        events: torch.Tensor, 
        target_time: Optional[float] = None
    ) -> torch.Tensor:
        if events.numel() == 0:
            return torch.zeros((2, self.height, self.width), dtype=torch.float32)

        x = events[:, 0].long()
        y = events[:, 1].long()
        t = events[:, 2].float()
        p = events[:, 3].long()

        if target_time is None:
            target_time = t.max().item()

        p_idx = torch.where(p < 0, torch.tensor(0), torch.tensor(1))
        last_t = torch.zeros((2, self.height, self.width), dtype=torch.float32)

        for i in range(len(events)):
            xi, yi, ti, pi = x[i], y[i], t[i], p_idx[i]
            if 0 <= xi < self.width and 0 <= yi < self.height:
                if ti > last_t[pi, yi, xi]:
                    last_t[pi, yi, xi] = ti

        delta_t = target_time - last_t
        mask = (last_t > 0).float()
        surface = torch.exp(-delta_t / self.tau) * mask

        return surface