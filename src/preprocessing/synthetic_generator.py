import torch
import numpy as np
from typing import Tuple, List


class SyntheticTargetGenerator:
    """
    Generates synthetic asynchronous event streams (x, y, t, p) 
    for moving geometric targets across a simulated DVS focal plane.
    """

    def __init__(self, sensor_size: Tuple[int, int] = (128, 128)):
        self.height, self.width = sensor_size

    def generate_ball_trajectory(
        self, 
        duration_us: float = 100000.0, 
        radius: int = 5, 
        event_density: float = 0.3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a moving circular target across the frame.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - events: Tensor of shape (N, 4) with [x, y, t, p]
                - target_center: Final (x, y) target center coordinates
        """
        events = []
        steps = int(duration_us / 1000)  # 1ms micro-steps

        # Linear motion path across sensor
        start_pos = np.array([20.0, 20.0])
        end_pos = np.array([100.0, 100.0])

        for step in range(steps):
            t = float(step * 1000)
            alpha = step / steps
            center = (1 - alpha) * start_pos + alpha * end_pos

            # Generate events around target perimeter
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx**2 + dy**2 <= radius**2:
                        if np.random.rand() < event_density:
                            px = int(center[0] + dx)
                            py = int(center[1] + dy)
                            if 0 <= px < self.width and 0 <= py < self.height:
                                p = 1 if np.random.rand() > 0.5 else -1
                                events.append([px, py, t, p])

        events_tensor = torch.tensor(events, dtype=torch.float32)
        final_center = torch.tensor(end_pos, dtype=torch.float32)

        return events_tensor, final_center