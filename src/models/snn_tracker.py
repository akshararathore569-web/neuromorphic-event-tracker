import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SNNTracker(nn.Module):
    """
    Spiking Neural Network with Leaky Integrate-and-Fire (LIF) neurons 
    for asynchronous event-surface processing and spatial target localization.
    """

    def __init__(self, beta: float = 0.5, num_steps: int = 10):
        super().__init__()
        self.num_steps = num_steps
        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Convolutional Spiking Backbone
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 64x64
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 32x32
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # Tracking Head (Predicted target coordinates: [center_x, center_y])
        self.fc = nn.Linear(64 * 32 * 32, 2)

    def forward(self, x_surface: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_surface: Time-surface tensor of shape (B, 2, 128, 128)
        Returns:
            torch.Tensor: Predicted spatial target coordinates (B, 2)
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk3_rec = []

        for _ in range(self.num_steps):
            cur1 = self.conv1(x_surface)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.conv3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk3_rec.append(spk3)

        spk3_stack = torch.stack(spk3_rec, dim=0).mean(dim=0)
        spk3_flat = spk3_stack.flatten(start_dim=1)
        coords = self.fc(spk3_flat)

        return coords