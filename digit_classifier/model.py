"""CNN по мотивам LeNet: ядра 5×5, Tanh и усредняющий pooling."""

import torch
from torch import nn

ARCHITECTURE = "lenet5_mnist_v2"


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2), nn.Tanh(), nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2),
            nn.Flatten(), nn.Linear(16 * 5 * 5, 120), nn.Tanh(),
            nn.Linear(120, 84), nn.Tanh(), nn.Linear(84, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)
