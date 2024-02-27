import torch.nn as nn


class GrayscaleLayer(nn.Module):
    def __init__(self) -> None:
        super(GrayscaleLayer, self).__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x
