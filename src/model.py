import torch
import torch.nn as nn

class ImageRestorationCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.encoder(x)

        x = torch.nn.functional.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        x = self.decoder(x)

        return x