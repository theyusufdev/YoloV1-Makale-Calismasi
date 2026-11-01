import torch
from torch import nn

class YouOnlyLookOnceV1(nn.Module):
    def __init__(self, input_size, output_size = None) -> None:
        super().__init__()

        self.feature_extractor = nn.Sequential(

            nn.Conv2d(input_size, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(192, 128, kernel_size=1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 256, kernel_size=1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(1024, 512, kernel_size=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),

            nn.Dropout(p=0.5)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, 7 * 7 * 30),
        )

    def forward(self, x :torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.classifier(x)
        x = x.reshape(x.shape[0], 7, 7, 30)
        return x
