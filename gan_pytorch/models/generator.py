# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

model_urls = {
    "gan": "https://github.com/Lornatang/GAN-PyTorch/releases/download/v0.2.0/GAN_MNIST-5c625d6ee2e8b3550f20080336ae81d7a2a8af93ea780a849d5b1dd4c428b133.pth",
}


class Generator(nn.Module):
    r""" It is mainly based on the mobile net network as the backbone network generator.

    Args:
        image_size (int): The size of the image. (Default: 28)
        channels (int): The channels of the image. (Default: 1)
    """

    def __init__(self, image_size: int = 28, channels: int = 1) -> None:
        super(Generator, self).__init__()
        self.image_size = image_size
        self.channels = channels

        self.main = nn.Sequential(
            nn.Linear(100, 128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(1024, channels * image_size * image_size),
            nn.Tanh()
        )

        # Initializing all neural network weights.
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.main(x)
        out = out.reshape(out.size(0), self.channels, self.image_size, self.image_size)

        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def _gan(arch: str, image_size: int, channels: int, pretrained: bool, progress: bool) -> Generator:
    r""" Used to create GAN model.

    Args:
        arch (str): GAN model architecture name.
        image_size (int): The size of the image.
        channels (int): The channels of the image.
        pretrained (bool): If True, returns a model pre-trained on MNIST.
        progress (bool): If True, displays a progress bar of the download to stderr.

    Returns:
        Generator model.
    """
    model = Generator(image_size, channels)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    return model


def gan(pretrained: bool = False, progress: bool = True) -> Generator:
    r"""GAN model architecture from the `"One weird trick..." <https://arxiv.org/abs/1406.2661>` paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    return _gan("gan", 28, 1, pretrained, progress)
