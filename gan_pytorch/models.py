# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
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

__all__ = [
    "Discriminator", "Generator", "discriminator",
    "mnist", "tfd", "cifar10"
]

model_urls = {
    "mnist": "https://github.com/Lornatang/GAN-PyTorch/releases/download/0.1.2/mnist-5539a1a7.pth",
    "tfd": "https://github.com/Lornatang/GAN-PyTorch/releases/download/0.1.2/tfd-4e44e2ca.pth",
    "cifar10": "https://github.com/Lornatang/GAN-PyTorch/releases/download/0.1.2/cifar10-5f3f1de6.pth"
}


class Discriminator(nn.Module):
    r""" An Discriminator model.

    `Generative Adversarial Networks model architecture from the One weird trick...
    <https://arxiv.org/abs/1406.2661>`_ paper.
    """

    def __init__(self, image_size: int = 28, channels: int = 1):
        """
        Args:
            image_size (int): The size of the image. (Default: 28).
            channels (int): The channels of the image. (Default: 1).
        """
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(channels * image_size * image_size, 512),
            nn.LeakyReLU(0.2, True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, True),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r""" Defines the computation performed at every call.

        Args:
          input (tensor): input tensor into the calculation.

        Returns:
          A four-dimensional vector (NCHW).
        """
        input = torch.flatten(input, 1)
        out = self.main(input)
        return out


class Generator(nn.Module):
    r""" An Generator model.

    `Generative Adversarial Networks model architecture from the One weird trick...
    <https://arxiv.org/abs/1406.2661>`_ paper.
    """

    def __init__(self, image_size: int = 28, channels: int = 1):
        """
        Args:
            image_size (int): The size of the image. (Default: 28).
            channels (int): The channels of the image. (Default: 1).
        """
        super(Generator, self).__init__()
        self.image_size = image_size
        self.channels = channels

        self.main = nn.Sequential(
            nn.Linear(100, 128),
            nn.LeakyReLU(0.2, True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, True),

            nn.Linear(1024, channels * image_size * image_size),
            nn.Tanh()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Defines the computation performed at every call.s

        Args:
            input (tensor): input tensor into the calculation.

        Returns:
            A four-dimensional vector (NCHW).
        """
        out = self.main(input)
        out = out.reshape(out.size(0), self.channels, self.image_size, self.image_size)
        return out


def _gan(arch, image_size, channels, pretrained, progress):
    model = Generator(image_size, channels)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], 
                                              progress=progress, 
                                              map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    return model


def discriminator(image_size, channels) -> Discriminator:
    r"""GAN model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1406.2661>`_ paper.
    """
    model = Discriminator(image_size, channels)
    return model


def mnist(pretrained: bool = False, progress: bool = True) -> Generator:
    r"""GAN model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1406.2661>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _gan("mnist", 28, 1, pretrained, progress)


def tfd(pretrained: bool = False, progress: bool = True) -> Generator:
    r"""GAN model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1406.2661>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _gan("tfd", 32, 1, pretrained, progress)


def cifar10(pretrained: bool = False, progress: bool = True) -> Generator:
    r"""GAN model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1406.2661>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _gan("cifar10", 32, 3, pretrained, progress)
