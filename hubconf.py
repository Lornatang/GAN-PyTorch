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
"""File for accessing GAN via PyTorch Hub https://pytorch.org/hub/
Usage:
    import torch
    model = torch.hub.load("Lornatang/GAN-PyTorch", "gan", pretrained=True, progress=True, verbose=False)
"""
import torch
from torch.hub import load_state_dict_from_url

from gan_pytorch.models.generator import Generator

model_urls = {
    "gan": "https://github.com/Lornatang/GAN-PyTorch/releases/download/v0.2.0/GAN_MNIST-5c625d6ee2e8b3550f20080336ae81d7a2a8af93ea780a849d5b1dd4c428b133.pth",
}

dependencies = ["torch"]


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
    model = _gan("gan", 28, 1, pretrained, progress)

    return model
