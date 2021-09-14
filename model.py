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

# ==============================================================================
# File description: Realize the model definition function.
# ==============================================================================
import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "Discriminator", "Generator"
]


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: channels * image size * image size.
            nn.Linear(1 * 28 * 28, 512, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 256, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        out = torch.flatten(x, 1)
        out = self.main(out)

        return out


class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 128, bias=True),
            nn.LeakyReLU(0.2, True),

            nn.Linear(128, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True),

            nn.Linear(256, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),

            nn.Linear(512, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, True),

            # Output: channels * image size * image size.
            nn.Linear(1024, 1 * 28 * 28, bias=True),
            nn.Tanh()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support PyTorch.script function.
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.main(x)
        out = out.reshape(out.size(0), 1, 28, 28)

        return out
