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
            nn.Linear(1 * 28 * 28, 512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # Initializing all neural network weights.
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        out = torch.flatten(x, 1)
        out = self.main(out)

        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()
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

            # Output: channels * image size * image size.
            nn.Linear(1024, 1 * 28 * 28),
            nn.Tanh()
        )

        # Init all layer weights.
        self._initialize_weights()

    # The tracking operator in the PyTorch model must be written like this.
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.main(x)
        out = out.reshape(out.size(0), 1, 28, 28)

        return out

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
