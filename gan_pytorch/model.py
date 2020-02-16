# Copyright 2020 Lorna Authors. All Rights Reserved.
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

from .utils import get_model_params
from .utils import load_pretrained_weights
from .utils import model_params


# Generative Adversarial Networks model architecture from the One weird trick...
# <https://arxiv.org/abs/1406.2661>`_ paper.
class Generator(nn.Module):
  r""" An Generator model. Most easily loaded with the .from_name or
      .from_pretrained methods

  Args:
    global_params (namedtuple): A set of GlobalParams shared between blocks

  Examples:
      >>> import torch
      >>> from gan_pytorch import Generator
      >>> from gan_pytorch import Discriminator
      >>> generator = Generator.from_pretrained("g-mnist")
      Loaded generator pretrained weights for `g-mnist`
      >>> discriminator = Discriminator.from_pretrained("g-mnist")
      Loaded discriminator pretrained weights for `d-mnist`
      >>> generator.eval()
      Generator(
        (main): Sequential(
          (0): Linear(in_features=100, out_features=128, bias=True)
          (1): LeakyReLU(negative_slope=0.2, inplace=True)
          (2): Linear(in_features=128, out_features=256, bias=True)
          (3): BatchNorm1d(256, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
          (4): LeakyReLU(negative_slope=0.2, inplace=True)
          (5): Linear(in_features=256, out_features=512, bias=True)
          (6): BatchNorm1d(512, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
          (7): LeakyReLU(negative_slope=0.2, inplace=True)
          (8): Linear(in_features=512, out_features=1024, bias=True)
          (9): BatchNorm1d(1024, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
          (10): LeakyReLU(negative_slope=0.2, inplace=True)
          (11): Linear(in_features=1024, out_features=784, bias=True)
          (12): Tanh()
        )
      )
      >>> discriminator.eval()
      Discriminator(
        (main): Sequential(
          (0): Linear(in_features=784, out_features=512, bias=True)
          (1): LeakyReLU(negative_slope=0.2, inplace=True)
          (2): Linear(in_features=512, out_features=256, bias=True)
          (3): LeakyReLU(negative_slope=0.2, inplace=True)
          (4): Linear(in_features=256, out_features=1, bias=True)
          (5): Sigmoid()
        )
      )
      >>> noise = torch.randn(1, 100)
      >>> discriminator(generator(noise)).item()
      0.11109194904565811
  """

  def __init__(self, global_params=None):
    super(Generator, self).__init__()
    self.noise = global_params.noise
    self.channels = global_params.channels
    self.image_size = global_params.image_size
    self.batch_norm_momentum = global_params.batch_norm_momentum
    self.relu_negative_slope = global_params.relu_negative_slope

    def block(in_features, out_features, normalize=True):
      r""" Define neuron module layer.

      Args:
        in_features (int): size of each input sample.
        out_features (int): size of each output sample.
        normalize (bool): If set to ``False``, the block will not add
                            an batch normalization method. Default: ``True``.

      Returns:
        Some neural model layers

      Examples:
        >>> block(6, 16, normalize=False)
        [Linear(in_features=6, out_features=16, bias=True),
        LeakyReLU(negative_slope=0.2, inplace=True)]
        >>> block(6, 16)
        [Linear(in_features=6, out_features=16, bias=True),
        BatchNorm1d(16, eps=0.8, momentum=0.1, affine=True, track_running_stats=True),
        LeakyReLU(negative_slope=0.2, inplace=True)]
      """
      layers = [nn.Linear(in_features, out_features)]
      if normalize:
        layers.append(nn.BatchNorm1d(out_features, self.batch_norm_momentum))
      layers.append(nn.LeakyReLU(self.relu_negative_slope, inplace=True))
      return layers

    self.main = nn.Sequential(
      *block(self.noise, 128, normalize=False),
      *block(128, 256),
      *block(256, 512),
      *block(512, 1024),
      nn.Linear(1024, self.channels * self.image_size * self.image_size),
      nn.Tanh()
    )

    # custom weights initialization called on netG and netD
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
      elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

  def forward(self, x):
    r"""Defines the computation performed at every call.

    Args:
      x (tensor): input tensor into the calculation.

    Returns:
      A four-dimensional vector (NCHW).
    """
    x = self.main(x)
    x = x.reshape(x.size(0), self.channels, self.image_size, self.image_size)
    return x

  @classmethod
  def from_name(cls, model_name):
    cls._check_model_name_is_valid(model_name)
    global_params = get_model_params(model_name)
    return cls(global_params)

  @classmethod
  def from_pretrained(cls, model_name):
    model = cls.from_name(model_name, )
    load_pretrained_weights(model, model_name)
    return model

  @classmethod
  def get_image_size(cls, model_name):
    cls._check_model_name_is_valid(model_name)
    _, res = model_params(model_name)
    return res

  @classmethod
  def _check_model_name_is_valid(cls, model_name):
    """ Validates model name. """
    valid_list = ["mnist", "fmnist"]
    valid_models = ["g-" + str(i) for i in valid_list]
    if model_name not in valid_models:
      raise ValueError("model_name should be one of: " + ", ".join(valid_models))


class Discriminator(nn.Module):
  r""" An Discriminator model. Most easily loaded with the .from_name or
      .from_pretrained methods

  Args:
    global_params (namedtuple): A set of GlobalParams shared between blocks

  Examples:
    >>> import torch
    >>> from gan_pytorch import Discriminator
    >>> discriminator = Discriminator.from_pretrained("d-mnist")
    Loaded discriminator pretrained weights for `d-mnist`
    >>> discriminator.eval()
    Discriminator(
      (main): Sequential(
        (0): Linear(in_features=784, out_features=512, bias=True)
        (1): LeakyReLU(negative_slope=0.2, inplace=True)
        (2): Linear(in_features=512, out_features=256, bias=True)
        (3): LeakyReLU(negative_slope=0.2, inplace=True)
        (4): Linear(in_features=256, out_features=1, bias=True)
        (5): Sigmoid()
      )
    )
    >>> noise = torch.randn(1, 784)
    >>> discriminator(noise).item()
    0.00048593798419460654
  """

  def __init__(self, global_params=None):
    super(Discriminator, self).__init__()
    self.channels = global_params.channels
    self.image_size = global_params.image_size
    self.relu_negative_slope = global_params.relu_negative_slope

    def block(in_features, out_features):
      r""" Define neuron module layer.

      Args:
        in_features (int): size of each input sample.
        out_features (int): size of each output sample.

      Returns:
        Some neural model layers

      Examples:
        >>> block(6, 16)
        [Linear(in_features=6, out_features=16, bias=True),
        LeakyReLU(negative_slope=0.2, inplace=True)]
      """
      layers = [nn.Linear(in_features, out_features),
                nn.LeakyReLU(self.relu_negative_slope, inplace=True)]
      return layers

    self.main = nn.Sequential(
      *block(self.channels * self.image_size * self.image_size, 512),
      *block(512, 256),
      nn.Linear(256, 1),
      nn.Sigmoid()
    )

    # custom weights initialization called on netG and netD
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
      elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

  def forward(self, x):
    r""" Defines the computation performed at every call.

    Args:
      x (tensor): input tensor into the calculation.

    Returns:
      A four-dimensional vector (NCHW).
    """
    x = torch.flatten(x, 1)
    x = self.main(x)
    return x

  @classmethod
  def from_name(cls, model_name):
    cls._check_model_name_is_valid(model_name)
    global_params = get_model_params(model_name)
    return cls(global_params)

  @classmethod
  def from_pretrained(cls, model_name):
    model = cls.from_name(model_name)
    load_pretrained_weights(model, model_name)
    return model

  @classmethod
  def get_image_size(cls, model_name):
    cls._check_model_name_is_valid(model_name)
    _, res = model_params(model_name)
    return res

  @classmethod
  def _check_model_name_is_valid(cls, model_name):
    """ Validates model name. """
    valid_list = ["mnist", "fmnist"]
    valid_models = ["d-" + str(i) for i in valid_list]
    if model_name not in valid_models:
      raise ValueError("model_name should be one of: " + ", ".join(valid_models))
