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

import collections
import ssl

import torch.utils.model_zoo as model_zoo

# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple("GlobalParams", [
  "noise", "channels", "image_size",
  "batch_norm_momentum", "relu_negative_slope"
])

# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)


# custom weights initialization called on netG and netD
def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    m.weight.data.normal_(0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    m.weight.data.normal_(1.0, 0.02)
    m.bias.data.fill_(0)


########################################################################
############## HELPERS FUNCTIONS FOR LOADING MODEL PARAMS ##############
########################################################################


def model_params(model_name):
  r""" Map Generator and Discriminator model name to parameter coefficients.
    
    Args:
        model_name (string): The name of the model corresponding to the dataset.

    Return:
        A binary tuple value.

    """
  params_dict = {
    # Coefficients: channels, image_size
    "g-mnist": (1, 28),
    "g-fmnist": (1, 28),
    "d-mnist": (1, 28),
    "d-fmnist": (1, 28),
  }
  return params_dict[model_name]


def model(channels=None, image_size=None):
  r""" Gets the parameters of the model
    
    Args:
        channels (int): size of each input image channels.
        image_size (int): size of each input image size. 

    Return:
        A set of GlobalParams shared between blocks.

    """
  global_params = GlobalParams(
    noise=100,
    batch_norm_momentum=0.8,
    relu_negative_slope=0.2,
    channels=channels,
    image_size=image_size,
  )

  return global_params


def get_model_params(model_name):
  """ Get the block args and global params for a given model
    
    Args:
        model_name (string): The name of the model corresponding to the dataset.

    Return:
        A set of GlobalParams shared between blocks.

    """
  if model_name.startswith("g") or model_name.startswith("d"):
    c, s = model_params(model_name)
    global_params = model(channels=c, image_size=s)
  else:
    raise NotImplementedError(f"model name is not pre-defined: {model_name}.")
  return global_params


url_maps = {
  "g-mnist": "https://github.com/changyu98/models/raw/master/pytorch/gan/g-mnist-bd2658ac.pth",
  "g-fmnist": "https://github.com/changyu98/models/raw/master/pytorch/gan/g-fmnist-55b38893.pth",
  "d-mnist": "https://github.com/changyu98/models/raw/master/pytorch/gan/d-mnist-201c43ef.pth",
  "d-fmnist": "https://github.com/changyu98/models/raw/master/pytorch/gan/d-fmnist-28695d56.pth",
}


def load_pretrained_weights(model_arch, model_name):
  """ Loads pretrained weights, and downloads if loading for the first time. """

  try:
    _create_unverified_https_context = ssl._create_unverified_context
  except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
  else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

  state_dict = model_zoo.load_url(url_maps[model_name])
  model_arch.load_state_dict(state_dict)
  print(f"Loaded model pretrained weights for `{model_name}`.")
