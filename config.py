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
# File description: Realize the parameter configuration function of data set, model, training and verification code.
# ==============================================================================
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import Discriminator
from model import Generator

# ==============================================================================
#                              Common configure
# ==============================================================================
torch.manual_seed(0)                       # Set random seed.
device           = torch.device("cuda:0")  # Use the first GPU for processing by default.
cudnn.benchmark  = True                    # If the dimension or type of the input data of the network does not change much, turn it on, otherwise turn it off.
mode             = "train"                 # Run mode. Specific mode loads specific variables.
exp_name         = "exp000"                # Experiment name.

# ==============================================================================
#                              Train configure
# ==============================================================================
if mode == "train":
    # Configure dataset.
    dataset_dir           = "data"                                # The address of the training dataset.
    batch_size            = 128                                   # Training data batch size.

    # Configure model.
    discriminator         = Discriminator().to(device)            # Load the discriminator model.
    generator             = Generator().to(device)                # Load the generative model.

    # Resume training.
    start_epoch           = 0                                     # The number of initial iterations of the adversarial network training. When set to 0, it means incremental training.
    resume                = False                                 # Set to `True` to continue training from the previous training progress.
    resume_d_weight       = ""                                    # Restore the weight of the discriminator model during training.
    resume_g_weight       = ""                                    # Restore the weight of the generative model during training.

    # Train epochs.
    epochs                = 128                                   # The total number of cycles in the training phase of the adversarial network.

    # Loss function.
    criterion = nn.BCELoss().to(device)                           # Adversarial loss.

    # Optimizer.
    d_optimizer           = optim.Adam(discriminator.parameters(), 0.0002, (0.5, 0.999))  # Discriminator learning rate during adversarial network training.
    g_optimizer           = optim.Adam(generator.parameters(),     0.0002, (0.5, 0.999))  # Generator learning rate during adversarial network training.

    # Training log.
    writer                = SummaryWriter(os.path.join("samples",  "logs", exp_name))

    # Additional variables.
    exp_dir1 = os.path.join("samples", exp_name)
    exp_dir2 = os.path.join("results", exp_name)

# ==============================================================================
#                              Validate configure
# ==============================================================================
if mode == "valid":
    exp_dir      = os.path.join("results", "test", exp_name)  # Additional variables.

    model        = Generator().to(device)                     # Load model.
    model_path   = f"results/{exp_name}/g-last.pth"           # Model weight address.
    dataset_dir  = f"data"                                    # Dataset address.

