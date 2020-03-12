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

# Refrence https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from scipy.stats import entropy
from torch.autograd import Variable
from torch.nn import functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)


def inception_score(images, batch_size=32, splits=1):
    """ Computes the inception score of the generated images

    Args:
        images (tensor): Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
        batch_size (int): Batch size for feeding into LeNet.
        splits (int): Number of splits.

    Returns:
        Inception Score.
    """
    images_num = len(images)

    assert batch_size > 0
    assert images_num > batch_size

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(images, batch_size=batch_size)

    # Load inception model
    model = LeNet().to(device)
    state = torch.load("../research/mnist.pth")
    model.load_state_dict(state)
    model.eval()

    def get_pred(x):
        x = model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((images_num, 10))

    for i, batch in enumerate(dataloader, 0):
        batch = batch
        batch = batch.to(device)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (images_num // splits): (k + 1) * (images_num // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


if __name__ == '__main__':
    dataset = dset.MNIST(root='../research/data/',
                         download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.5,), (0.5,))
                         ]))

    print("Calculating Inception Score...")
    print(inception_score(IgnoreLabelDataset(dataset), batch_size=32, splits=10))
