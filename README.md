# GAN-PyTorch

### Overview

This repository contains an op-for-op PyTorch reimplementation
of [Generative Adversarial Networks](http://xxx.itp.ac.cn/pdf/1406.2661v1).

### Table of contents

1. [About Generative Adversarial Networks](#about-generative-adversarial-networks)
2. [Model Description](#model-description)
3. [Installation](#installation)
    * [Clone and install requirements](#clone-and-install-requirements)
    * [Download pretrained weights](#download-pretrained-weights-eg-mnist)
4. [Test](#test)
    * [Torch Hub call](#torch-hub-call)
    * [Base call](#base-call)
5. [Train](#train-eg-mnist)
6. [Contributing](#contributing)
7. [Credit](#credit)

### About Generative Adversarial Networks

If you're new to GANs, here's an abstract straight from the paper:

We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train
two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the
probability that a sample came from the training data rather than G. The training procedure for G is to maximize the
probability of D making a mistake. This framework corresponds to a minimax two-player game. In the space of arbitrary
functions G and D, a unique solution exists, with G recovering the training data distribution and D equal to 1/2
everywhere. In the case where G and D are defined by multilayer perceptrons, the entire system can be trained with
backpropagation. There is no need for any Markov chains or unrolled approximate inference networks during either
training or generation of samples. Experiments demonstrate the potential of the framework through qualitative and
quantitative evaluation of the generated samples.

### Model Description

We have two networks, G (Generator) and D (Discriminator).The Generator is a network for generating images. It receives
a random noise z and generates images from this noise, which is called G(z).Discriminator is a discriminant network that
discriminates whether an image is real. The input is x, x is a picture, and the output is D of x is the probability that
x is a real picture, and if it's 1, it's 100% real, and if it's 0, it's not real.

### Installation

#### Clone and install requirements

```bash
$ git clone https://github.com/Lornatang/GAN-PyTorch.git
$ cd GAN-PyTorch/
$ pip3 install -r requirements.txt
```

#### Download pretrained weights (e.g. mnist)

```bash
$ cd weights/
$ python3 download_weights.py
```

### Test

#### Torch hub call

```python
# Using Torch Hub library.
import torch
import torchvision.utils as vutils

# Choose to use the device.
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load the model into the specified device.
model = torch.hub.load("Lornatang/GAN-PyTorch", "mnist", pretrained=True, verbose=False)
model.eval()
model = model.to(device)

# Create random noise image.
num_images = 64
noise = torch.randn(num_images, 100, device=device)

# The noise is input into the generator model to generate the image.
with torch.no_grad():
    generated_images = model(noise)

# Save generate image.
vutils.save_image(generated_images, "mnist.png", normalize=True)
```

#### Base call

Using pre training model to generate pictures.

```text
usage: test.py [-h] [-a ARCH] [-n NUM_IMAGES] [--outf PATH] [--device DEVICE]

Research and application of GAN based super resolution technology for
pathological microscopic images.

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  model architecture: _gan | cifar10 | discriminator |
                        load_state_dict_from_url | mnist | tfd (default:
                        mnist)
  -n NUM_IMAGES, --num-images NUM_IMAGES
                        How many samples are generated at one time. (default:
                        64).
  --outf PATH           The location of the image in the evaluation process.
                        (default: ``test``).
  --device DEVICE       device id i.e. `0` or `0,1` or `cpu`. (default:
                        ``cpu``).

# Example (e.g. MNIST)
$ python3 test.py -a mnist
```

<span align="center"><img src="assets/mnist.gif" alt="">
</span>

### Train (e.g. MNIST)

```text
usage: train.py [-h] --dataset DATASET [--dataroot DATAROOT] [-j N]
                [--manualSeed MANUALSEED] [--device DEVICE] [-p N] [-a ARCH]
                [--pretrained] [--netD PATH] [--netG PATH] [--start-iter N]
                [--iters N] [-b N] [--image-size IMAGE_SIZE]
                [--channels CHANNELS] [--hidden-channels HIDDEN_CHANNELS]
                [--lr LR]

Research and application of GAN based super resolution technology for
pathological microscopic images.

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     mnist | tfd | cifar10 |.
  --dataroot DATAROOT   Path to dataset. (default: ``data``).
  -j N, --workers N     Number of data loading workers. (default:4)
  --manualSeed MANUALSEED
                        Seed for initializing training. (default:1111)
  --device DEVICE       device id i.e. `0` or `0,1` or `cpu`. (default: ````).
  -p N, --save-freq N   Save frequency. (default: 1000).
  -a ARCH, --arch ARCH  model architecture: _gan | cifar10 | discriminator |
                        load_state_dict_from_url | mnist | tfd (default:
                        mnist)
  --pretrained          Use pre-trained model.
  --netD PATH           Path to latest discriminator checkpoint. (default:
                        ````).
  --netG PATH           Path to latest generator checkpoint. (default: ````).
  --start-iter N        manual iter number (useful on restarts)
  --iters N             The number of iterations is needed in the training of
                        PSNR model. (default: 40000)
  -b N, --batch-size N  mini-batch size (default: 64), this is the total batch
                        size of all GPUs on the current node when using Data
                        Parallel or Distributed Data Parallel.
  --image-size IMAGE_SIZE
                        The height / width of the input image to network.
                        (default: 28).
  --channels CHANNELS   The number of channels of the image. (default: 1).
  --hidden-channels HIDDEN_CHANNELS
                        The number of hidden channels of the image. (default:
                        1200).
  --lr LR               Learning rate. (default:3e-4)


# Example (e.g. MNIST)
$ python3 train.py -a mnist --dataset mnist --image-size 28 --channels 1 --pretrained
```

If you want to load weights that you've trained before, run the following command.

```bash
$ python3 train.py -a mnist \
                   --dataset mnist \
                   --image-size 28 \
                   --channels 1 \
                   --hidden-channels 1200 \
                   --start-iter 10000 \
                   --netG weights/netG_iter_10000.pth \
                   --netD weights/netD_iter_10000.pth
```

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### Generative Adversarial Networks

*Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua
Bengio*

**Abstract**

We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train
two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the
probability that a sample came from the training data rather than G. The training procedure for G is to maximize the
probability of D making a mistake. This framework corresponds to a minimax two-player game. In the space of arbitrary
functions G and D, a unique solution exists, with G recovering the training data distribution and D equal to 1/2
everywhere. In the case where G and D are defined by multilayer perceptrons, the entire system can be trained with
backpropagation. There is no need for any Markov chains or unrolled approximate inference networks during either
training or generation of samples. Experiments demonstrate the potential of the framework through qualitative and
quantitative evaluation of the generated samples.

[[Paper]](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) [[Authors' Implementation]](https://github.com/goodfeli/adversarial)

```
@article{adversarial,
  title={Generative Adversarial Networks},
  author={Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio},
  journal={nips},
  year={2014}
}
```