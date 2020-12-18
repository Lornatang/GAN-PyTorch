# GAN-PyTorch

### Overview
This repository contains an op-for-op PyTorch reimplementation of [Generative Adversarial Networks](http://xxx.itp.ac.cn/pdf/1406.2661v1).

### Table of contents
1. [About Generative Adversarial Networks](#about-generative-adversarial-networks)
2. [Model Description](#model-description)
3. [Installation](#installation)
    * [Clone and install requirements](#clone-and-install-requirements)
    * [Download pretrained weights](#download-pretrained-weights-eg-mnist)
4. [Test](#test)
5. [Train](#train-eg-div2k)
6. [Contributing](#contributing) 
7. [Credit](#credit)

### About Generative Adversarial Networks

If you're new to GANs, here's an abstract straight from the paper:

We propose a new framework for estimating generative models via an adversarial process, 
in which we simultaneously train two models: a generative model G that captures the data distribution, 
and a discriminative model D that estimates the probability that a sample came from the training data rather than G. 
The training procedure for G is to maximize the probability of D making a mistake. This framework corresponds to 
a minimax two-player game. In the space of arbitrary functions G and D, a unique solution exists, with G recovering 
the training data distribution and D equal to 1/2 everywhere. In the case where G and D are defined by multilayer 
perceptrons, the entire system can be trained with backpropagation. There is no need for any Markov chains or 
unrolled approximate inference networks during either training or generation of samples. Experiments demonstrate 
the potential of the framework through qualitative and quantitative evaluation of the generated samples.

### Model Description

We have two networks, G (Generator) and D (Discriminator).The Generator is a network for generating images. 
It receives a random noise z and generates images from this noise, which is called G(z).Discriminator is 
a discriminant network that discriminates whether an image is real. The input is x, x is a picture, 
and the output is D of x is the probability that x is a real picture, and if it's 1, it's 100% real, 
and if it's 0, it's not real.

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

Using pre training model to generate pictures.
```shell

```

MNIST
<span align="center"><img src="assets/mnist.gif" alt="">
</span>

### Train (e.g. MNIST)

```text
usage: train.py [-h] --dataset DATASET [--dataroot DATAROOT] [-j N]
                [--manualSeed MANUALSEED] [--device DEVICE] [-p N] [-a ARCH]
                [--model-path PATH] [--pretrained] [--netD PATH] [--netG PATH]
                [--start-epoch N] [--iters N] [-b N] [--image-size IMAGE_SIZE]
                [--channels CHANNELS] [--lr LR]

Research and application of GAN based super resolution technology for
pathological microscopic images.

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     mnist | fashion-mnist | cifar10 |.
  --dataroot DATAROOT   Path to dataset. (default: ``data``).
  -j N, --workers N     Number of data loading workers. (default:4)
  --manualSeed MANUALSEED
                        Seed for initializing training. (default:1111)
  --device DEVICE       device id i.e. `0` or `0,1` or `cpu`. (default: ````).
  -p N, --save-freq N   Save frequency. (default: 50).
  -a ARCH, --arch ARCH  model architecture: cifar10 | discriminator | fashion-mnist |
                        mnist (default: mnist)
  --model-path PATH     Path to latest checkpoint for model. (default: ````).
  --pretrained          Use pre-trained model.
  --netD PATH           Path to latest discriminator checkpoint. (default:
                        ````).
  --netG PATH           Path to latest generator checkpoint. (default: ````).
  --start-epoch N       manual epoch number (useful on restarts)
  --iters N             The number of iterations is needed in the training of
                        PSNR model. (default: 1e5)
  -b N, --batch-size N  mini-batch size (default: 64), this is the total batch
                        size of all GPUs on the current node when using Data
                        Parallel or Distributed Data Parallel.
  --image-size IMAGE_SIZE
                        The height / width of the input image to network.
                        (default: 28).
  --channels CHANNELS   The number of channels of the image. (default: 1).
  --lr LR               Learning rate. (default:3e-4)

# Example (e.g MNIST)
$ python3 train.py -a mnist --dataset mnist --image-size 28 --channels 1 --pretrained
```

If you want to load weights that you've trained before, run the following command.

```bash
$ python3 train.py -a mnist \
                   --dataset mnist \
                   --image-size 28 \
                   --channels 1 \
                   --start-epoch 18 \
                   --netG weights/netG_epoch_18.pth \
                   --netD weights/netD_epoch_18.pth
```

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.   

I look forward to seeing what the community does with these models!

### Credit

#### Generative Adversarial Networks

*Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio*

**Abstract**

We propose a new framework for estimating generative models via an adversarial process, 
in which we simultaneously train two models: a generative model G that captures the data 
distribution, and a discriminative model D that estimates the probability that a sample 
came from the training data rather than G. The training procedure for G is to maximize 
the probability of D making a mistake. This framework corresponds to a minimax two-player 
game. In the space of arbitrary functions G and D, a unique solution exists, with G 
recovering the training data distribution and D equal to 1/2 everywhere. In the case 
where G and D are defined by multilayer perceptrons, the entire system can be trained 
with backpropagation. There is no need for any Markov chains or unrolled approximate 
inference networks during either training or generation of samples. Experiments demonstrate 
the potential of the framework through qualitative and quantitative evaluation of the generated samples.

[[Paper]](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) [[Authors' Implementation]](https://github.com/goodfeli/adversarial)

```
@article{adversarial,
  title={Generative Adversarial Networks},
  author={Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio},
  journal={nips},
  year={2014}
}
```