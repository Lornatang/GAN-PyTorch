# GAN-PyTorch

### Update (January 29, 2020)

The mnist and fmnist models are now available. Their usage is identical to the other models: 
```python
from gan_pytorch import Generator
model = Generator.from_pretrained('g-mnist') 
```

### Overview
This repository contains an op-for-op PyTorch reimplementation of [Generative Adversarial Networks](http://xxx.itp.ac.cn/pdf/1406.2661v1).

The goal of this implementation is to be simple, highly extensible, and easy to integrate into your own projects. This implementation is a work in progress -- new features are currently being implemented.  

At the moment, you can easily:  
 * Load pretrained Generate models 
 * Use Generate models for extended dataset

_Upcoming features_: In the next few days, you will be able to:
 * Quickly finetune an Generate on your own dataset
 * Export Generate models for production

### Table of contents
1. [About Generative Adversarial Networks](#about-generative-adversarial-networks)
2. [Model Description](#model-description)
3. [Installation](#installation)
4. [Usage](#usage)
    * [Load pretrained models](#loading-pretrained-models)
    * [Example: Extended dataset](#example-extended-dataset)
    * [Example: Visual](#example-visual)
5. [Contributing](#contributing) 

### About Generative Adversarial Networks

If you're new to GANs, here's an abstract straight from the paper:

We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. The training procedure for G is to maximize the probability of D making a mistake. This framework corresponds to a minimax two-player game. In the space of arbitrary functions G and D, a unique solution exists, with G recovering the training data distribution and D equal to 1/2 everywhere. In the case where G and D are defined by multilayer perceptrons, the entire system can be trained with backpropagation. There is no need for any Markov chains or unrolled approximate inference networks during either training or generation of samples. Experiments demonstrate the potential of the framework through qualitative and quantitative evaluation of the generated samples.

### Model Description

We have two networks, G (Generator) and D (Discriminator).The Generator is a network for generating images. It receives a random noise z and generates images from this noise, which is called G(z).Discriminator is a discriminant network that discriminates whether an image is real. The input is x, x is a picture, and the output is D of x is the probability that x is a real picture, and if it's 1, it's 100% real, and if it's 0, it's not real.

### Installation

Install from source:
```bash
git clone https://github.com/lornatang/Generative-Adversarial-Networks
cd Generative-Adversarial-Networks
python setup.py install
``` 

### Usage

#### Loading pretrained models

Load an Generative-Adversarial-Networks:  
```python
from gan_pytorch import Generator
model = Generator.from_name("g-mnist")
```

Load a pretrained Generative-Adversarial-Networks: 
```python
from gan_pytorch import Generator
model = Generator.from_pretrained("g-mnist")
```

#### Example: Extended dataset

As mentioned in the example, if you load the pre-trained weights of the MNIST dataset, it will create a new `imgs` directory and generate 64 random images in the `imgs` directory.

```python
import os
import torch
import torchvision.utils as vutils
from gan_pytorch import Generator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Generator.from_pretrained("g-mnist")
model.to(device)
# switch to evaluate mode
model.eval()

try:
    os.makedirs("./imgs")
except OSError:
    pass

with torch.no_grad():
    for i in range(64):
        noise = torch.randn(64, 100, device=device)
        fake = model(noise)
        vutils.save_image(fake.detach().cpu(), f"./imgs/fake_{i:04d}.png", normalize=True)
    print("The fake image has been generated!")
```

#### Example: Visual

```text
cd $REPO$/framework
python manage.py runserver
```

Then open the browser and type in the browser address [http://127.0.0.1:8000/](http://127.0.0.1:8000/).
Enjoy it.

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.   

I look forward to seeing what the community does with these models! 