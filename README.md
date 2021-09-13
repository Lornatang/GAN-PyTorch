# GAN-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation  
of [Generative Adversarial Networks](http://arxiv.org/pdf/1406.2661).

### Table of contents

- [GAN-PyTorch](#gan-pytorch)
  - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Test](#test)
    - [Train](#train)
    - [Contributing](#contributing)
    - [Credit](#credit)
      - [Generative Adversarial Networks](#generative-adversarial-networks)

### Download weights

- [Google Driver](https://drive.google.com/file/d/1lBT7msKjLkkAxYee80_mEby_e6pTeLMV/view?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1UpmKF5ABCP2L2DKT9cUlqg) access:`llot`

### Test

Modify the contents of the file as follows.

1. `config.py` line 35 `mode="train"` change to `model="valid"`;
2. `config.py` line 80 `model_path=f"results/{exp_name}/g-last.pth"` change to `model_path=f"<YOUR-WEIGHTS-PATH>.pth"`;
3. Run `python validate.py`.

### Train

Modify the contents of the file as follows.

1. `config.py` line 35 `mode="valid"` change to `model="train"`;
2. Run `python train.py`.

If you want to load weights that you've trained before, modify the contents of the file as follows.

1. `config.py` line 35 `mode="valid"` change to `model="train"`;
2. `config.py` line 51 `start_epoch=0` change to `start_epoch=XXX`;
3. `config.py` line 52 `resume=False` change to `resume=True`;
4. `config.py` line 53 `resume_d_weight=""` change to `resume_p_weight=<YOUR-RESUME-D-WIGHTS-PATH>`;
5. `config.py` line 54 `resume_g_weight=""` change to `resume_p_weight=<YOUR-RESUME-G-WIGHTS-PATH>`;
6. Run `python train.py`.

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### Generative Adversarial Networks

*Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua
Bengio*

**Abstract** <br>
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

```bibtex
@article{adversarial,
  title={Generative Adversarial Networks},
  author={Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio},
  journal={nips},
  year={2014}
}
```
