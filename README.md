# Deep Vacuum Cleaner

Radio telescope deconvolution based using a Conditional Generative Adversarial Deep Network.

Based on [pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow)

Whch is based on [pix2pix](https://phillipi.github.io/pix2pix/) by Isola et al.

[Article about this implemention](https://affinelayer.com/pix2pix/)


## Setup

```
$ pip install .

```

or if you want to try the GPU accelerated version:

```
$ pip install ".[gpu]"

```
But the tensorflow-gpu package is not the most portable package available.

## Usage

This software is in early alpha stage, and is not ready for end-user usage.
Developement of vacuum-cleaner has been discontinued (for now), so this repository is merely and hopefully an example for future radio astronomers wanting to experiment with deep learning techniques for solving the deconvolultion problem.

## Training

We used [spiel](https://github.com/gijzelaerr/spiel/) for generating training data used for our experiments.
