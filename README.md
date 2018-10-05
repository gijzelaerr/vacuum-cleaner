# Deep Vacuum Cleaner

Radio telescope deconvolution based using a Conditional Generative Adversarial Deep Network.

Based on [pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow)

Whch is based on [pix2pix](https://phillipi.github.io/pix2pix/) by Isola et al.

[Article about this implemention](https://affinelayer.com/pix2pix/)


## preparations

You probably want to download a pretrained model.

download:

http://repo.kernsuite.info/vacuum/model.tar.xz

And extract to `share/vacuum/model`.
 

## Setup

```
$ pip install vacuum-cleaner

```

or if you want to try the GPU accelerated version:

```
$ pip install "vacuum-cleaner[gpu]"

```
But the tensorflow-gpu package is not the most portable package available.

## Usage

```
$ vacuum-clean dirty.fits psf.fits
```       
      


## Training

Have a look at `vacuum-train --help` or at the source. Intended to be trained with
[spiel](https://github.com/gijzelaerr/spiel/) as training data generator.


