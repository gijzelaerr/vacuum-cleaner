# Deep Vacuum Cleaner

Radio telescope deconvolulion version of the tensorflow implementation of pix2pix. 

Based on [pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow)

Whch is based on [pix2pix](https://phillipi.github.io/pix2pix/) by Isola et al.

[Article about this implemention](https://affinelayer.com/pix2pix/)


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
The PSF needs to be 256x256 (for now).       


## Training

For now undocumented, but you should use ``vacuum.manual`` and the output of
[spiel](https://github.com/gijzelaerr/spiel/) as training data.


