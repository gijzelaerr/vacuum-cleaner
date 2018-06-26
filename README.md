# Deep Vacuum Cleaner

Radio telescope deconvolultion version of the tensorflow implementation of pix2pix. 

Based on [pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow)

Whch is based on [pix2pix](https://phillipi.github.io/pix2pix/) by Isola et al.

[Article about this implemention](https://affinelayer.com/pix2pix/)

[Interactive Demo](https://affinelayer.com/pixsrv/)

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
$ vacuum-clean dirty-0.fits,dirty-1.fits,dirty-2.fits  psf-0.fits,psf-1.fits,psf2.fits
```       
Names don't matter, order does. only supports fits files of 256 256 for now. Will write output the current folder.
       

## Training

For now undocumented, but you should use ``vacuum.manual`` and the output of [spiel](https://github.com/gijzelaerr/spiel/).


