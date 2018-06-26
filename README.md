# Deep Vacuum Cleaner

Based on [pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow)

Whch is based on [pix2pix](https://phillipi.github.io/pix2pix/) by Isola et al.

[Article about this implemention](https://affinelayer.com/pix2pix/)

[Interactive Demo](https://affinelayer.com/pixsrv/)

Radio telescope deconvolultion version of the tensorflow implementation of pix2pix. 

## Setup

### Prerequisites
- python 3.6
- Tensorflow 

### Recommended
- Linux with Tensorflow GPU edition + cuDNN

### Getting Started

```sh
# clone this repo
git clone https://github.com/gijzelaerr/astro-pix2pix
cd astro-pix2pix
# download the spiel dataset TODO
# train the model (this may take 1-8 hours depending on GPU, on CPU you will be waiting for a bit)
python pix2pix.py \
  --mode train \
  --output_dir scratch/spiel_train \
  --max_epochs 200 \
  --input_dir datasets/train \
  --which_direction BtoA
# test the model
python pix2pix.py \
  --mode test \
  --output_dir scratch/spiel_test \
  --input_dir datasets/val \
  --checkpoint scratch/spiel_train
```

The test run will output an HTML file at `scratch/spiel_test/index.html` that shows input/output/target image sets.

If you have Docker installed, you can use the provided Docker image to run pix2pix without installing the correct version of Tensorflow:

```sh
# train the model
python tools/dockrun.py python pix2pix.py \
      --mode train \
      --output_dir scratch/spiel_train \
      --max_epochs 200 \
      --input_dir datasets/train \
      --which_direction BtoA
# test the model
python tools/dockrun.py python pix2pix.py \
      --mode test \
      --output_dir scratch/spiel_test \
      --input_dir datasets/val \
      --checkpoint scratch/spiel_train
```

## Datasets and Trained Models

For now we assume you are training and testing on the spiel radio telescope simulator

https://github.com/gijzelaerr/spiel/


### Creating your own dataset

you need to manually combine input and output fits into a numpy array using `tools/fits_merge.py`

### Tips

You can look at the loss and computation graph using tensorboard:
```sh
tensorboard --logdir=scratch/spiel_train
```

<img src="docs/tensorboard-scalar.png" width="250px"/> <img src="docs/tensorboard-image.png" width="250px"/> <img src="docs/tensorboard-graph.png" width="250px"/>

If you wish to write in-progress pictures as the network is training, use `--display_freq 50`.  This will update `scratch/spiel_train/index.html` every 50 steps with the current training inputs and outputs.

## Testing

Testing is done with `--mode test`.  You should specify the checkpoint to use with `--checkpoint`, this should point to the `output_dir` that you created previously with `--mode train`:

```sh
python pix2pix.py \
  --mode test \
  --output_dir scratch/spiel_test \
  --input_dir datasets/spiel/val \
  --checkpoint scratch/spiel_train
```

The testing mode will load some of the configuration options from the checkpoint provided so you do not need to specify `which_direction` for instance.

The test run will output an HTML file at `facades_test/index.html` that shows input/output/target image sets:


## Unimplemented Features

The following models have not been implemented:
- defineG_encoder_decoder
- defineG_unet_128
- defineD_pixelGAN


## Acknowledgments
This is a astro port of the port of [pix2pix](https://github.com/phillipi/pix2pix) from Torch to Tensorflow.  
