from glob import glob
import tensorflow as tf
from astropy.io import fits
from scipy.signal import fftconvolve
from random import choice, randint
from itertools import count
import numpy as np
import logging


logger = logging.getLogger(__name__)


def fits_open_with_header(path, index=0):
    # type: (str, int) -> (np.array, str)
    f = fits.open(str(path))[index]
    return f.data.squeeze(), f.header


def make_skymodel(nsrc=100, size=256, flux_exp_min=-2, flux_exp_max=-1):
    img = np.zeros([size, size], dtype=np.float32)
    flux_scale = 10 ** np.random.randint(flux_exp_min, flux_exp_max)
    fs = (np.random.pareto(5, nsrc) * flux_scale)
    xs = np.random.randint(0, size, nsrc)
    ys = np.random.randint(0, size, nsrc)
    for f, x, y in zip(fs, xs, ys):
        img[x, y] = f
    return img


def convolve(uncolvolved, convolver):
    assert uncolvolved.shape == (256, 256)
    assert convolver.shape == (512, 512)
    p = convolver.shape[0]
    r = slice(p // 2, -p // 2 + 1)  # uneven PSF needs +2, even psf +1
    convolved = fftconvolve(uncolvolved, convolver, mode="full")[r, r]
    assert convolved.shape == uncolvolved.shape
    return convolved, convolver[p // 4:-p // 4, p // 4:-p // 4]


def transform(image, flip, seed):
    if flip:
        image = tf.image.random_flip_left_right(image, seed=seed)
        image = tf.image.random_flip_up_down(image, seed=seed+1)
    return image


def generator(psfs, size=256):
    counter = count()
    while True:
        psf, header = fits_open_with_header(choice(psfs))
        skymodel = make_skymodel(size=size)
        dirty, small_psf = convolve(skymodel, psf)
        min_flux = dirty.min()
        max_flux = dirty.max()
        yield next(counter), min_flux, max_flux, small_psf[:, :, np.newaxis], dirty[:, :, np.newaxis], skymodel[:, :, np.newaxis]


def generative_model(psf_glob, size=256, flip=True, batch_size=1):
    psfs = glob(psf_glob)
    logger.info("Found {} PSFs with glob '{}'".format(len(psfs), psf_glob))

    ds = tf.data.Dataset.from_generator(lambda: generator(psfs, size),
                                        output_shapes=((), (), ()) + ((size, size, 1),) * 3,
                                        output_types=(tf.int32, tf.float32, tf.float32) + (tf.float32,) * 3
                                        )

    # transforming
    # synchronize seed for image operations so that we do the same operations on all
    seed = randint(0, 2 ** 31 - 1)
    t = lambda i: transform(i, flip, seed)
    ds = ds.map(lambda i, mn, mx, psf, drty, skmd: (i, mn, mx, t(psf), t(drty), t(skmd)))

    ds = ds.batch(batch_size)
    return ds