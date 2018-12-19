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


def make_skymodel(nsrc=100, size=256, flux_scale_min=10, flux_scale_max=10):
    img = np.zeros([size, size], dtype=np.float32)

    sefd = 520
    dtime = 10
    dfreq = 200e6  # 300e6
    nant = 16

    assert(flux_scale_min <= flux_scale_max)

    if flux_scale_min == flux_scale_max:
        flux_scale = flux_scale_min
    else:
        flux_scale = np.random.randint(flux_scale_min, flux_scale_max)
    Srms = sefd / np.sqrt(2 * dtime * dfreq * (nant * (nant-1))/2)
    noise = np.sqrt(Srms*2)
    fluxes = np.random.pareto(8, nsrc) * noise * flux_scale

    xs = np.random.randint(0, size, nsrc)
    ys = np.random.randint(0, size, nsrc)
    for f, x, y in zip(fluxes, xs, ys):
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


def do_flip(image, flip, seed):
    if flip:
        image = tf.image.random_flip_left_right(image, seed=seed)
        image = tf.image.random_flip_up_down(image, seed=seed+1)
    return image


def do_scale(image, seed):
    scale_size = 286
    crop_size = 256
    r = tf.image.resize_images(image, [scale_size, scale_size], method=tf.image.ResizeMethod.AREA)
    offset = tf.cast(tf.floor(tf.random_uniform([2], 0, scale_size - crop_size + 1, seed=seed)), dtype=tf.int32)
    r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], crop_size, crop_size)
    return r


def generator(psfs, size=256, flux_scale_min=10, flux_scale_max=10):
    counter = count()
    while True:
        psf, header = fits_open_with_header(choice(psfs))
        skymodel = make_skymodel(size=size, flux_scale_min=flux_scale_min, flux_scale_max=flux_scale_max)
        dirty, small_psf = convolve(skymodel, psf)
        min_flux = dirty.min()
        max_flux = dirty.max()
        yield next(counter), min_flux, max_flux, small_psf[:, :, np.newaxis], dirty[:, :, np.newaxis], skymodel[:, :, np.newaxis]


def generative_model(psf_glob, size=256, flip=True, batch_size=1, flux_scale_min=10, flux_scale_max=10):
    psfs = glob(psf_glob)
    print("Found {} PSFs with glob '{}'".format(len(psfs), psf_glob))

    ds = tf.data.Dataset.from_generator(lambda: generator(psfs, size, flux_scale_min=flux_scale_min,
                                                          flux_scale_max=flux_scale_max),
                                        output_shapes=((), (), ()) + ((size, size, 1),) * 3,
                                        output_types=(tf.int32, tf.float32, tf.float32) + (tf.float32,) * 3
                                        )

    # transforming
    # synchronize seed for image operations so that we do the same operations on all
    seed = randint(0, 2 ** 31 - 1)
    f = lambda i: do_flip(i, flip, seed)
    s = lambda i: do_scale(i, seed)
    ds = ds.map(lambda i, mn, mx, psf, drty, skmd: (i, mn, mx, f(psf), f(s(drty)), f(s(skmd))))
    ds = ds.batch(batch_size)
    return ds
