"""
This is the cleaning (testing) only code used for the vacuum-clean command
"""
from __future__ import division
import os
from argparse import ArgumentParser
import logging
import numpy as np
from itertools import product
from astropy.io import fits
import tensorflow as tf
from scipy.signal import fftconvolve
from vacuum.io_ import deprocess, preprocess
from vacuum.model import create_generator
from vacuum.util import IterableQueue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = ArgumentParser()
parser.add_argument("checkpoint", help="path to trained model directory")
parser.add_argument("dirty", help="path to dirty fits file")
parser.add_argument("psf", help="path to psf fits file")
a = parser.parse_args()

NGF = 64
SIZE = 256
PAD = 50
SEPERABLE_CONV = False

stride = SIZE - PAD * 2

# shortcut for the corners
TL = (slice(None, SIZE), slice(None, SIZE))
BL = (slice(-SIZE, None), slice(None, SIZE))
TR = (slice(None, SIZE), slice(-SIZE, None))
BR = (slice(-SIZE, None), slice(-SIZE, None))


def padded_generator(big_data, psf, n_r, n_c):
    """
    This will take 2 equal size tensors and yields a sliding window subset
    """
    i = 0
    for r, c in (TL, TR, BL, BR):  # step 1, corners
        stamp = big_data[r, c]
        yield i, stamp.min(), stamp.max(), psf, stamp
        i += 1

    for r in range(1, n_r):  # step 2: edges left to right
        start = stride * r
        stamp = big_data[start:start + SIZE, :SIZE]  # 0,0 -> r,0
        yield i, stamp.min(), stamp.max(), psf, stamp
        i += 1

        stamp = big_data[start:start + SIZE, -SIZE:]  # 0,c -> r,c
        yield i, stamp.min(), stamp.max(), psf, stamp
        i += 1

    for c in range(1, n_c):  # step 2: edges, top to bottom
        start = stride * c
        stamp = big_data[:SIZE, start:start + SIZE]  # 0,0 -> 0,c
        yield i, stamp.min(), stamp.max(), psf, stamp
        i += 1

        stamp = big_data[-SIZE:, start:start + SIZE]  # 0,0 -> r,c
        yield i, stamp.min(), stamp.max(), psf, stamp
        i += 1

    for r, c in product(range(1, n_r), range(1, n_c)):  # step 3, inside
        start_r = stride * r
        start_c = stride * c
        stamp = big_data[start_r:start_r + SIZE, start_c:start_c + SIZE]
        yield i, stamp.min(), stamp.max(), psf, stamp
        i += 1


def load_data(big_data, psf_data, n_r, n_c):
    """
    We need to wrap the generator into a Tensorflow dataset
    """

    count = 4 + (n_r - 1 + n_c - 1) * 2 + (n_r - 1) * (n_c - 1)
    logger.debug("PSF: {}".format(psf_data.shape))
    logger.debug("big FIST: {}".format(big_data.shape))
    logger.info("generating a maximum of {} images".format(count))

    ds = tf.data.Dataset.from_generator(lambda: padded_generator(big_data, psf_data, n_r, n_c),
                                        output_shapes=((), (), ()) + ((256, 256, 1),) * 2,
                                        output_types=(tf.int32, tf.float32, tf.float32) + (tf.float32,) * 2)

    ds = ds.batch(1)
    return ds, count


def restore(shape, generator, n_r, n_c):
    """
    restores a set of tiny cleaned images into a big image.
    """
    restored = np.zeros(shape=shape, dtype='>f4')

    logger.info("step 1: corners")
    for r, c in (TL, TR, BL, BR):
        stamp = next(generator).squeeze()
        restored[r, c] = stamp

    logger.info("step 2: edges")
    for r in range(1, n_r):
        start = stride * r
        stamp = next(generator).squeeze()[PAD:-PAD,:]
        restored[start + PAD:start + SIZE - PAD, :SIZE] = stamp
        stamp = next(generator).squeeze()[PAD:-PAD,:]
        restored[start + PAD:start + SIZE - PAD, -SIZE:] = stamp

    for c in range(1, n_c):
        start = stride * c
        stamp = next(generator).squeeze()[:,PAD:-PAD]
        restored[:SIZE, start + PAD:start + SIZE - PAD] = stamp
        stamp = next(generator).squeeze()[:,PAD:-PAD]
        restored[-SIZE:, start + PAD:start + SIZE - PAD] = stamp

    logger.info("step 3: edges")
    for r, c in product(range(1, n_r), range(1, n_c)):
        start_r = stride * r
        start_c = stride * c
        stamp = next(generator).squeeze()[PAD:-PAD,PAD:-PAD]
        restored[start_r + PAD:start_r + SIZE - PAD,
                 start_c + PAD:start_c + SIZE - PAD] = stamp

    return restored


def main():
    dirty_path = os.path.realpath(a.dirty)
    psf_path = os.path.realpath(a.psf)
    big_fits = fits.open(str(dirty_path))[0]
    big_data = big_fits.data.squeeze()[:, :, np.newaxis]
    big_psf_fits = fits.open(str(psf_path))[0]
    assert(big_psf_fits.data.shape == big_fits.data.shape)

    # we need a smaller PSF to give as a channel to the dirty tiles
    big_psf_data = big_psf_fits.data.squeeze()
    big_psf_data = big_psf_data / big_psf_data.max()
    psf_small = big_psf_data[big_psf_data.shape[0] // 2 - SIZE // 2 + 1:big_psf_data.shape[0] // 2 + SIZE // 2 + 1,
                big_psf_data.shape[1] // 2 - SIZE // 2 + 1:big_psf_data.shape[1] // 2 + SIZE // 2 + 1]

    logger.debug(psf_small.shape)
    logger.debug((big_psf_data.shape[0] // 2 - SIZE // 2 + 1, big_psf_data.shape[0] // 2 + SIZE // 2 + 1,
           big_psf_data.shape[1] // 2 - SIZE // 2 + 1, big_psf_data.shape[1] // 2 + SIZE // 2 + 1))

    psf_small = psf_small[:, :, np.newaxis]

    n_r = int(big_data.shape[0] / stride)
    n_c = int(big_data.shape[1] / stride)

    # set up the data loading
    batch, count = load_data(big_data, psf_small, n_r, n_c)
    steps_per_epoch = count
    iterator = batch.make_one_shot_iterator()
    index, min_flux, max_flux, psf, dirty = iterator.get_next()
    scaled_dirty = preprocess(dirty, min_flux, max_flux)
    scaled_psf = (psf * 2) - 1
    input_ = tf.concat([scaled_dirty, scaled_psf], axis=3)

    # set up the network
    with tf.variable_scope("generator"):
        outputs = create_generator(input_, 1, NGF, SEPERABLE_CONV)
        deprocessed_output = deprocess(outputs, min_flux, max_flux)

    # run all data through the network
    queue_ = IterableQueue()
    with tf.Session() as sess:
        logger.info("restoring data from checkpoint " + a.checkpoint)
        checkpoint = tf.train.latest_checkpoint(a.checkpoint)
        tf.train.Saver().restore(sess, checkpoint)

        for step in range(steps_per_epoch):
            n = sess.run(deprocessed_output)
            queue_.put(n)

    # reconstruct the data
    big_model = restore(big_data.squeeze().shape, iter(queue_), n_r, n_c)
    p = big_psf_data.shape[0]
    #r = slice(p // 2, -p // 2 + 1)  # uneven PSF needs +2, even psf +1
    r = slice(p // 2 + 1, -p // 2 + 2)
    convolved = fftconvolve(big_model, big_psf_data, mode="full")[r, r]
    residual = big_fits.data.squeeze() - convolved

    # write the data
    hdu = fits.PrimaryHDU(big_model.squeeze())
    hdu.header = big_fits.header
    hdul = fits.HDUList([hdu])
    hdul.writeto("vacuum-model.fits", overwrite=True)

    hdu = fits.PrimaryHDU(residual.squeeze())
    hdu.header = big_fits.header
    hdul = fits.HDUList([hdu])
    hdul.writeto("vacuum-residual.fits", overwrite=True)

    logger.info("done!")


if __name__ == '__main__':
    main()
