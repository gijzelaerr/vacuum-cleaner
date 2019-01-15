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


from collections import namedtuple


Model = namedtuple("Model", "session, input, output, min_flux, max_flux")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def _padded_generator(dirty_data, psf, n_r, n_c):
    """
    This will take 2 equal size tensors and yields a sliding window subset
    """
    i = 0
    for r, c in (TL, TR, BL, BR):  # step 1, corners
        stamp = dirty_data[r, c]
        yield i, stamp.min(), stamp.max(), psf, stamp
        i += 1

    for r in range(1, n_r):  # step 2: edges left to right
        start = stride * r
        stamp = dirty_data[start:start + SIZE, :SIZE]  # 0,0 -> r,0
        yield i, stamp.min(), stamp.max(), psf, stamp
        i += 1

        stamp = dirty_data[start:start + SIZE, -SIZE:]  # 0,c -> r,c
        yield i, stamp.min(), stamp.max(), psf, stamp
        i += 1

    for c in range(1, n_c):  # step 2: edges, top to bottom
        start = stride * c
        stamp = dirty_data[:SIZE, start:start + SIZE]  # 0,0 -> 0,c
        yield i, stamp.min(), stamp.max(), psf, stamp
        i += 1

        stamp = dirty_data[-SIZE:, start:start + SIZE]  # 0,0 -> r,c
        yield i, stamp.min(), stamp.max(), psf, stamp
        i += 1

    for r, c in product(range(1, n_r), range(1, n_c)):  # step 3, inside
        start_r = stride * r
        start_c = stride * c
        stamp = dirty_data[start_r:start_r + SIZE, start_c:start_c + SIZE]
        yield i, stamp.min(), stamp.max(), psf, stamp
        i += 1


def _restore(shape, generator, n_r, n_c):
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


def init(checkpoint):

    min_flux = tf.placeholder(tf.float32, shape=(1,))
    max_flux = tf.placeholder(tf.float32, shape=(1,))
    input_ = tf.placeholder(tf.float32, shape=(1, SIZE, SIZE, 2))

    # set up the network
    with tf.variable_scope("generator"):
        outputs = create_generator(input_, 1, NGF, SEPERABLE_CONV)
        deprocessed_output = deprocess(outputs, min_flux, max_flux)

    sess = tf.Session()
    logger.info("restoring data from checkpoint " + checkpoint)
    checkpoint = tf.train.latest_checkpoint(checkpoint)
    tf.train.Saver().restore(sess, checkpoint)
    return Model(session=sess, output=deprocessed_output, input=input_, max_flux=max_flux, min_flux=min_flux)


def clean(model, dirty_data, psf_data):

    assert(psf_data.shape == dirty_data.shape)

    # todo: check if we really need this
    dirty_data = dirty_data[:, :, np.newaxis]

    # we need a smaller PSF to give as a channel to the dirty tiles
    psf_data = psf_data / psf_data.max()
    psf_small = psf_data[psf_data.shape[0] // 2 - SIZE // 2 + 1:psf_data.shape[0] // 2 + SIZE // 2 + 1,
                psf_data.shape[1] // 2 - SIZE // 2 + 1:psf_data.shape[1] // 2 + SIZE // 2 + 1]

    logger.debug(psf_small.shape)
    logger.debug((psf_data.shape[0] // 2 - SIZE // 2 + 1, psf_data.shape[0] // 2 + SIZE // 2 + 1,
           psf_data.shape[1] // 2 - SIZE // 2 + 1, psf_data.shape[1] // 2 + SIZE // 2 + 1))

    psf_small = psf_small[:, :, np.newaxis]

    n_r = int(dirty_data.shape[0] / stride)
    n_c = int(dirty_data.shape[1] / stride)

    # run all data through the network
    queue_ = IterableQueue()

    for index, min_flux, max_flux, psf, dirty in _padded_generator(dirty_data, psf_data, n_r, n_c):
        assert (dirty.shape == psf_small.shape)
        scaled_dirty = (dirty / (max_flux/2.0)) - 1
        scaled_psf = (psf_small * 2) - 1
        input_numpy = np.expand_dims(np.concatenate((scaled_dirty, scaled_psf), axis=2), axis=0)

        n = model.session.run(model.output, feed_dict={model.input: input_numpy, model.min_flux: [min_flux],
                                                       model.max_flux: [max_flux]})
        queue_.put(n)

    # reconstruct the data
    reconstructed = _restore(dirty_data.squeeze().shape, iter(queue_), n_r, n_c)
    return reconstructed


def main():
    parser = ArgumentParser()
    parser.add_argument("checkpoint", help="path to trained model directory")
    parser.add_argument("dirty", help="path to dirty fits file")
    parser.add_argument("psf", help="path to psf fits file")
    a = parser.parse_args()

    psf_path = os.path.realpath(a.psf)
    psf_fits = fits.open(str(psf_path))[0]
    psf_data = psf_fits.data.squeeze()

    dirty_path = os.path.realpath(a.dirty)
    dirty_fits = fits.open(str(dirty_path))[0]
    dirty_data = dirty_fits.data.squeeze()

    model = init(checkpoint=a.checkpoint)
    reconstructed = clean(model, dirty_data, psf_data)

    p = psf_data.shape[0]
    #r = slice(p // 2, -p // 2 + 1)  # uneven PSF needs +2, even psf +1
    r = slice(p // 2 + 1, -p // 2 + 2)
    convolved = fftconvolve(reconstructed, psf_data, mode="full")[r, r]
    residual = dirty_fits.data.squeeze() - convolved

    # write the data
    hdu = fits.PrimaryHDU(reconstructed.squeeze())
    hdu.header = dirty_fits.header
    hdul = fits.HDUList([hdu])
    hdul.writeto("vacuum-model.fits", overwrite=True)

    hdu = fits.PrimaryHDU(residual.squeeze())
    hdu.header = dirty_fits.header
    hdul = fits.HDUList([hdu])
    hdul.writeto("vacuum-residual.fits", overwrite=True)
    logger.info("done!")


if __name__ == '__main__':
    main()
