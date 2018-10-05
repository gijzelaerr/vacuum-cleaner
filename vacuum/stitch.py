"""
This is the cleaning (testing) only code used for the vacuum-clean command
"""
from __future__ import division
import os
import sys
import numpy as np
from itertools import product
from astropy.io import fits
import tensorflow as tf
from scipy.signal import fftconvolve
from vacuum.io import deprocess, preprocess
from vacuum.model import create_generator
from vacuum.util import get_prefix, AttrDict, IterableQueue, shift

a = AttrDict()
a.checkpoint = os.path.join(get_prefix(), "share/vacuum/model")
a.ngf = 64
a.output_dir = "."
a.size = 256
a.pad = 50
a.separable_conv = False

stride = a.size - a.pad * 2


# shortcut for the corners
TL = (slice(None, a.size), slice(None, a.size))
BL = (slice(-a.size, None), slice(None, a.size))
TR = (slice(None, a.size), slice(-a.size, None))
BR = (slice(-a.size, None), slice(-a.size, None))


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
        stamp = big_data[start:start + a.size, :a.size]  # 0,0 -> r,0
        yield i, stamp.min(), stamp.max(), psf, stamp
        i += 1

        stamp = big_data[start:start + a.size, -a.size:]  # 0,c -> r,c
        yield i, stamp.min(), stamp.max(), psf, stamp
        i += 1

    for c in range(1, n_c):  # step 2: edges, top to bottom
        start = stride * c
        stamp = big_data[:a.size, start:start + a.size]  # 0,0 -> 0,c
        yield i, stamp.min(), stamp.max(), psf, stamp
        i += 1

        stamp = big_data[-a.size:, start:start + a.size]  # 0,0 -> r,c
        yield i, stamp.min(), stamp.max(), psf, stamp
        i += 1

    for r, c in product(range(1, n_r), range(1, n_c)):  # step 3, inside
        start_r = stride * r
        start_c = stride * c
        stamp = big_data[start_r:start_r + a.size, start_c:start_c + a.size]
        yield i, stamp.min(), stamp.max(), psf, stamp
        i += 1


def load_data(big_data, psf_data, n_r, n_c):
    """
    We need to wrap the generator into a Tensorflow dataset
    """

    count = 4 + (n_r - 1 + n_c - 1) * 2 + (n_r - 1) * (n_c - 1)
    print(f"PSF: {psf_data.shape}")
    print(f"big FIST: {big_data.shape}")
    print(f"generating a maximum of {count} images")

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

    print("step 1: corners")
    for r, c in (TL, TR, BL, BR):
        stamp = next(generator).squeeze()
        restored[r, c] = stamp

    print("step 2: edges")
    for r in range(1, n_r):
        start = stride * r
        stamp = next(generator).squeeze()[a.pad:-a.pad,:]
        restored[start + a.pad:start + a.size - a.pad, :a.size] = stamp
        stamp = next(generator).squeeze()[a.pad:-a.pad,:]
        restored[start + a.pad:start + a.size - a.pad, -a.size:] = stamp

    for c in range(1, n_c):
        start = stride * c
        stamp = next(generator).squeeze()[:,a.pad:-a.pad]
        restored[:a.size, start + a.pad:start + a.size - a.pad] = stamp
        stamp = next(generator).squeeze()[:,a.pad:-a.pad]
        restored[-a.size:, start + a.pad:start + a.size - a.pad] = stamp

    print("step 3: edges")
    for r, c in product(range(1, n_r), range(1, n_c)):
        start_r = stride * r
        start_c = stride * c
        stamp = next(generator).squeeze()[a.pad:-a.pad,a.pad:-a.pad]
        restored[start_r + a.pad:start_r + a.size - a.pad,
                 start_c + a.pad:start_c + a.size - a.pad] = stamp

    return restored


def main():
    if len(sys.argv) != 3:
        print(f"""
usage: {sys.argv[0]}  dirty.fits psf.fits
""")
        sys.exit(1)

    dirty_path = os.path.realpath(sys.argv[1])
    psf_path = os.path.realpath(sys.argv[2])
    big_fits = fits.open(str(dirty_path))[0]
    big_data = big_fits.data.squeeze()[:, :, np.newaxis]
    big_psf_fits = fits.open(str(psf_path))[0]
    assert(big_psf_fits.data.shape == big_fits.data.shape)

    # we need a smaller PSF to give as a channel to the dirty tiles
    big_psf_data = big_psf_fits.data.squeeze()
    big_psf_data = big_psf_data / big_psf_data.max()
    psf_small = big_psf_data[big_psf_data.shape[0] // 2 - a.size // 2 + 1:big_psf_data.shape[0] // 2 + a.size // 2 + 1,
                big_psf_data.shape[1] // 2 - a.size // 2 + 1:big_psf_data.shape[1] // 2 + a.size // 2 + 1]

    print(psf_small.shape)
    print((big_psf_data.shape[0] // 2 - a.size // 2 + 1, big_psf_data.shape[0] // 2 + a.size // 2 + 1,
           big_psf_data.shape[1] // 2 - a.size // 2 + 1, big_psf_data.shape[1] // 2 + a.size // 2 + 1))

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
        outputs = create_generator(input_, 1, a.ngf, a.separable_conv)
        deprocessed_output = deprocess(outputs, min_flux, max_flux)

    # run all data through the network
    queue_ = IterableQueue()
    with tf.Session() as sess:
        print("restoring data from checkpoint " + a.checkpoint)
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

    print("done!")


if __name__ == '__main__':
    main()
