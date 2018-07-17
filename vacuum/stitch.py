"""
This is the cleaning (testing) only code used for the vacuum-clean command
"""
from __future__ import division
from math import ceil
import os
import sys
import numpy as np
from itertools import product
from astropy.io import fits
from queue import Queue
import tensorflow as tf

from vacuum.io import deprocess, preprocess, fits_open
from vacuum.model import create_model
from vacuum.util import get_prefix, AttrDict


a = AttrDict()
a.EPS = 1e-12
a.beta1 = 0.5
a.checkpoint = os.path.join(get_prefix(), "share/vacuum/model")
a.batch_size = 5
a.gan_weight = 1.0
a.l1_weight = 100.0
a.lr = 0.0002
a.ndf = 64
a.ngf = 64
a.output_dir = "."
a.size = 256
a.separable_conv = False


def load_data(dirty_path, psf_path):

    psf = fits_open(psf_path)[:, :, np.newaxis]
    big_fits_data = fits_open(dirty_path)[:, :, np.newaxis]

    print(f"PSF: {psf.shape}")
    print(f"big FIST: {big_fits_data.shape}")

    count = int(ceil(big_fits_data.shape[0] / a.size)) * int(ceil(big_fits_data.shape[1] / a.size))
    print(f"generating a maximum of {count} images")

    def dataset_generator():

        i = 0
        if big_fits_data.shape[0] % a.size:
            print(f"big image x  size ({big_fits_data.shape[0]}) not multiple of {a.size}")
            border = range(int(big_fits_data.shape[1] / a.size))
            for y in border:
                print(f"cleaning {i}: x = {-a.size}:{big_fits_data.shape[0]}, y = {y * a.size}:{(y + 1) * a.size}")
                dirty = big_fits_data[-a.size:, y * a.size:(y + 1) * a.size]
                yield i, dirty.min(), dirty.max(), psf, dirty
                i += 1

        # big image y range not multiple of a.size
        if big_fits_data.shape[1] % a.size:
            print(f"big image y  size ({big_fits_data.shape[1]}) not multiple of {a.size}")
            border = range(int(big_fits_data.shape[0] / a.size))
            for x in border:
                print(f"cleaning {i}: x = {x * a.size}:{(x + 1) * a.size}, y = {-a.size}:{big_fits_data.shape[1]}")
                dirty = big_fits_data[x * a.size:(x + 1) * a.size, -a.size:]
                yield i, dirty.min(), dirty.max(), psf, dirty
                i += 1

        if big_fits_data.shape[0] % a.size and big_fits_data.shape[1] % a.size:
            print(f"both x and y not multiple of {a.size}, generating one corner piece")
            print(f"cleaning {i}: x = {-a.size}:{big_fits_data.shape[0]}, y = {-a.size}:{big_fits_data.shape[1]}")
            dirty = big_fits_data[-a.size:, -a.size:]
            yield i, dirty.min(), dirty.max(), psf, dirty
            i += 1

        cartesian = product(range(int(big_fits_data.shape[0] / a.size)), range(int(big_fits_data.shape[1] / a.size)))
        for x, y in cartesian:
            print(f"cleaning {i}: x = {x * a.size}:{(x + 1) * a.size}, y = {y * a.size}:{(y + 1) * a.size}")
            dirty = big_fits_data[x * a.size:(x + 1) * a.size, y * a.size:(y + 1) * a.size]
            yield i, dirty.min(), dirty.max(), psf, dirty
            i +=1

    ds = tf.data.Dataset.from_generator(dataset_generator,
                                        output_shapes=((), (), ()) + ((256, 256, 1),) * 2,
                                        output_types=(tf.int32, tf.float32, tf.float32) + (tf.float32,) * 2)

    ds = ds.batch(1)
    return ds, count



def main():
    if len(sys.argv) != 3:
        print(f"""
usage: {sys.argv[0]}  dirty.fits psf.fits
""")
        sys.exit(1)

    dirty_path = os.path.realpath(sys.argv[1])
    psf_path = os.path.realpath(sys.argv[2])
    big_fits = fits.open(str(dirty_path))[0]
    big_shape = big_fits.data.squeeze().shape
    batch, count = load_data(dirty_path, psf_path)
    steps_per_epoch = count
    iter = batch.make_one_shot_iterator()
    index, min_flux, max_flux, psf, dirty = iter.get_next()

    scaled_dirty = preprocess(dirty, min_flux, max_flux)
    scaled_psf = (psf * 2) - 1

    input_ = tf.concat([scaled_dirty, scaled_psf], axis=3)

    model = create_model(input_, scaled_dirty, a.EPS, a.separable_conv, beta1=a.beta1, gan_weight=a.gan_weight,
                         l1_weight=a.l1_weight, lr=a.lr, ndf=a.ndf, ngf=a.ngf)

    deprocessed_output = deprocess(model.outputs, min_flux, max_flux)

    queue = Queue()
    with tf.Session() as sess:
        checkpoint = tf.train.latest_checkpoint(a.checkpoint)
        tf.train.Saver().restore(sess, checkpoint)

        for step in range(steps_per_epoch):
            n = sess.run(deprocessed_output)
            queue.put(n)

    big_model = np.empty(big_shape)

    i = 0
    print("combining")
    if big_shape[0] % a.size:
        print(f"big image x  size ({big_shape[0]}) not multiple of {a.size}")
        border = range(int(big_shape[1] / a.size))
        for y in border:
            print(f"writing {i}: x = {-a.size}:{big_shape[0]}, y = {y * a.size}:{(y + 1) * a.size}")
            print(big_model.shape)
            big_model[-a.size:, y * a.size:(y + 1) * a.size] = queue.get().squeeze()
            i += 1

    if big_shape[1] % a.size:
        print(f"big image y  size ({big_shape[1]}) not multiple of {a.size}")
        border = range(int(big_shape[0] / a.size))
        for x in border:
            print(f"writing {i}: x = {x * a.size}:{(x + 1) * a.size}, y = {-a.size}:{big_shape[1]}")
            big_model[x * a.size:(x + 1) * a.size, -a.size:] = queue.get().squeeze()
            i += 1

    if big_shape[0] % a.size and big_shape[1] % a.size:
        print(f"both x and y not multiple of {a.size}, generating one corner piece")
        print(f"writing {i}: x = {-a.size}:{big_shape[0]}, y = {-a.size}:{big_shape[1]}")
        big_model[-a.size:, -a.size:] = queue.get().squeeze()
        i += 1

    cartesian = product(range(int(big_shape[0] / a.size)), range(int(big_shape[1] / a.size)))
    for x, y in cartesian:
        print(f"writing {i}: x = {x * a.size}:{(x + 1) * a.size}, y = {y * a.size}:{(y + 1) * a.size}")
        big_model[x * a.size:(x + 1) * a.size, y * a.size:(y + 1) * a.size] = queue.get().squeeze()
        i += 1

    hdu = fits.PrimaryHDU(big_model.squeeze())
    hdu.header = big_fits.header
    hdul = fits.HDUList([hdu])
    hdul.writeto("stitched.fits")
    print("done!")


if __name__ == '__main__':
    main()
