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
import queue
import tensorflow as tf

from vacuum.io import deprocess, preprocess, fits_open
from vacuum.model import create_model
from vacuum.util import get_prefix, AttrDict


#graph_path = '/home/gijs/Work/vacuum-cleaner/train/meerkat16/frozen.pb'
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
a.pad = 50
a.separable_conv = False


TL = (slice(None, a.size), slice(None, a.size))
BL = (slice(-a.size, None), slice(None, a.size))
TR = (slice(None, a.size), slice(-a.size, None))
BR = (slice(-a.size, None), slice(-a.size, None))

stride = a.size - a.pad * 2


class IterableQueue(queue.Queue):
    def __iter__(self):
        while True:
            try:
                yield self.get_nowait()
            except queue.Empty:
                return


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="")
    return graph


def padded_generator(big_data, psf, n_r, n_c):
    i = 0
    for r, c in (TL, TR, BL, BR):  # step 1
        #print(f"cleaning {i}: {r}, {c}")
        stamp = big_data[r, c]
        yield i, stamp.min(), stamp.max(), psf, stamp
        i += 1

    for r in range(1, n_r):  # step 2: edges left right
        start = stride * r
        #print(f"cleaning {i}: {start}:{start + a.size}, :{a.size}")
        stamp = big_data[start:start + a.size, :a.size]  # 0,0 -> r,0
        yield i, stamp.min(), stamp.max(), psf, stamp
        i += 1

        #print(f"cleaning {i}: {start}:{start + a.size}, {-a.size}:")
        stamp = big_data[start:start + a.size, -a.size:]  # 0,c -> r,c
        yield i, stamp.min(), stamp.max(), psf, stamp
        i += 1

    for c in range(1, n_c):  # step 2: edges, top bottom
        start = stride * c
        #print(f"cleaning {i}: :{a.size}, {start}:{start + a.size}")
        stamp = big_data[:a.size, start:start + a.size]  # 0,0 -> 0,c
        yield i, stamp.min(), stamp.max(), psf, stamp
        i += 1

        #print(f"cleaning {i}: {-a.size}:, {start}:{start + a.size}")
        stamp = big_data[-a.size:, start:start + a.size]  # 0,0 -> r,c
        yield i, stamp.min(), stamp.max(), psf, stamp
        i += 1

    for r, c in product(range(1, n_r), range(1, n_c)):  # step 3
        start_r = stride * r
        start_c = stride * c
        #print(f"cleaning {i}: {start_r}:{start_r + a.size}, {start_c}:{start_c + a.size}")
        stamp = big_data[start_r:start_r + a.size, start_c:start_c + a.size]
        yield i, stamp.min(), stamp.max(), psf, stamp
        i += 1


def load_data(big_data, psf_data, n_r, n_c):

    print(f"PSF: {psf_data.shape}")
    print(f"big FIST: {big_data.shape}")

    count = 4 + (n_r-1 + n_c-1) * 2 + (n_r-1) * (n_c-1)

    print(f"generating a maximum of {count} images")

    ds = tf.data.Dataset.from_generator(lambda: padded_generator(big_data, psf_data, n_r, n_c),
                                        output_shapes=((), (), ()) + ((256, 256, 1),) * 2,
                                        output_types=(tf.int32, tf.float32, tf.float32) + (tf.float32,) * 2)

    ds = ds.batch(1)
    return ds, count


def restore(shape, generator, n_r, n_c):
    restored = np.zeros(shape=shape)

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

    psf_data = fits_open(psf_path)[:, :, np.newaxis]
    big_data = big_fits.data.squeeze()[:, :, np.newaxis]

    n_r = int(big_data.shape[0] / stride)
    n_c = int(big_data.shape[1] / stride)

    batch, count = load_data(big_data, psf_data, n_r, n_c)
    steps_per_epoch = count
    iterator = batch.make_one_shot_iterator()
    index, min_flux, max_flux, psf, dirty = iterator.get_next()
    scaled_dirty = preprocess(dirty, min_flux, max_flux)
    scaled_psf = (psf * 2) - 1
    input_ = tf.concat([scaled_dirty, scaled_psf], axis=3)

    model = create_model(input_, scaled_dirty, a.EPS, a.separable_conv, beta1=a.beta1, gan_weight=a.gan_weight,
                         l1_weight=a.l1_weight, lr=a.lr, ndf=a.ndf, ngf=a.ngf)

    deprocessed_output = deprocess(model.outputs, min_flux, max_flux)

    #graph = load_graph(graph_path)
    #with tf.Session(graph=graph) as sess:
        #init = tf.global_variables_initializer()
        #sess.run(init)

    queue_ = IterableQueue()
    with tf.Session() as sess:
        checkpoint = tf.train.latest_checkpoint(a.checkpoint)
        tf.train.Saver().restore(sess, checkpoint)

        for step in range(steps_per_epoch):
            n = sess.run(deprocessed_output)
            queue_.put(n)

        big_model = restore(big_data.squeeze().shape, iter(queue_), n_r, n_c)
        hdu = fits.PrimaryHDU(big_model.squeeze())
        hdu.header = big_fits.header
        hdul = fits.HDUList([hdu])
        hdul.writeto("stitched.fits", overwrite=True)
        print("done!")


if __name__ == '__main__':
    main()
