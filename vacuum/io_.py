"""
Data ingestion logic
"""
import os
import random
import re
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import tensorflow as tf
from astropy.io import fits


def fits_open(path):
    return fits.open(str(path))[0].data.squeeze()


def transform(image, flip, seed, scale_size, crop_size):
    r = image
    if flip:
        r = tf.image.random_flip_left_right(r, seed=seed)
        r = tf.image.random_flip_up_down(r, seed=seed)

    # area produces a nice downscaling, but does nearest neighbor for upscaling
    # assume we're going to be doing downscaling here
    if scale_size > crop_size:
        r = tf.image.resize_images(r, [scale_size, scale_size], method=tf.image.ResizeMethod.AREA)
        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, scale_size - crop_size + 1, seed=seed)), dtype=tf.int32)
        r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], crop_size, crop_size)
    elif scale_size < crop_size:
        raise Exception("scale size cannot be less than crop size")
    return r


def load_data(path, crop_size, flip, scale_size, max_epochs, batch_size, loop=False, start=None, end=None):
    # type: (str, int, bool, int, int, int, bool, Optional[int], Optional[int]) -> (tf.data.Dataset, int)
    """
    Point this to a path containing fits files fallowing naming schema <number>-<type>.fits

    Returns: a tensorflow dataset generator of dimensions [batch_size, x, y, channels]
    """

    p = Path(path)

    # find out our range
    r = re.compile('.*/(\d+)-skymodel.fits')
    ints = sorted([int(r.match(str(i)).group(1)) for i in p.glob("*-skymodel.fits")])
    if len(ints) > 1:
        min_, max_ = ints[0], ints[-1]
    else:
        min_, max_ = 0, 1

    if start:
        if min_ <= start < max_:
            min_ = start
        else:
            raise Exception("start out of range")

    if end:
        if min_ < end <= max_ + 1:
            max_ = end
        else:
            raise Exception("end ({}) out of range {} - {}".format(end, min_, max_))

    count = max_ - min_

    def dataset_generator():
        for i in range(min_, max_):
            # add one channel
            # header = fits.open("{}/{}-skymodel.fits".format(path, i))[0].header todo: we need to encode this as a string
            psf = fits_open("{}/{}-wsclean-psf.fits".format(path, i))[:, :, np.newaxis]
            dirty = fits_open("{}/{}-wsclean-dirty.fits".format(path, i))[:, :, np.newaxis]
            skymodel = fits_open("{}/{}-skymodel.fits".format(path, i))[:, :, np.newaxis]
            min_flux = dirty.min()
            max_flux = dirty.max()
            yield i, min_flux, max_flux, psf, dirty, skymodel

    ds = tf.data.Dataset.from_generator(dataset_generator,
                                        output_shapes=((), (), ()) + ((256, 256, 1),) * 3,
                                        output_types=(tf.int32, tf.float32, tf.float32) + (tf.float32,) * 3
                                        )

    # transforming
    # synchronize seed for image operations so that we do the same operations on all
    seed = random.randint(0, 2 ** 31 - 1)
    t = lambda i: transform(i, flip, seed, scale_size, crop_size)
    ds = ds.map(lambda i, mn, mx, psf, drty, skmd: (i, mn, mx, t(psf), t(drty), t(skmd)))

    if loop:
        ds = ds.repeat(max_epochs)

    ds = ds.batch(batch_size)
    return ds, count


def fits_decode(content):
    def internal(data):
        return fits.open(BytesIO(data))[0].data.squeeze().astype(np.float32)

    return tf.py_func(internal, [content], tf.float32)


def fits_encode(image):
    def internal(data):
        buf = BytesIO()
        hdu = fits.PrimaryHDU(data.squeeze())
        hdul = fits.HDUList([hdu])
        hdul.writeto(buf)
        return buf.getvalue()

    return tf.py_func(internal, [image], tf.string)


def save_images(fetches, output_dir, step=None, subfolder="images", extention="png"):
    # type: (Dict, str, Optional[int], Optional[str], Optional[str]) -> List[Dict]
    if subfolder:
        image_dir = os.path.join(output_dir, subfolder)
    else:
        image_dir = output_dir
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    indexs = fetches.pop('indexs')
    for key, value in fetches.items():
        for index, contents in zip(indexs, value):
            filename = str(index) + "-" + key + "." + extention
            filesets.append({"name": filename, "step": step})
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            out_path = os.path.join(image_dir, filename)
            with open(out_path, 'wb') as f:
                f.write(contents)
    return filesets


def preprocess(image, min_, max_):
    with tf.name_scope("preprocess"):
        return (image / (max_/2.0)[:, None, None, None]) - 1


def deprocess(image, min_, max_):
    with tf.name_scope("deprocess"):
        return (image + 1) * (max_/2.0)[:, None, None, None]


