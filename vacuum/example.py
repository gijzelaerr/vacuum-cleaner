"""
This is an example on how to use vacuum to clean.
"""
import tensorflow as tf
import os
import sys
import numpy as np

from vacuum.io_ import fits_encode, save_images, deprocess, preprocess, fits_open
from vacuum.model import create_generator
from vacuum.util import shift, get_prefix, AttrDict
from typing import List

EPS = 1e-12
CROP_SIZE = 256

a = AttrDict()
a.beta1 = 0.5
a.checkpoint = os.path.join(get_prefix(), "share/vacuum/model")
a.gan_weight = 1.0
a.l1_weight = 100.0
a.lr = 0.0002
a.ndf = 64
a.ngf = 64
a.output_dir = "."
a.scale_size = CROP_SIZE
a.separable_conv = False


def load_data(dirties, psfs):
    # type: (List[str], List[str]) -> (tf.data.Dataset, int)
    count = len(dirties)

    def dataset_generator():
        for i, (dirty_path, psf_path) in enumerate(zip(dirties, psfs)):
            psf = fits_open(psf_path)[:, :, np.newaxis]
            dirty = fits_open(dirty_path)[:, :, np.newaxis]
            min_flux = dirty.min()
            max_flux = dirty.max()
            yield i, min_flux, max_flux, psf, dirty

    ds = tf.data.Dataset.from_generator(dataset_generator,
                                        output_shapes=((), (), ()) + ((256, 256, 1),) * 2,
                                        output_types=(tf.int32, tf.float32, tf.float32) + (tf.float32,) * 2)

    ds = ds.batch(count)
    return ds, count


def main():
    if len(sys.argv) != 3:
        print("""
usage: {}  dirty-0.fits,dirty-1.fits,dirty-2.fits  psf-0.fits,psf-1.fits,psf2.fits
        
 note: names don't matter, order does. only supports fits files of {}x{}
       will write output the current folder.
""".format(sys.argv[0], CROP_SIZE, CROP_SIZE))
        sys.exit(1)

    dirties = [os.path.realpath(i) for i in sys.argv[1].split(',')]
    psfs = [os.path.realpath(i) for i in sys.argv[2].split(',')]
    assert len(dirties) == len(psfs)
    batch, count = load_data(dirties, psfs)
    steps_per_epoch = count
    iter = batch.make_one_shot_iterator()
    index, min_flux, max_flux, psf, dirty = iter.get_next()

    scaled_dirty = preprocess(dirty, min_flux, max_flux)
    scaled_psf = (psf * 2) - 1

    input_ = tf.concat([scaled_dirty, scaled_psf], axis=3)

    # set up the network
    with tf.variable_scope("generator"):
        outputs = create_generator(input_, 1, a.ngf, a.separable_conv)
        deprocessed_output = deprocess(outputs, min_flux, max_flux)

    with tf.name_scope("calculate_residuals"):
        shifted = shift(psf, y=-1, x=-1)
        filter_ = tf.expand_dims(tf.expand_dims(tf.squeeze(shifted), 2), 3)
        convolved = tf.nn.conv2d(deprocessed_output, filter_, [1, 1, 1, 1], "SAME")
        residuals = dirty - convolved

    with tf.name_scope("encode_fitss"):
        fits_fetches = {
            "indexs": index,
            "outputs": tf.map_fn(fits_encode, deprocessed_output, dtype=tf.string, name="output_fits"),
            "residuals": tf.map_fn(fits_encode, residuals, dtype=tf.string, name="residuals_fits"),
        }

    with tf.Session() as sess:
        checkpoint = tf.train.latest_checkpoint(a.checkpoint)
        tf.train.Saver().restore(sess, checkpoint)

        for step in range(steps_per_epoch):
            results = sess.run(fits_fetches)
            filesets = save_images(results, subfolder=None, extention="fits", output_dir=a.output_dir)
            for f in filesets:
                print("wrote " + f['name'])


if __name__ == '__main__':
    main()
