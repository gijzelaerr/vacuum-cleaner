import tensorflow as tf
import os
import math
import sys
import numpy as np

from vacuum.io import fits_encode, save_images, deprocess, preprocess, fits_open
from vacuum.model import create_model
from vacuum.util import shift, get_prefix
from typing import List


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


here = os.path.dirname(os.path.realpath(__file__))

EPS = 1e-12
CROP_SIZE = 256
a = AttrDict()

a.batch_size = 1
a.beta1 = 0.5
a.checkpoint = os.path.join(get_prefix(), "share/vacuum/model")
a.data_end = 1800
a.data_start = 0
a.flip = False
a.gan_weight = 1.0
a.input_multiply = 1.0
a.l1_weight = 100.0
a.lr = 0.0002
a.max_epochs = None
a.ndf = 64
a.ngf = 64
a.output_dir = "."
a.scale_size = CROP_SIZE
a.separable_conv = False


def load_data(dirties: List[str], psfs: List[str], input_multiply: float=1.0):
    count = len(dirties)

    def dataset_generator():
        for i, files in enumerate(zip(dirties, psfs)):
            yield (i,) + tuple(fits_open(j)[:, :, np.newaxis] for j in files)

    ds = tf.data.Dataset.from_generator(dataset_generator,
                                        output_shapes=((),) + ((256, 256, 1),) * 2,
                                        output_types=(tf.int32,) + (tf.float32,) * 2)

    # processing
    p = lambda i: preprocess(i, input_multiply)
    ds = ds.map(lambda a, b, c: (a, p(b), p(c)))

    ds = ds.batch(count)

    return ds, count


def main():
    if len(sys.argv) != 3:
        print(f"""
usage: {sys.argv[0]}  dirty-0.fits,dirty-1.fits,dirty-2.fits  psf-0.fits,psf-1.fits,psf2.fits
        
 note: names don't matter, order does. only supports fits files of {CROP_SIZE}x{CROP_SIZE}
       will write output the current folder.
""")
        sys.exit(1)

    dirties = [os.path.realpath(i) for i in sys.argv[1].split(',')]
    psfs = [os.path.realpath(i) for i in sys.argv[2].split(',')]
    assert len(dirties) == len(psfs)
    batch, count = load_data(dirties, psfs)
    steps_per_epoch = int(math.ceil(count / a.batch_size))
    iter = batch.make_one_shot_iterator()
    index, dirty, psf = iter.get_next()

    input_ = tf.concat([dirty, psf], axis=3)

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(input_, dirty, EPS, a.separable_conv, beta1=a.beta1, gan_weight=a.gan_weight,
                         l1_weight=a.l1_weight, lr=a.lr, ndf=a.ndf, ngf=a.ngf)

    deprocessed_input = deprocess(dirty, a.input_multiply)
    deprocessed_output = deprocess(model.outputs, a.input_multiply)
    deprocessed_psf = deprocess(psf, a.input_multiply)

    with tf.name_scope("calculate_residuals"):
        shifted = shift(deprocessed_psf, y=-1, x=-1)
        filter_ = tf.expand_dims(tf.expand_dims(tf.squeeze(shifted), 2), 3)
        convolved = tf.nn.conv2d(deprocessed_output, filter_, [1, 1, 1, 1], "SAME")
        residuals = deprocessed_input - convolved

    with tf.name_scope("encode_fitss"):
        fits_fetches = {
            "indexs": index,
            "outputs": tf.map_fn(fits_encode, deprocessed_output, dtype=tf.string, name="output_fits"),
            "residuals": tf.map_fn(fits_encode, residuals, dtype=tf.string, name="residuals_fits"),
        }

    saver = tf.train.Saver(max_to_keep=100)

    sv = tf.train.Supervisor(logdir=None, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        if a.checkpoint is not None:
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        # repeat the same for fits arrays
        for step in range(steps_per_epoch):
            results = sess.run(fits_fetches)
            filesets = save_images(results, subfolder=None, extention="fits", output_dir=a.output_dir)
            for f in filesets:
                print("wrote " + f['name'])


if __name__ == '__main__':
    main()
