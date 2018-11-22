"""
testing the trained model
"""
import tensorflow as tf
import argparse
import os
import json
import math
from vacuum.io_ import load_data, fits_encode, save_images, deprocess, preprocess
from vacuum.model import create_generator
from vacuum.util import shift


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="path to folder containing images")
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--checkpoint", required=True, help="directory with checkpoint to use for testing")
parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--separable_conv", action="store_true", help="use separable convolutions in the generator")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--test_start", type=int, help="start index of test dataset subset", default=1800)
parser.add_argument("--test_end", type=int, help="end index of test dataset subset", default=1900)
parser.add_argument('--disable_psf', action='store_true', help="disable the concatenation of the PSF as a channel")

a = parser.parse_args()

CROP_SIZE = 256


def prepare():
    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    # load some options from the checkpoint
    options = {"ngf", "ndf", "disable_psf"}
    with open(os.path.join(a.checkpoint, "options.json")) as f:
        for key, val in json.loads(f.read()).items():
            if key in options:
                print("loaded", key, "=", val)
                setattr(a, key, val)

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))


def main():
    prepare()

    batch, count = load_data(path=a.input_dir, flip=False, crop_size=CROP_SIZE, scale_size=CROP_SIZE, max_epochs=1,
                             batch_size=a.batch_size, start=a.test_start, end=a.test_end)
    steps_per_epoch = int(math.ceil(count / a.batch_size))
    iter_ = batch.make_one_shot_iterator()
    index, min_flux, max_flux, psf, dirty, skymodel = iter_.get_next()
    print("train count = %d" % count)

    with tf.name_scope("scaling_flux"):
        scaled_dirty = preprocess(dirty, min_flux, max_flux)
        scaled_psf = (psf * 2) - 1

    if a.disable_psf:
        input_ = scaled_dirty
    else:
        input_ = tf.concat([scaled_dirty, scaled_psf], axis=3)

    with tf.variable_scope("generator"):
        generator = create_generator(input_, 1, ngf=a.ngf, separable_conv=a.separable_conv)
        deprocessed_output = deprocess(generator, min_flux, max_flux)

    with tf.name_scope("calculate_residuals"):
        shifted = shift(psf, y=-1, x=-1)
        filter_ = tf.expand_dims(tf.expand_dims(tf.squeeze(shifted), 2), 3)
        convolved = tf.nn.conv2d(deprocessed_output, filter_, [1, 1, 1, 1], "SAME")
        residuals = dirty - convolved

    with tf.name_scope("encode_fitss"):
        fits_fetches = {
            "indexs": index,
            "inputs": tf.map_fn(fits_encode, dirty, dtype=tf.string, name="input_fits"),
            "outputs": tf.map_fn(fits_encode, deprocessed_output, dtype=tf.string, name="output_fits"),
            "residuals": tf.map_fn(fits_encode, residuals, dtype=tf.string, name="residuals_fits"),
        }

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=100)

    sv = tf.train.Supervisor(logdir=None, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            print("loaded {}".format(checkpoint))
            saver.restore(sess, checkpoint)

        max_steps = 2 ** 32
        if a.max_epochs is not None:
            max_steps = steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        # at most, process the test data once
        max_steps = min(steps_per_epoch, max_steps)

        # repeat the same for fits arrays
        for step in range(max_steps):
            results = sess.run(fits_fetches)
            filesets = save_images(results, subfolder="fits", extention="fits", output_dir=a.output_dir)
            for f in filesets:
                print("wrote " + f['name'])


if __name__ == '__main__':
    main()
