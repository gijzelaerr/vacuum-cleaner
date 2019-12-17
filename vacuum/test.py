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

CROP_SIZE = 256


def prepare(a):
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



def test(
        input_dir,
        output_dir,
        checkpoint,
        batch_size=1,
        test_start=0,
        test_end=999,
        disable_psf=False,
        ngf=64,
        separable_conv=False,
        write_residuals=False,
        write_input=False,
):
    batch, count = load_data(path=input_dir, flip=False, crop_size=CROP_SIZE, scale_size=CROP_SIZE, max_epochs=1,
                             batch_size=batch_size, start=test_start, end=test_end)
    steps_per_epoch = int(math.ceil(count / batch_size))
    iter_ = batch.make_one_shot_iterator()
    index, min_flux, max_flux, psf, dirty, skymodel = iter_.get_next()
    print("train count = %d" % count)

    with tf.name_scope("scaling_flux"):
        scaled_dirty = preprocess(dirty, min_flux, max_flux)
        scaled_psf = (psf * 2) - 1

    if disable_psf:
        input_ = scaled_dirty
    else:
        input_ = tf.concat([dirty, psf[:, 128:-128, 128:-128, :]], axis=3)

    with tf.variable_scope("generator"):
        generator = create_generator(input_, 1, ngf=ngf, separable_conv=separable_conv)
        deprocessed_output = deprocess(generator, min_flux, max_flux)

    if write_residuals:
        with tf.name_scope("calculate_residuals"):
            shifted = shift(psf, y=-1, x=-1)
            filter_ = tf.expand_dims(tf.expand_dims(tf.squeeze(shifted), 2), 3)
            convolved = tf.nn.conv2d(deprocessed_output, filter_, [1, 1, 1, 1], "SAME")
            residuals = dirty - convolved

    with tf.name_scope("encode_fitss"):
        work = {
            "indexs": index,
            "outputs": tf.map_fn(fits_encode, deprocessed_output, dtype=tf.string, name="output_fits"),
        }
        if write_residuals:
            work["residuals"] = tf.map_fn(fits_encode, residuals, dtype=tf.string, name="residuals_fits")
        if write_input:
            work["inputs"] = tf.map_fn(fits_encode, dirty, dtype=tf.string, name="input_fits")

    sv = tf.train.Supervisor(logdir=None)
    with sv.managed_session() as sess:
        sv.saver.restore(sess, checkpoint)

        for step in range(steps_per_epoch):
            results = sess.run(work)
            filesets = save_images(results, subfolder="fits", extention="fits", output_dir=output_dir)
            for f in filesets:
                print("wrote " + f['name'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="path to folder containing images")
    parser.add_argument("--output_dir", required=True, help="where to put output files")
    parser.add_argument("--checkpoint", required=True, help="directory with checkpoint to use for testing")
    parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
    parser.add_argument("--max_epochs", type=int, help="number of training epochs")
    parser.add_argument("--separable_conv", action="store_true", help="use separable convolutions in the generator")
    parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
    parser.add_argument("--test_start", type=int, help="start index of test dataset subset", default=0)
    parser.add_argument("--test_end", type=int, help="end index of test dataset subset", default=999)
    parser.add_argument('--disable_psf', action='store_true', help="disable the concatenation of the PSF as a channel")
    parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")

    a = parser.parse_args()
    #prepare(a)
    #print("loading model from checkpoint")
    #checkpoint = tf.train.latest_checkpoint(a.checkpoint)
    #print("loaded {}".format(checkpoint))

    test(a.input_dir, a.output_dir, a.checkpoint, a.batch_size, a.test_start,
          a.test_end, a.disable_psf, a.ngf, a.separable_conv)


if __name__ == '__main__':
    main()