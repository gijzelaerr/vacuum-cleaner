"""
use this for both manual training and testing
"""
import tensorflow as tf
import numpy as np
import argparse
import os
import json
import random
import math
import time

from vacuum.io import load_data, fits_encode, save_images, deprocess
from vacuum.model import create_model
from vacuum.util import shift

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--mode", required=True, choices=["train", "test"])
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None,
                    help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0,
                    help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--separable_conv", action="store_true", help="use separable convolutions in the generator")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

parser.add_argument("--input_multiply", type=float, default=1.0, help="use this to scale the input image")
parser.add_argument("--data_start", type=int, help="start number of dataset subset")
parser.add_argument("--data_end", type=int, help="end number of dataset subset")
parser.add_argument('--disable_psf', action='store_true', help="disable the concatenation of the PSF as a channel")


a = parser.parse_args()

EPS = 1e-12
CROP_SIZE = 256


def main():
    if a.seed is None:
        a.seed = random.randint(0, 2 ** 31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"ngf", "ndf"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)
        # disable these features in test mode
        a.scale_size = CROP_SIZE
        a.flip = False

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    batch, count = load_data(a.input_dir, CROP_SIZE, a.flip, a.scale_size, a.max_epochs, a.batch_size,
                             input_multiply=a.input_multiply, start=a.data_start, end=a.data_end)
    steps_per_epoch = int(math.ceil(count / a.batch_size))
    iter = batch.make_one_shot_iterator()
    index, psf, dirty, skymodel = iter.get_next()
    print("examples count = %d" % count)

    if a.disable_psf:
        input_ = dirty
    else:
        input_ = tf.concat([dirty, psf], axis=3)

    # inputs and targets are [batch_size, height, width, channelsa]
    model = create_model(input_, skymodel, EPS, a.separable_conv, beta1=a.beta1, gan_weight=a.gan_weight,
                         l1_weight=a.l1_weight, lr=a.lr, ndf=a.ndf, ngf=a.ngf)

    deprocessed_input = deprocess(dirty, a.input_multiply)
    deprocessed_target = deprocess(skymodel, a.input_multiply)
    deprocessed_output = deprocess(model.outputs, a.input_multiply)
    deprocessed_psf = deprocess(psf, a.input_multiply)

    with tf.name_scope("calculate_residuals"):
        shifted = shift(deprocessed_psf, y=-1, x=-1)
        filter_ = tf.expand_dims(tf.expand_dims(tf.squeeze(shifted), 2), 3)
        convolved = tf.nn.conv2d(deprocessed_output, filter_, [1, 1, 1, 1], "SAME")
        residuals = deprocessed_input - convolved

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = tf.image.convert_image_dtype(deprocessed_input, dtype=tf.uint8, saturate=True)

    with tf.name_scope("convert_targets"):
        converted_targets = tf.image.convert_image_dtype(deprocessed_target, dtype=tf.uint8, saturate=True)

    with tf.name_scope("convert_outputs"):
        converted_outputs = tf.image.convert_image_dtype(deprocessed_output, dtype=tf.uint8, saturate=True)

    with tf.name_scope("convert_psfs"):
        converted_psfs = tf.image.convert_image_dtype(deprocessed_psf, dtype=tf.uint8, saturate=True)

    with tf.name_scope("convert_residuals"):
        converted_residuals = tf.image.convert_image_dtype(residuals, dtype=tf.uint8, saturate=True)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "indexs": index,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
            "psfs": tf.map_fn(tf.image.encode_png, converted_psfs, dtype=tf.string, name="psf_pngs"),
            "residuals": tf.map_fn(tf.image.encode_png, converted_residuals, dtype=tf.string, name="residual_pngs"),
        }

    with tf.name_scope("encode_fitss"):
        fits_fetches = {
            "indexs": index,
            "inputs": tf.map_fn(fits_encode, deprocessed_input, dtype=tf.string, name="input_fits"),
            "targets": tf.map_fn(fits_encode, deprocessed_target, dtype=tf.string, name="target_fits"),
            "outputs": tf.map_fn(fits_encode, deprocessed_output, dtype=tf.string, name="output_fits"),
            "residuals": tf.map_fn(fits_encode, residuals, dtype=tf.string, name="residuals_fits"),
        }

    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", converted_inputs)

    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", converted_targets)

    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", converted_outputs)

    with tf.name_scope("psfs_summary"):
        tf.summary.image("psfss", converted_psfs)

    with tf.name_scope("residuals_summary"):
        tf.summary.image("residuals", converted_residuals)

    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))

    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=100)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2 ** 32
        if a.max_epochs is not None:
            max_steps = steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":
            # testing
            # at most, process the test data once
            max_steps = min(steps_per_epoch, max_steps)

            # repeat the same for fits arrays
            for step in range(max_steps):
                results = sess.run(fits_fetches)
                filesets = save_images(results, subfolder="fits", extention="fits", output_dir=a.output_dir)
                for f in filesets:
                    print("wrote " + f['name'])

        else:
            # training
            start = time.time()

            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(a.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(a.progress_freq):
                    fetches["discrim_loss"] = model.discrim_loss
                    fetches["gen_loss_GAN"] = model.gen_loss_GAN
                    fetches["gen_loss_L1"] = model.gen_loss_L1

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = display_fetches

                results = sess.run(fetches, options=options, run_metadata=run_metadata)

                if should(a.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(a.display_freq):
                    print("saving display images")
                    save_images(results["display"], step=results["global_step"], output_dir=a.output_dir)

                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / steps_per_epoch)
                    train_step = (results["global_step"] - 1) % steps_per_epoch + 1
                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (
                        train_epoch, train_step, rate, remaining / 60))
                    print("discrim_loss", results["discrim_loss"])
                    print("gen_loss_GAN", results["gen_loss_GAN"])
                    print("gen_loss_L1", results["gen_loss_L1"])

                if should(a.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break


if __name__ == '__main__':
    main()
