"""
use this for both manual training and testing
"""
import tensorflow as tf
import numpy as np
import argparse
import os
import json
import random
import time

from vacuum.io_ import save_images, deprocess, preprocess
from vacuum.model import create_model
from vacuum.util import shift
from vacuum.datasets import generative_model

parser = argparse.ArgumentParser()
parser.add_argument("--psf_glob", required=True, help="GLob to find psfs")
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)

parser.add_argument("--max_steps", type=int, default=100000, help="number of training steps (0 to disable)")
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
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=0, help="weight on L1 term for generator gradient")
parser.add_argument("--l0_weight", type=float, default=0, help="weight on L0 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=10, help="weight on GAN term for generator gradient")
parser.add_argument("--res_weight", type=float, default=1, help="weight on residual term for generator gradient")

parser.add_argument("--train_start", type=int, help="start index of train dataset subset", default=0)
parser.add_argument("--train_end", type=int, help="end index of train dataset subset", default=1800)
parser.add_argument('--disable_psf', action='store_true', help="disable the concatenation of the PSF as a channel")

parser.add_argument("--flux_scale_min", type=float, default=10.0, help="Min flux scale relative to estimated noise")
parser.add_argument("--flux_scale_max", type=float, default=10.0, help="Max flux scale relative to estimated noise")

a = parser.parse_args()

EPS = 1e-12
CROP_SIZE = 256


def prepare():
    if a.seed is None:
        a.seed = random.randint(0, 2 ** 31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))


def visual_scaling(img):
    """ go from (-1, 1) to (0, 1)"""
    return (img + 1) / 2


def main():
    prepare()

    train_batch = generative_model(a.psf_glob,
                                   flux_scale_min=a.flux_scale_min,
                                   flux_scale_max=a.flux_scale_max)
    iterator = train_batch.make_one_shot_iterator()
    index, min_flux, max_flux, psf, dirty, skymodel = iterator.get_next()

    with tf.name_scope("scaling_flux"):
        scaled_skymodel = preprocess(skymodel, min_flux, max_flux)
        scaled_dirty = preprocess(dirty, min_flux, max_flux)
        scaled_psf = (psf * 2) - 1

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(scaled_dirty, scaled_skymodel, EPS, a.separable_conv, beta1=a.beta1, gan_weight=a.gan_weight,
                         l1_weight=a.l1_weight, lr=a.lr, ndf=a.ndf, ngf=a.ngf, psf=psf, min_flux=min_flux,
                         max_flux=max_flux, res_weight=a.res_weight, l0_weight=a.l0_weight, disable_psf=a.disable_psf)

    deprocessed_output = deprocess(model.outputs, min_flux, max_flux)

    with tf.name_scope("calculate_residuals"):
        shifted = shift(psf, y=-1, x=-1)
        filter_ = tf.expand_dims(tf.expand_dims(tf.squeeze(shifted), 2), 3)
        convolved = tf.nn.conv2d(deprocessed_output, filter_, [1, 1, 1, 1], "SAME")
        residuals = dirty - convolved

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_images"):
        converted_inputs = tf.image.convert_image_dtype(visual_scaling(scaled_dirty), dtype=tf.uint8, saturate=True)
        converted_targets = tf.image.convert_image_dtype(visual_scaling(scaled_skymodel), dtype=tf.uint8, saturate=True)
        converted_outputs = tf.image.convert_image_dtype(visual_scaling(model.outputs), dtype=tf.uint8, saturate=True)
        converted_psfs = tf.image.convert_image_dtype(visual_scaling(scaled_psf), dtype=tf.uint8, saturate=True)
        converted_residuals = tf.image.convert_image_dtype(visual_scaling(residuals), dtype=tf.uint8, saturate=True)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "indexs": index,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
            "psfs": tf.map_fn(tf.image.encode_png, converted_psfs, dtype=tf.string, name="psf_pngs"),
            "residuals": tf.map_fn(tf.image.encode_png, converted_residuals, dtype=tf.string, name="residual_pngs"),
        }

    # summaries
    with tf.name_scope("combined_summary"):
        tf.summary.image("inputs", converted_inputs)
        tf.summary.image("outputs", converted_outputs)
        tf.summary.image("targets", converted_targets)
        tf.summary.image("residuals", converted_residuals)

    with tf.name_scope("psfs_summary"):
        tf.summary.image("psfss", converted_psfs)

    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))

    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)
    tf.summary.scalar("generator_loss_RES", model.gen_loss_RES)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None

    train_summary_op = tf.summary.merge_all()
    train_summary_writer = tf.summary.FileWriter(logdir=logdir + '/train')

    saver = tf.train.Saver(max_to_keep=100)
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=saver, summary_writer=None,
                             summary_op=None)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)

    with sv.managed_session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        print("parameter_count =", sess.run(parameter_count))

        start = time.time()

        for step in range(a.max_steps):
            def should(freq):
                return freq > 0 and ((step + 1) % freq == 0 or step == a.max_steps - 1)

            options = None
            run_metadata = None
            if should(a.trace_freq):
                print("preparing")
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            fetches = {
                "train": model.train,
                "global_step": sv.global_step,
            }

            if should(a.progress_freq):
                print("progress step")
                fetches["discrim_loss"] = model.discrim_loss
                fetches["gen_loss_GAN"] = model.gen_loss_GAN
                fetches["gen_loss_L1"] = model.gen_loss_L1
                fetches["gen_loss_RES"] = model.gen_loss_RES

            if should(a.summary_freq):
                print("preparing summary")
                fetches["summary"] = train_summary_op

            if should(a.display_freq):
                print("display step step")
                fetches["display"] = display_fetches

            results = sess.run(fetches, options=options, run_metadata=run_metadata)

            if should(a.summary_freq):
                print("recording summary")
                train_summary_writer.add_summary(results["summary"], results["global_step"])

            if should(a.display_freq):
                print("saving display images")
                save_images(results["display"], step=results["global_step"], output_dir=a.output_dir)

            if should(a.trace_freq):
                print("recording trace")
                train_summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

            if should(a.progress_freq):
                rate = (step + 1) * a.batch_size / (time.time() - start)
                remaining = (a.max_steps - step) * a.batch_size / rate
                print("progress  step %d  image/sec %0.1f  remaining %dm" % (results["global_step"], rate, remaining / 60))
                print("discrim_loss", results["discrim_loss"])
                print("gen_loss_GAN", results["gen_loss_GAN"])
                print("gen_loss_L1", results["gen_loss_L1"])
                print("gen_loss_RES", results["gen_loss_RES"])

            if should(a.save_freq):
                print("saving model")
                saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

            if sv.should_stop():
                print("supervisor things we should stop!")
                break

        print("done! bye")


if __name__ == '__main__':
    main()
