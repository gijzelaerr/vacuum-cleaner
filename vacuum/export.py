import argparse
import json
import os
import numpy as np
import tensorflow as tf
from vacuum.model import create_generator
from vacuum.io_ import deprocess, preprocess
from vacuum.io_ import fits_open


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, help="where to put output files")
    parser.add_argument("--checkpoint", required=True, help="directory with checkpoint to resume training from or use for testing")
    parser.add_argument("--separable_conv", action="store_true", help="use separable convolutions in the generator")
    parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
    parser.add_argument('--disable_psf', action='store_true', help="disable the concatenation of the PSF as a channel")

    a = parser.parse_args()

    def load_data(dirty_path, psf_path):
        # type: (str, str) -> tf.data.Dataset
        def dataset_generator():
            psf = fits_open(psf_path)[:, :, np.newaxis]
            dirty = fits_open(dirty_path)[:, :, np.newaxis]
            min_flux = dirty.min()
            max_flux = dirty.max()
            yield min_flux, max_flux, psf, dirty

        ds = tf.data.Dataset.from_generator(dataset_generator,
                                            output_shapes=((), ()) + ((256, 256, 1),) * 2,
                                            output_types=(tf.float32, tf.float32) + (tf.float32,) * 2
                                            )
        ds = ds.batch(1)
        return ds

    dirty_path = tf.placeholder(tf.string, shape=[1])
    psf_path = tf.placeholder(tf.string, shape=[1])
    batch = load_data(dirty_path, psf_path)

    iter = batch.make_one_shot_iterator()
    min_flux, max_flux, psf, dirty = iter.get_next()

    scaled_dirty = preprocess(dirty, min_flux, max_flux)
    scaled_psf = preprocess(psf, min_flux, max_flux)

    if a.disable_psf:
        input_ = scaled_dirty
    else:
        input_ = tf.concat([scaled_dirty, scaled_psf], axis=3)

    with tf.variable_scope("generator"):
        generator =  create_generator(input_, 1, ngf=a.ngf, separable_conv=a.separable_conv)
        batch_output = deprocess(generator, min_flux, max_flux)

    output_image = tf.image.convert_image_dtype(batch_output, dtype=tf.uint8)[0]

    # lets just assume png for now
    output_data = tf.image.encode_png(output_image)
    output = tf.convert_to_tensor([tf.encode_base64(output_data)])

    key = tf.placeholder(tf.string, shape=[1])
    inputs = {
        "key": key.name,
        "input": dirty.name
    }
    tf.add_to_collection("inputs", json.dumps(inputs))
    outputs = {
        "key": tf.identity(key).name,
        "output": output.name,
    }
    tf.add_to_collection("outputs", json.dumps(outputs))

    init_op = tf.global_variables_initializer()
    restore_saver = tf.train.Saver()
    export_saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        print("loading model from checkpoint")
        checkpoint = tf.train.latest_checkpoint(a.checkpoint)
        restore_saver.restore(sess, checkpoint)
        print("exporting model")
        #export_saver.export_meta_graph(filename=os.path.join(a.output_dir, "export.meta"))
        export_saver.save(sess, os.path.join(a.output_dir, "export"), write_meta_graph=True) #, save_relative_paths=True)
        # save_relative_paths is only supported by more recent tensorflows


if __name__ == '__main__':
    main()