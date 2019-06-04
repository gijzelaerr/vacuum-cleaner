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
        input_ = tf.concat([scaled_dirty, scaled_psf], axis=3)

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

def mytf_convolve(arr1, arr2):
    """
    Custom fftconvolve function (in 2D) using tensorflow.
    If arrays have different shape then arr1 has to be 
    the larger of the two
    """
    from tensorflow.python import roll as _roll
    from tensorflow.python.framework import ops
    def myFs(img):
        # note taken from https://gist.github.com/Gurpreetsingh9465/f76cc9e53107c29fd76515d64c294d3f
        x = ops.convert_to_tensor_v2(img)
        axes = tuple(range(x.ndim))
        shift = [dim // 2 for dim in x.shape]
        return _roll(x, shift, axes)
    def myiFs(img):
        # note taken from https://gist.github.com/Gurpreetsingh9465/f76cc9e53107c29fd76515d64c294d3f
        x = ops.convert_to_tensor_v2(img)
        axes = tuple(range(x.ndim))
        shift = [-(int(dim) // 2) for dim in x.shape]
        return _roll(x, shift, axes)

    arr1x, arr1y = arr1.shape
    arr2x, arr2y = arr2.shape
    padx = (arr1x - arr2x)//2
    pady = (arr1y - arr2y)//2
    
    paddings = tf.constant([[padx, padx], [pady, pady]], dtype=tf.int32)
    arr2 = tf.pad(arr2, paddings, "CONSTANT")

    arr1 = tf.convert_to_tensor(arr1, dtype=tf.float32)
    arr2= tf.convert_to_tensor(arr2, dtype=tf.float32)
    arr1hat = tf.signal.rfft2d(myiFs(arr1))
    arr2hat = tf.signal.rfft2d(myiFs(arr2))
    result = myFs(tf.signal.irfft2d(tf.multiply(arr1hat, arr2hat)))
    return result[padx:-padx, pady:-pady]

def test_lik_loss(ID, PSF, IM):
    """
    Compare the likelihood loss computed by tensorflow to that computed using numpy
    loss = I.H.dot(PSF * I - 2 ID)
    Assume all inputs are the same shape i.e. npix x npix
    :return: 
    """
    from tensorflow.python import roll as _roll
    from tensorflow.python.framework import ops
    def myFs(img):
        # note taken from https://gist.github.com/Gurpreetsingh9465/f76cc9e53107c29fd76515d64c294d3f
        x = ops.convert_to_tensor_v2(img)
        axes = tuple(range(x.ndim))
        shift = [dim // 2 for dim in x.shape]
        return _roll(x, shift, axes)
    def myiFs(img):
        # note taken from https://gist.github.com/Gurpreetsingh9465/f76cc9e53107c29fd76515d64c294d3f
        x = ops.convert_to_tensor_v2(img)
        axes = tuple(range(x.ndim))
        shift = [-(int(dim) // 2) for dim in x.shape]
        return _roll(x, shift, axes)

    npix, _ = IM.shape
    npix_psf, _ = PSF.shape
    # convolve image with PSF
    import numpy as np
    Fs = np.fft.fftshift
    iFs = np.fft.ifftshift

    npad = (npix_psf - npix)//2
    I_unpad = slice(npad, -npad)
    IMpad = np.pad(IM, npad, mode='constant')

    IMhat = np.fft.rfft2(iFs(IMpad))
    PSFhat = np.fft.rfft2(iFs(PSF))
    PSFconvI = Fs(np.fft.irfft2(IMhat * PSFhat))[I_unpad, I_unpad]
    import matplotlib.pyplot as plt
    plt.figure('numpy')
    plt.imshow(PSFconvI)
    plt.colorbar()
    PSFconvIflat = PSFconvI.flatten()
    IMflat = IM.flatten()
    IDflat = ID.flatten()
    loss_np2 = IMflat.T.dot(PSFconvIflat - 2 * IDflat)

    print("numpy loss = ", loss_np2)

    plt.show()


    # get tensorflow equivalent (Note tf fft's only seem to give correct answer for float32)
    tf.enable_eager_execution()
    IDtf = tf.convert_to_tensor(ID, dtype=tf.float32)
    IMtf = tf.convert_to_tensor(IM, dtype=tf.float32)
    PSFconvItf = mytf_convolve(PSF, IM)
    plt.figure('tf')
    plt.imshow(PSFconvItf)
    plt.colorbar()
    plt.show()
    loss_tf = tf.reduce_sum(tf.multiply(tf.squeeze(IMtf), tf.squeeze(PSFconvItf) - 2 * IDtf))

    print("tensorflow loss = ", loss_tf)



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
    # main()
    import numpy as np
    # load in ID, PSF
    from astropy.io import fits
    ID = fits.getdata('/home/landman/Projects/Data/MS_dir/ddfacet_test_data/WSCMS_MSMF_TestSuite/natural_512-dirty.fits').squeeze()
    PSF = fits.getdata('/home/landman/Projects/Data/MS_dir/ddfacet_test_data/WSCMS_MSMF_TestSuite/natural_512-psf.fits').squeeze()

    ID = np.asarray(ID, dtype=np.float32)
    PSF = np.asarray(PSF, dtype=np.float32)

    # model is arbitrary
    npix, _ = ID.shape

    IM = np.zeros((npix, npix), dtype=np.float32)
    IM[np.random.randint(0, npix, 5), np.random.randint(0, npix, 5)] = np.random.random(5)

    test_lik_loss(ID, PSF, IM)
