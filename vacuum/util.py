import queue

import tensorflow as tf
from repoze.lru import lru_cache
from os import path, getcwd
import sys


def shift(i, x=0, y=0):
    """
    Use this to shift your filter for a conv2d, which is probably needed if you do a conv2d with
    even input and/or filter.
    """
    shape = i.shape.as_list()
    width = shape[1]
    height = shape[2]
    offset_width = min(abs(width), 0)
    offset_height = min(abs(height), 0)

    shifted = tf.contrib.image.translate(i, translations=[x, y])
    return tf.image.crop_to_bounding_box(shifted, offset_height, offset_width, height, width)


@lru_cache(maxsize=1)
def get_prefix(file='share/vacuum/model/checkpoint'):
    # type: (str) -> str
    """
    Returns the Python prefix where vacuum is installed
    returns:
        str: path to Python installation prefix
    """
    local = path.dirname(path.dirname(path.abspath(__file__)))
    here = getcwd()

    options = [sys.prefix, here, local, path.expanduser('~/.local'), '/usr/local', '/usr/']
    for option in options:
        if path.isfile(path.join(option, file)):
            return option
    raise Exception("Can't find vacuum installation")


def vis(dirty, batch_size, crop_size):
    """convert image to visibilies"""
    vis = tf.fft2d(tf.complex(dirty, tf.zeros(shape=(batch_size, crop_size, crop_size, 1))))
    real = tf.real(vis)
    imag = tf.imag(vis)
    return real, imag


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class IterableQueue(queue.Queue):
    def __iter__(self):
        while True:
            try:
                yield self.get_nowait()
            except queue.Empty:
                return


def gaussian_kernel(size=1, mean=0.0, std=0.8):
    # type: (int, float, float) -> Any
    d = tf.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start=-size, limit=size+1, dtype=tf.float32))
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    return gauss_kernel / tf.reduce_max(gauss_kernel)
    # return gauss_kernel / tf.reduce_sum(gauss_kernel)


def blur(image):
    kernel = gaussian_kernel(3, 0.0, 0.8)[:, :, tf.newaxis, tf.newaxis]
    blurred = tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding="SAME")
    return blurred
