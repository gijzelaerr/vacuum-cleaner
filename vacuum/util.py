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
    return tf.image.pad_to_bounding_box(
        i,
        max(0, y),
        max(0, x),
        i.shape.as_list()[1] + abs(y),
        i.shape.as_list()[2] + abs(x))


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