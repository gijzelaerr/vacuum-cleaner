from tensorflow import image
from functools import lru_cache
from os import path, getcwd
import sys


def shift(i, x=0, y=0):
    """
    Use this to shift your filter for a conv2d, which is probably needed if you do a conv2d with
    even input and/or filter.
    """
    return image.pad_to_bounding_box(
        i,
        max(0, y),
        max(0, x),
        i.shape.as_list()[1] + abs(y),
        i.shape.as_list()[2] + abs(x))


@lru_cache(maxsize=1)
def get_prefix(file: str='share/vacuum/model/checkpoint'):
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


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
