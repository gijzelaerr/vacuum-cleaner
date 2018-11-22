#!/usr/bin/env python

import sys
from astropy.io import fits


def copy_header(source_path, target_path):
    source = fits.open(source_path, mode='readonly')
    target = fits.open(target_path, mode='update')

    target[0].header = source[0].header
    target.flush()


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("\n  usage: {} <source.fits> <target.fits> [<target.fits> [...]]\n".format(sys.argv[0]))
        sys.exit(1)

    source_path = sys.argv[1]
    target_paths = sys.argv[2:]
    for target in target_paths:
        copy_header(source_path, target)
