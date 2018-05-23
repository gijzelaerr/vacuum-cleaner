import sys
import os
from astropy.io import fits
import numpy as np

axis = 0
size = 512

if len(sys.argv) != 4:
    print("usage: $ {} <A.fits> <B.fits> <output prefix>".format(sys.argv[0]))
    sys.exit(1)

fits1_path = sys.argv[1]
fits2_path = sys.argv[2]
npy_path = sys.argv[3]

if os.access(npy_path + '.npy', os.R_OK):
    print('"{}" already exists'.format(npy_path + '.npy'))
    sys.exit(1)

fits1_data = fits.open(fits1_path)[axis].data.squeeze()
fits2_data = fits.open(fits2_path)[axis].data.squeeze()

assert(fits1_data.shape == (size, size))
assert(fits2_data.shape == (size, size))

result = np.concatenate((fits1_data, fits2_data), axis=1)
np.save(npy_path, result)
