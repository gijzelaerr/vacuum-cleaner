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
npy_prefix = sys.argv[3]

if os.access(npy_prefix + '.npy', os.R_OK):
    print('"{}" already exists'.format(npy_prefix + '.npy'))
    sys.exit(1)

fits1_data = fits.open(fits1_path)[axis].data.squeeze()
fits2_data = fits.open(fits2_path)[axis].data.squeeze()

assert(fits1_data.shape == fits2_data.shape)

count = 0
for x in range(int(fits1_data.shape[0] / size)):
    for y in range(int(fits1_data.shape[1] / size)):
        sub1 = fits1_data[x * size:(x + 1) * size, y * size:(y + 1) * size]
        sub2 = fits2_data[x * size:(x + 1) * size, y * size:(y + 1) * size]
        assert(sub1.shape == (size, size))
        assert(sub2.shape == (size, size))
        result = np.concatenate((sub1, sub2), axis=1)
        npy_path = "{}/{}".format(npy_prefix, count)
        print(npy_path)
        np.save(npy_path, result)
        count += 1
