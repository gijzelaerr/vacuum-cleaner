import sys
import os
from astropy.io import fits
import numpy as np

axis = 0
size = 512

if len(sys.argv) != 3:
    print("usage: $ {} <A.fits> <stamp prefix>".format(sys.argv[0]))
    sys.exit(1)

fits1_path = sys.argv[1]
npy_prefix = sys.argv[2]


if os.access(npy_prefix + '.npy', os.R_OK):
    print('"{}" already exists'.format(npy_prefix + '.npy'))
    sys.exit(1)


fits_template = fits.open(fits1_path)[axis]

data = fits_template.data.squeeze()

count = 0
for x in range(int(data.shape[0] / size)):
    for y in range(int(data.shape[1] / size)):
        tile = np.load("{}/{}-outputs.npy".format(npy_prefix, count)).squeeze()
        data[x * size:(x + 1) * size, y * size:(y + 1) * size] = tile
        count += 1


hdu = fits.PrimaryHDU(data.squeeze())
hdu.header = fits_template.header
hdul = fits.HDUList([hdu])
hdul.writeto("stitched")
