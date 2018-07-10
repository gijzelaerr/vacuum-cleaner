import sys
import os
from astropy.io import fits
from itertools import product

axis = 0
size = 256

if len(sys.argv) != 4:
    print(f"""
usage: $ {sys.argv[0]} <in.fits> <output prefix> <output postfix>

example: $ {sys.argv[0]} big.fits /output/ -dirty.fits
    """)
    sys.exit(1)

fits_in_path = sys.argv[1]
out_prefix = sys.argv[2]
out_postfix = sys.argv[3]

first = "{}{}{}".format(out_prefix, 0, out_postfix)
if os.access(first, os.R_OK):
    print('"{}" already exists'.format(first))
    sys.exit(1)

print(f"opening {fits_in_path}")
big_fits_data = fits.open(fits_in_path)[axis].data.squeeze()
print(f"Image is {big_fits_data.shape[0]} by {big_fits_data.shape[1]}")

cartesian = product(range(int(big_fits_data.shape[0] / size)), range(int(big_fits_data.shape[1] / size)))
for i, (x, y) in enumerate(cartesian):
    print(f"writing {i}: x = {x * size}:{(x + 1) * size}, y = {y * size}:{(y + 1) * size}")
    data = big_fits_data[x * size:(x + 1) * size, y * size:(y + 1) * size]
    assert(data.shape == (size, size))
    hdu = fits.PrimaryHDU(data)
    hdul = fits.HDUList([hdu])
    hdul.writeto("{}{}{}".format(out_prefix, i, out_postfix))
