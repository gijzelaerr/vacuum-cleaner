#!/bin/bash -ve

INPUT=$1
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TARGET=/scratch/datasets/kat7_2000
INPUT=/scratch/datasets/kat7_2000/raw

mkdir -p $TARGET/train
mkdir -p $TARGET/test
mkdir -p $TARGET/val


for i in $(seq 0 1799); do
    echo train/$i
    python $HERE/fits_merge.py \
        $INPUT/$i-skymodel.fits \
        $INPUT/$i-wsclean-dirty.fits \
        $TARGET/train/$i
done

for i in $(seq 1800 1899); do
    echo test/$i
    python $HERE/fits_merge.py \
        $INPUT/$i-skymodel.fits \
        $INPUT/$i-wsclean-dirty.fits \
        $TARGET/test/$i
done

for i in $(seq 1900 1999); do
    echo val/$i
    python $HERE/fits_merge.py \
        $INPUT/$i-skymodel.fits \
        $INPUT/$i-wsclean-dirty.fits \
        $TARGET/val/$i
done

