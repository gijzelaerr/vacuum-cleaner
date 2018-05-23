#!/bin/bash -ve

INPUT=$1
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TARGET=${HERE}/../datasets/spiel_512/
INPUT=/home/gijs/Work/spiel/runs/run_2018-05-17-10-52-04/results

mkdir -p $TARGET/train
mkdir -p $TARGET/test
mkdir -p $TARGET/val


for i in $(seq 0 399); do
    echo train/$i
    python $HERE/fits_merge.py \
        $INPUT/$i-skymodel.fits \
        $INPUT/$i-wsclean-dirty.fits \
        $TARGET/train/$i
done

for i in $(seq 400 499); do
    echo test/$i
    python $HERE/fits_merge.py \
        $INPUT/$i-skymodel.fits \
        $INPUT/$i-wsclean-dirty.fits \
        $TARGET/test/$i
done

for i in $(seq 500 599); do
    echo val/$i
    python $HERE/fits_merge.py \
        $INPUT/$i-skymodel.fits \
        $INPUT/$i-wsclean-dirty.fits \
        $TARGET/val/$i
done

