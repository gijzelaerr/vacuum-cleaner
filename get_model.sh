#!/usr/bin/env bash

TARBALL=model.tar.xz
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
MODEL=${HERE}/share/vacuum/model
URL='http://repo.kernsuite.info/vacuum/'${TARBALL}

cd ${MODEL}
if [ ! -f "${TARBALL}" ]; then
    wget ${URL}
else
    echo "$MODEL/$TARBALL already downloaded"
fi;

if [ ! -f "export.data-00000-of-00001" ]; then
    tar Jxvf ${TARBALL}
else
    echo "tarball already extracted"
fi;

