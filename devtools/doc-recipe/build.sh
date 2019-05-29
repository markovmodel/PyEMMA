#!/usr/bin/env bash -x

set -x

# we want to install some dependencies via pip
export PIP_IGNORE_INSTALLED=false
export PIP_NO_INDEX=false

pyemma_version=`python -c "import pyemma as e; print(e.version)"`
export BUILD_DIR=${PREFIX}/v${pyemma_version}

# disable progress bars
export PYEMMA_CFG_DIR=`mktemp -d`
python -c "import pyemma; pyemma.config.show_progress_bars=False; pyemma.config.save()";
# print new config
python -c "import pyemma; print(pyemma.config)"

# if we have the fu-berlin file system, we copy the unpublished data (bpti)
if [[ -d /group/ag_cmb/pyemma_performance/unpublished ]]; then
    cp /group/ag_cmb/pyemma_performance/unpublished ./pyemma-ipython -vuR
fi

# install requirements, which are not available in conda
pip install -vvv -r requirements-build-doc.txt

make clean
make html

# we only want to have the html contents
mv $BUILD_DIR/html/* $BUILD_DIR
rm -rf $BUILD_DIR/doctrees

# remove the deps from $PREFIX so we have only the docs left.
pip uninstall -y -r requirements-build-doc.txt
