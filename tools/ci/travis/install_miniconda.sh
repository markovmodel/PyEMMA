#!/bin/bash

TARGET=$HOME/miniconda

function install_miniconda {
	wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O mc.sh -o /dev/null
	bash mc.sh -b -f -p $TARGET
}

function create_env {
	conda create -q --yes -n ci -c https://conda.binstar.org/omnia \
		python=$TRAVIS_PYTHON_VERSION $deps $common_py_deps
}

# check if miniconda is available
if [[ -d $TARGET]]; then
	if [[ ! -x $TARGET/bin/conda ]]; then
		install_miniconda
	fi
fi

export PATH=$TARGET/bin:$PATH

create_env