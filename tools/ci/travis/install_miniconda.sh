#!/bin/bash

TARGET=$HOME/miniconda
BASE_ENV=$TARGET/envs/ci

function install_miniconda {
	echo "installing miniconda"
	wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O mc.sh -o /dev/null
	bash mc.sh -b -f -p $TARGET
}

function create_env {
	echo "create env"
	# initially create env
	if [[ ! -d $BASE_ENV ]]; then
		echo "base env does not exists, creating it"
		conda create -q --yes -n ci -c https://conda.binstar.org/omnia \
			python=$TRAVIS_PYTHON_VERSION $deps $common_py_deps
	fi
}

# check if miniconda is available
#if [[ -d $TARGET ]]; then
#	if [[ ! -x $TARGET/bin/conda ]]; then
#		echo "conda not available, install miniconda now..."
#		install_miniconda
#	fi
#fi

export PATH=$TARGET/bin:$PATH

install_miniconda
create_env
