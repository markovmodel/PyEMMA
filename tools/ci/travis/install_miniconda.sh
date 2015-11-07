#!/bin/bash

# make TARGET overrideable with env
: ${TARGET:=$HOME/miniconda}

function install_miniconda {
	echo "installing miniconda to $TARGET"
	wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O mc.sh -o /dev/null
	bash mc.sh -b -f -p $TARGET
}

install_miniconda
export PATH=$TARGET/bin:$PATH
