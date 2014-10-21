#!/bin/bash
set -x
function install_deps {
	sudo apt-get install -qq pandoc
	#conda install -q --yes conda-build # for pipbuild rtdtheme
	# most of doc requirements installed with conda (faster); install remaining with pip (compiling...)
	#conda pipbuild sphinx_rtd_theme
	conda install -q --yes $doc_deps
	pip install -r requirements-build-doc.txt wheel
}

function build_doc {
	pushd doc; make ipython-rst html
}

# TODO: build docs only for python 2.7 and for normal commits (not pull requests) 
echo $TRAVIS_PYTHON_VERSION
if [[ $TRAVIS_PYTHON_VERSION = "2.7" ]]; then
	install_deps
	build_doc
fi
