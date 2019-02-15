# Makefile for Sphinx documentation
#

SHELL := /bin/bash

# You can set these variables from the command line.
SPHINXOPTS    = -j8 -v
SPHINXBUILD   = LANG=C sphinx-build
PAPER         =

BUILD_DIR?=build
FILES=

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILD_DIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source

# ipython notebook index file
NBCONVERT_CONFIG = $(realpath jupyter_nbconvert_config.py)
NBCONVERT_OUTPUT = source/generated
NBCONVERT_CMD = jupyter nbconvert --output-dir=$(NBCONVERT_OUTPUT) $(nbflags) --config $(NBCONVERT_CONFIG)

.PHONY: help clean html ipython-rst

#------------------------------------------------------------------------------
all: html
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html      to make standalone HTML files"
	@echo "  latex     to make LaTeX files, you can set PAPER=a4 or PAPER=letter"

clean:
	-rm -rf $(BUILD_DIR)/* source/generated source/api/generated

html:
	mkdir -p $(BUILD_DIR)/html $(BUILD_DIR)/doctrees
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILD_DIR)/html $(FILES)
	@echo
	@echo "Build finished. The HTML pages are in build/html."

out_dir:
	mkdir -p source/generated

ipython-rst: out_dir
	cd source/generated
	pwd
	exec $(NBCONVERT_CMD)
	@echo "Conversion finished."

latex:
	mkdir -p build/latex build/doctrees
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) build/latex $(FILES)

html_check:
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) build/html $(FILES)

