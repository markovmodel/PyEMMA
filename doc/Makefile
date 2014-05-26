# Makefile for Sphinx documentation
#

PYVER = 2.7
PYTHON = python$(PYVER)

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = LANG=C sphinx-build
PAPER         =

FILES=

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d build/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source

# ipython notebook index file
F			 = source/generated/ipynb-list.rst
HEADER = .. toctree:: \n   :maxdepth: 1 \n

.PHONY: help clean html ipython-rst

#------------------------------------------------------------------------------
all: html
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html      to make standalone HTML files"

clean:
	-rm -rf build/* source/generated

html:
	mkdir -p build/html build/doctrees
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) build/html $(FILES)
	@echo
	@echo "Build finished. The HTML pages are in build/html."

source/generated:
	mkdir -p source/generated

ipython-rst: source/generated
	# find all ipython notebooks (without their checkpoints)
	@$(eval notebooks:= $$(shell cd source/generated && \
						find ../../../ipython/ -name \*.ipynb \
							-and -not -name \*-checkpoint.ipynb))
	# convert notebooks to rst
	cd source/generated; ipython nbconvert --to rst $(notebooks)
	# strip directory part and extensions
	@$(eval notebooks=$$(notdir $(notebooks)))
	@$(eval notebooks=$$(basename $(notebooks)))
	# write header
	@echo $(HEADER) > $(F)
	# put every notebook as a link in index file
	@$(foreach n,$(notebooks), echo "   generated/$n\n" >> $(F);)

	@echo "Conversion finished."
