# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = mlneuro
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help Makefile html html-noplot

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) --help

html-noplot:
	@$(SPHINXBUILD) -D plot_gallery=0 -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	cp _examples_cache/* _build -r

html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	cp _build/_images/sphx_glr_plot* _examples_cache/_images/
	cp _build/generated/examples/* _examples_cache/generated/examples


# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
