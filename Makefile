# Gini--Bayes paper: build the PDF and package the arXiv upload.
#
#   make        compile the PDF (xelatex, run twice for cross-references)
#   make zip    compile, then stage source + data into tarball/ and create
#               tarball/gini-bayes-arxiv.tar.gz (the exact arXiv upload)
#   make clean  remove build artifacts and tarball/

TEX := gini-bayes.tex
PDF := gini-bayes.pdf
DATA := $(wildcard data/*.dat)
DIST := tarball
TARBALL := $(DIST)/gini-bayes-arxiv.tar.gz
LATEX := xelatex -interaction=nonstopmode -halt-on-error

.PHONY: all pdf zip clean

all: pdf

pdf: $(PDF)

# Recompile only when the source or its figure data change.
$(PDF): $(TEX) $(DATA)
	$(LATEX) $(TEX)
	$(LATEX) $(TEX)

# Stage exactly what arXiv receives (source + data, no PDF, no top-level
# folder) and tar it. Depends on pdf so a broken build never gets packaged.
zip: pdf
	rm -rf $(DIST)
	mkdir -p $(DIST)/data
	cp $(TEX) $(DIST)/
	cp $(DATA) $(DIST)/data/
	cd $(DIST) && tar czf gini-bayes-arxiv.tar.gz $(TEX) data
	@echo "wrote $(TARBALL):"
	@tar tzf $(TARBALL)

clean:
	rm -rf $(DIST)
	rm -f $(PDF) *.aux *.log *.out *.toc *.synctex.gz