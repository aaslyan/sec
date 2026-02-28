# Binary DNA paper build
# Usage:
#   make          – build PDF (runs pdflatex + bibtex twice)
#   make clean    – remove auxiliary files
#   make distclean – also remove the PDF

PAPER   = main
LATEX   = pdflatex
BIBTEX  = bibtex
FLAGS   = -interaction=nonstopmode -halt-on-error

.PHONY: all clean distclean view

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib
	$(LATEX) $(FLAGS) $(PAPER)
	$(BIBTEX) $(PAPER)
	$(LATEX) $(FLAGS) $(PAPER)
	$(LATEX) $(FLAGS) $(PAPER)

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg \
	       $(PAPER).log $(PAPER).out $(PAPER).toc \
	       $(PAPER).lof $(PAPER).lot $(PAPER).synctex.gz \
	       texput.log

distclean: clean
	rm -f $(PAPER).pdf

view: $(PAPER).pdf
	xdg-open $(PAPER).pdf 2>/dev/null || open $(PAPER).pdf
