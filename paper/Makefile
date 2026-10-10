# Reproducible build of the covariance-validation paper.
#   make products RAW="dir1 dir2 ..."   reduce the unpacked NERSC bundles to products/ (once per new NERSC run)
#   make                                 figures + numbers + paper (only what changed is rebuilt)
#   make clean                           remove figures, numbers and LaTeX output (products are kept)
#   make notebook                        paper.ipynb: the text as markdown + every figure script as an editable cell
#   make sync                            write figure cells edited in paper.ipynb back to scripts/fig_*.py
PY      ?= python
S        = scripts
F        = figures
T        = tex
COMMON   = $(S)/common.py
PRODUCTS = products/holi_LRG1.npz products/holi_QSO.npz products/abacus_LRG1.npz

FIGS = budget anatomy diagonals_LRG1 diagonals_QSO corr chi2 eigen clouds binning fiber window params templates holes nz
FIGPDF = $(addprefix $(F)/,$(addsuffix .pdf,$(FIGS)))

all: paper

products:
	@test -n "$(RAW)" || (echo 'usage: make products RAW="<dirs with the unpacked NERSC bundles>"'; exit 1)
	$(PY) -I $(S)/make_products.py --raw $(RAW)

figures: $(FIGPDF)

$(F)/diagonals_%.pdf: $(S)/fig_diagonals.py $(COMMON) $(PRODUCTS)
	cd $(S) && $(PY) fig_diagonals.py $*

# hole maps need the optional NERSC export: skipped (placeholder in the text) when absent
$(F)/holes.pdf: $(S)/fig_holes.py $(COMMON)
	-cd $(S) && $(PY) fig_holes.py

$(F)/%.pdf: $(S)/fig_%.py $(COMMON) $(PRODUCTS)
	cd $(S) && $(PY) fig_$*.py

numbers: $(T)/numbers.tex
$(T)/numbers.tex: $(S)/make_numbers.py $(COMMON) $(PRODUCTS) $(F)/fiber.pdf $(F)/templates.pdf $(F)/binning.pdf $(F)/params.pdf $(F)/nz.pdf
	cd $(S) && $(PY) make_numbers.py

paper: $(T)/main.pdf
$(T)/main.pdf: $(T)/main.tex $(wildcard $(T)/sections/*.tex) $(T)/refs.bib $(T)/numbers.tex $(FIGPDF)
	cd $(T) && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null && bibtex main >/dev/null; \
	  pdflatex -interaction=nonstopmode main.tex >/dev/null; pdflatex -interaction=nonstopmode -halt-on-error main.tex | grep -E 'Warning: (Reference|Citation)|undefined' || true
	@echo "built $(T)/main.pdf"

clean:
	rm -f $(F)/*.pdf $(T)/numbers.tex $(T)/main.{aux,bbl,blg,log,out,toc,pdf} $(T)/sections/*.aux

notebook: $(T)/numbers.tex
	$(PY) $(S)/notebook.py build --execute

sync:
	$(PY) $(S)/notebook.py sync

.PHONY: all products figures numbers paper clean notebook sync
