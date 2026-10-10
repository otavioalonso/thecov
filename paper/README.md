# Validation of the DESI DR2 power-spectrum covariance against mocks (JCAP draft)

Everything in the paper is regenerated from scripts: one script per figure, one script for every number quoted in the
text, both reading only the reduced data products in `products/`.

```
paper/
  Makefile               make products RAW=... ; make ; make clean
  scripts/
    make_products.py     raw NERSC bundles -> products/*.npz + products/manifest.json (sha256 of every input)
    common.py            style (fixed colour per model/multipole), data access, statistics, provenance stamp
    fig_<name>.py        one figure each -> figures/<name>.pdf (metadata: script, git commit, product hashes)
    make_numbers.py      every quoted number -> tex/numbers.tex, used in the text as \val{key}
  tex/
    main.tex, sections/*.tex, refs.bib
    jcappub-standin.sty  used only if the official jcappub.sty is absent (drop it and JHEP.bst into tex/)
  products/              NOT in git (DESI mock data; this repository is public)
  figures/               NOT in git (generated)
```

## Data and policy

The repository is public. DESI mock measurements and the figures and numbers derived from them are kept out of git
(`.gitignore`): only code and text are committed. Before collaboration review the draft should move to a private
repository (or Overleaf with git sync); the layout is self-contained, so `paper/` can be copied as is.

## Build

Inputs (NERSC, `/global/cfs/cdirs/desicollab/users/oalves/thecov_validation`):

| product | from | NERSC script |
|---|---|---|
| `holi_<b>.npz` | `report_data_holi-kcore2-<b>.npz`, `ssc_holi-kcore2-<b>.npz`, `report_data_holi-altmtl.npz` | `dump_report_data`, `ssc_check` |
| `abacus_LRG1.npz` | `report_data_abacus-{complete,altmtl}.npz`, `*abacus-kcore-complete-LRG1.npz` | `run_abacus_t0_test.sh` |
| `paper_maps_<b>.npz`, `hole_fraction_<b>.json`, `nz_scatter_<b>.npz` | sky maps, hole statistics, n(z) of all mocks | `paper_export.sh` |

```
cd paper
make products RAW="<dirs where the bundles are unpacked>"
make            # figures, numbers, tex/main.pdf
```

Missing figures (e.g. the hole maps before `paper_export.sh` has run) appear as labelled placeholders; missing numbers
appear as red `??`.

## Working on figures in Jupyter

```
make notebook          # paper.ipynb: text (from tex/, numbers filled in) + each figure script as a cell, executed
jupyter lab paper.ipynb
make sync              # edited figure cells -> scripts/fig_*.py   (python scripts/notebook.py check: what differs)
make paper             # rebuild the PDF
```

Running a figure cell writes `figures/<name>.pdf` exactly as the script does, and shows it inline. The scripts remain
the source of truth: `sync` refuses to overwrite a script that also changed on disk since the notebook was built
(`python scripts/notebook.py sync --force` overrides). Text edits go to `tex/`; rebuild the notebook to see them.
`paper.ipynb` is not in git.

## Conventions for adding to the paper

* A new figure: `scripts/fig_<name>.py` using `common.py` (`holi()`, `load()`, `save()`), added to `FIGS` in the
  Makefile, included with `\paperfig{<name>}`.
* A new number: computed in `make_numbers.py` (or written by a figure script with `write_numbers`) and used as
  `\val{key}`; never typed into the text.
* Colours: one fixed colour per covariance model (`MODEL_COLOR`) and per multipole (`ELL_COLOR`), in every figure.
