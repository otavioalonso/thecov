# Validation report (LRG1, QSO; holi v3 altmtl, Abacus complete/altmtl)

`thecov_desi_validation_report.pdf` is built from the outputs of `desi_validation/dump_report_data.py`:

    export REPORT_DATA=~/thecov_desi          # or wherever report_data.tgz was unpacked
    cd desi_validation/report
    python figs.py && python tables.py        # figures + results.json + tab_*.tex
    pdflatex report.tex && pdflatex report.tex
