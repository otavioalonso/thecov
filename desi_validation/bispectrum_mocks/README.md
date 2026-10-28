# Bispectrum covariance from DESI DR2 mocks

Mock-to-mock comparison of the sample covariance of the diagonal Sugiyama
bispectrum multipoles B000(k,k), B202(k,k) (0.02 <= k < 0.205 h/Mpc) across
holi v3 altmtl, GLAM-Uchuu v2 altmtl, Abacus-HF v2 complete (rotated) / altmtl
and Abacus 2nd-gen complete / altmtl. Report: `report/bispectrum_mock_report.pdf`.

Pipeline
1. `inventory.py` (Perlmutter): lists all suites with mesh3 products -> `inventory_mesh3.csv`.
2. `collect.py <suite>` (Perlmutter, cosmodesi env): reads every mock's mesh3/mesh2 h5 files into
   `bispec_<suite>.npz` (+ `.json` metadata, in `meta/`). The npz files (2-80 MB) are not committed.
3. `analysis.py <dir>`: exclusions, variances, correlations, ratios, chi2 consistency, outliers -> `results.json`.
4. `make_report_figures.py <dir>`: figures (vector PDF) and `figures/figure_results.json` (jackknife band ratios, S/N).
5. `typst compile --root . report/bispectrum_mock_report.typ`.

Exclusions: holi `dubious_mock_*` and GLAM `dubious_mock*` directories; holi LRG1 mocks
122, 131, 166, 218, 470, 573, 664, 667, 746, 793, 880 (defective n(z)).
