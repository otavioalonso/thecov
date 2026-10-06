#!/bin/bash
# Bundle everything needed to rebuild and check the non-Gaussian covariance terms locally (no catalogues needed):
#   ssc_<label>.npz            all terms, their parts and inputs (P_lin, window integrals, long-mode variances, ...)
#   report_data_<label>.npz    the mock vectors and the Gaussian covariance (one per label)
#   ssc_plots_<label>/         figures; *.log in the output directory
#   with --windows also the SSC window pair counts (ssc_windows_<bin>_<cap>.npz), to recompute the long-mode variances.
# Usage (labels default to the LRG1 and QSO kernel runs; every report_data_holi-kcore2-*.npz found is added too):
#   bash ~/thecov/desi_validation/export_bundle.sh [--windows] [label ...]
set -e
OUT=/global/cfs/cdirs/desicollab/users/oalves/thecov_validation
DIR=$HOME/thecov_desi/holi_v3_mock173
WIN=0
if [ "$1" == "--windows" ]; then WIN=1; shift; fi
LABELS="$@"
[ -z "$LABELS" ] && LABELS="holi-kcore2-LRG1 holi-kcore2-QSO"
STAGE=$(mktemp -d)
for f in $DIR/report_data_holi-kcore2-*.npz; do [ -e "$f" ] && cp "$f" $STAGE/; done
for L in $LABELS; do
  [ -e $OUT/ssc_$L.npz ] && cp $OUT/ssc_$L.npz $STAGE/ || echo "missing $OUT/ssc_$L.npz"
  [ -e $DIR/report_data_$L.npz ] && cp $DIR/report_data_$L.npz $STAGE/
  [ -d $OUT/ssc_plots_$L ] && cp -r $OUT/ssc_plots_$L $STAGE/
done
cp $OUT/*.log $STAGE/ 2>/dev/null || true
if [ $WIN == 1 ]; then cp $DIR/ssc_windows_*.npz $STAGE/ 2>/dev/null || true; fi
TGZ=$OUT/thecov_bundle_$(date +%Y%m%d_%H%M).tgz
tar czf $TGZ -C $STAGE .
rm -rf $STAGE
ls -lh $TGZ
