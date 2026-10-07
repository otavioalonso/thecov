#!/bin/bash
# T0 test on the 25 AbacusSummit DR2 mocks (LRG1): does N-body (Abacus) prefer the tree-level trispectrum where the
# holi mocks reject it?  Chained debug jobs:
#   1-2. Gaussian (kernel) covariance dumps per cap, as for holi-kcore2 (fills the window / smoothing caches)
#   3.   the combined dump (NGC, SGC, GCcomb) from those caches -> report_data_abacus-kcore-<v>-LRG1.npz
#   4.   ssc_check (SSC, discreteness, T0) with the template-amplitude fit (Fisher errors), plots, bundle
# Usage:  bash ~/thecov/desi_validation/run_abacus_t0_test.sh [altmtl|complete]
# Result: $OUT/thecov_bundle_<date>.tgz (upload it); the log $OUT/abacus_t0_<v>_ssc.log has the amplitude table.
set -e
V=${1:-altmtl}
CS=abacus-2ndgen-dr2-$V
BASE=/dvs_ro/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/full_shape/base
OUT=/global/cfs/cdirs/desicollab/users/oalves/thecov_validation
DIR=$HOME/thecov_desi/${CS}_mock0
LABEL=abacus-kcore-$V-LRG1
ENV="source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main && export PYTHONPATH=\$HOME/thecov:\$PYTHONPATH && cd \$HOME/thecov"
DUMP="python -u -m desi_validation.dump_report_data --bins LRG1 --modes kernel --no-naive --target-near-pairs 1e10 --n-randoms-max 8e6 --cache-tag _r8p10 --cs-version $CS --spectra-dir $BASE/$CS --mock 0"
SB="sbatch --parsable -N 1 -C cpu -q debug -t 00:30:00"

cd $HOME/thecov && git pull origin thecov2
mkdir -p $OUT
j1=$($SB -J abN -o $OUT/abacus_t0_${V}_dump_NGC.log --wrap "$ENV && $DUMP --label abacus-kcore-$V-NGC-LRG1 --regions NGC")
j2=$($SB -J abS -o $OUT/abacus_t0_${V}_dump_SGC.log --wrap "$ENV && $DUMP --label abacus-kcore-$V-SGC-LRG1 --regions SGC")
j3=$($SB -J abAll --dependency=afterok:$j1:$j2 -o $OUT/abacus_t0_${V}_dump_all.log --wrap "$ENV && $DUMP --label $LABEL")
j4=$($SB -J abSSC --dependency=afterok:$j3 -o $OUT/abacus_t0_${V}_ssc.log --wrap "$ENV && \
python -m pytest -q tests/test_ssc.py tests/test_discreteness.py tests/test_trispectrum.py tests/test_covariance_tools.py && \
python -u -m desi_validation.ssc_check --bin LRG1 --label $LABEL --dir $DIR --cs-version $CS --mock 0 --fit-templates && \
python -u -m desi_validation.ssc_plots --bin LRG1 --label $LABEL --dir $DIR && \
DIR=$DIR bash desi_validation/export_bundle.sh $LABEL")
echo "submitted: dumps $j1 $j2 -> $j3 -> ssc_check + bundle $j4"
echo "follow with: squeue --me ; tail -f $OUT/abacus_t0_${V}_ssc.log"
