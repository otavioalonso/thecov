#!/bin/bash
# Bring every NERSC input of the validation paper up to date, then pack them into one tarball:
#   bash ~/thecov/desi_validation/paper_update.sh
# 1. git pull; check each input (present, and with the keys the paper needs: python -m desi_validation.paper_inputs --check)
# 2. submit debug jobs only for what is missing or stale: ssc_check (holi LRG1 / QSO, ~11 min for both) and/or
#    paper_export (maps, hole statistics, n(z) of all mocks)
# 3. a last job (after those) packs $OUT/paper_raw_<date>.tgz; upload it, then locally:
#      cd paper && make products RAW="<dir where it is unpacked>" && make
# Gaussian-covariance dumps and the Abacus run are expensive and are only reported, never resubmitted.
set -e
OUT=/global/cfs/cdirs/desicollab/users/oalves/thecov_validation
ENV="source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main && export PYTHONPATH=\$HOME/thecov:\$PYTHONPATH && cd \$HOME/thecov"
SB="sbatch --parsable -N 1 -C cpu -q debug -t 00:30:00"
cd $HOME/thecov && git pull origin thecov2
( eval "$ENV" && python -u -m desi_validation.paper_inputs --check )
ACT=$(cat $OUT/paper_inputs_actions.txt)
DEPS=""
SSC_BINS=$(echo "$ACT" | grep '^ssc:' | cut -d: -f2 | tr '\n' ' ')
if [ -n "$SSC_BINS" ]; then
  CMD="$ENV"
  for b in $SSC_BINS; do CMD="$CMD && python -u -m desi_validation.ssc_check --bin $b --label holi-kcore2-$b"; done
  j=$($SB -J pssc -o $OUT/paper_ssc.log --wrap "$CMD")
  echo "submitted ssc_check for $SSC_BINS: job $j (log $OUT/paper_ssc.log)"
  DEPS="$DEPS:$j"
fi
if echo "$ACT" | grep -q '^export$'; then
  # full-resolution maps from an earlier export only need slimming (seconds); then check again
  if [ -e $OUT/paper_maps_LRG1.npz ] && [ -e $OUT/paper_maps_QSO.npz ]; then
    ( eval "$ENV" && python -u -m desi_validation.paper_export --slim ) || true
  fi
  ( eval "$ENV" && python -m desi_validation.paper_inputs --check >/dev/null )
  if grep -q '^export$' $OUT/paper_inputs_actions.txt; then
    j1=$($SB -J pmaps -o $OUT/paper_maps.log --wrap "$ENV && python -u -m desi_validation.paper_export --bins LRG1 QSO && \
      ( [ -e $OUT/hole_fraction_LRG1.json ] || python -u -m desi_validation.hole_fraction --bins LRG1 QSO )")
    j2=$($SB -J pnzL -o $OUT/paper_nz_LRG1.log --wrap "$ENV && python -u -m desi_validation.nz_scatter --bin LRG1 --workers 32")
    j3=$($SB -J pnzQ -o $OUT/paper_nz_QSO.log --wrap "$ENV && python -u -m desi_validation.nz_scatter --bin QSO --workers 32")
    echo "submitted paper_export: maps $j1, n(z) $j2 $j3"
    DEPS="$DEPS:$j1:$j2:$j3"
  fi
fi
if [ -n "$DEPS" ]; then
  jb=$($SB -t 00:10:00 -J pbundle --dependency=afterany$DEPS -o $OUT/paper_bundle.log \
       --wrap "$ENV && python -u -m desi_validation.paper_inputs --check && python -u -m desi_validation.paper_inputs --bundle")
  echo "bundle job $jb runs after them"
  echo "check:  squeue --me ; tail -n 20 $OUT/paper_bundle.log"
else
  ( eval "$ENV" && python -u -m desi_validation.paper_inputs --bundle )
fi
echo "then upload the newest:  ls -t $OUT/paper_raw_*.tgz | head -1"
