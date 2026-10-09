#!/bin/bash
# NERSC inputs of the validation paper (paper/README.md): sky maps of the hole fill fraction and weights, the hole-fraction
# statistics, and the n(z) of all 859 holi mocks. Two debug jobs, then one tarball to upload:
#   bash ~/thecov/desi_validation/paper_export.sh            # submit
#   bash ~/thecov/desi_validation/paper_export.sh --bundle   # after both jobs finish: pack $OUT/paper_inputs_<date>.tgz
set -e
OUT=/global/cfs/cdirs/desicollab/users/oalves/thecov_validation
ENV="source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main && export PYTHONPATH=\$HOME/thecov:\$PYTHONPATH && cd \$HOME/thecov"
SB="sbatch --parsable -N 1 -C cpu -q debug -t 00:30:00"
if [ "$1" == "--bundle" ]; then
  ( eval "$ENV" && python -u -m desi_validation.paper_export --slim )
  STAGE=$(mktemp -d)
  for f in paper_maps_LRG1.npz paper_maps_QSO.npz hole_fraction_LRG1.json hole_fraction_QSO.json nz_scatter_LRG1.npz \
           nz_scatter_QSO.npz paper_maps.log paper_nz_LRG1.log paper_nz_QSO.log; do
    [ -e $OUT/$f ] && cp $OUT/$f $STAGE/ || echo "missing $OUT/$f"
  done
  TGZ=$OUT/paper_inputs_$(date +%Y%m%d_%H%M).tgz
  tar czf $TGZ -C $STAGE . && rm -rf $STAGE && ls -lh $TGZ
  exit 0
fi
cd $HOME/thecov && git pull origin thecov2
j1=$($SB -J pmaps -o $OUT/paper_maps.log --wrap "$ENV && python -u -m desi_validation.paper_export --bins LRG1 QSO && \
  ( [ -e $OUT/hole_fraction_LRG1.json ] || python -u -m desi_validation.hole_fraction --bins LRG1 QSO )")
j2=$($SB -J pnzL -o $OUT/paper_nz_LRG1.log --wrap "$ENV && python -u -m desi_validation.nz_scatter --bin LRG1 --workers 32")
j3=$($SB -J pnzQ -o $OUT/paper_nz_QSO.log --wrap "$ENV && python -u -m desi_validation.nz_scatter --bin QSO --workers 32")
echo "submitted: maps $j1, n(z) LRG1 $j2, n(z) QSO $j3"
echo "check:  squeue --me ; tail -n 5 $OUT/paper_maps.log $OUT/paper_nz_LRG1.log $OUT/paper_nz_QSO.log"
echo "then:   bash ~/thecov/desi_validation/paper_export.sh --bundle"
