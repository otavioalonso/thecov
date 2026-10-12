#!/bin/bash
# Gaussian random fields on the real footprint (exact Gaussian covariance of the windowed P0), split over debug jobs:
#   bash ~/thecov/desi_validation/run_grf_tasks.sh LRG1 NGC [NTASKS=4] [NREAL=100] [CELL=6]
# Task 0 runs the pilot (calibration of the input power, shared by all tasks) and NREAL realisations; tasks 1..NTASKS-1
# start after it and add NREAL each. Output in ~/thecov_desi/holi_v3_mock173/<bin>/gaussian_footprint_<bin>_<cap>_c<cell>_n512_task<i>.npz,
# read by paper/thecov_walkthrough.ipynb (section 3d). If a job hits the 30-min limit, lower NREAL (the log prints the time
# per realisation) and resubmit the missing tasks with the same CELL.
set -e
BIN=${1:-LRG1}; CAP=${2:-NGC}; NTASKS=${3:-4}; NREAL=${4:-100}; CELL=${5:-6}
OUT=/global/cfs/cdirs/desicollab/users/oalves/thecov_validation
ENV="source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main && export PYTHONPATH=\$HOME/thecov:\$PYTHONPATH && cd \$HOME/thecov"
RUN="python -u -m desi_validation.run_gaussian_footprint --bin $BIN --regions $CAP --no-thecov --save-realisations --nreal $NREAL --cell $CELL"
SB="sbatch --parsable -N 1 -C cpu -q debug -t 00:30:00"
j0=$($SB -J grf${CAP}0 -o $OUT/grf_${BIN}_${CAP}_task0.log --wrap "$ENV && $RUN --task 0 --pilot 60")
echo "task 0: job $j0 (pilot + $NREAL realisations), log $OUT/grf_${BIN}_${CAP}_task0.log"
for t in $(seq 1 $((NTASKS - 1))); do
  j=$($SB -J grf${CAP}$t --dependency=afterok:$j0 -o $OUT/grf_${BIN}_${CAP}_task$t.log --wrap "$ENV && $RUN --task $t --pilot 60")
  echo "task $t: job $j (after $j0)"
done
echo "check: squeue --me ; grep -h 'realisations\|saved\|Error' $OUT/grf_${BIN}_${CAP}_task*.log"
