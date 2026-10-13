#!/bin/bash
# Self-chaining debug jobs for the DESI-randoms GRF test (desi_validation/grf_randoms.py), all tracers.
# Each job pops the next line of logs/grfr_queue.txt, runs it, and submits the next job of its chain while
# lines remain. Start N chains (N <= 2) with:  sbatch -N 1 -C cpu -q debug -t 00:30:00 -o ~/thecov/logs/grfr_chain_%j.log ~/thecov/desi_validation/run_grfr_chain.sh
# Lines:  fields BIN CAP TASK [args]  |  thecov BIN CAP MODE [args]  |  plot BIN [DIR]   (args: extra grf_randoms options,
# e.g. --out-dir; plot is re-queued until its 4 thecov outputs exist in DIR, default OUT/BIN/grf_randoms)
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 00:30:00
L=$HOME/thecov/logs; Q=$L/grfr_queue.txt; SELF=$HOME/thecov/desi_validation/run_grfr_chain.sh
OUT=$HOME/thecov_desi/holi_v3_mock173
exec 9>$Q.lock; flock 9; line=$(head -n 1 $Q); sed -i 1d $Q; flock -u 9
[ -z "$line" ] && { echo "queue empty"; exit 0; }
echo "[$(date +%T)] job ${SLURM_JOB_ID:-local}: $line"
set -- $line
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main >/dev/null 2>&1
module unload thecov 2>/dev/null
export PYTHONPATH="$HOME/thecov:$HOME/thecov/.pylib:${PYTHONPATH}"
cd $HOME/thecov
F="grep -v -e 'WEIGHT not in catalog' -e CUDA -e 'Jax plugin' -e xla_bridge -e _check_cuda -e 'for d in range' -e '\^\^\^' -e plugin_module"
case $1 in
  fields) python -u -m desi_validation.grf_randoms fields --bin $2 --region $3 --task $4 --nreal 200 --max-time 1400 ${@:5} 2>&1 | eval $F > $L/grfr_$2_$3_task$4.log ;;
  thecov) python -u -m desi_validation.grf_randoms thecov --bin $2 --region $3 --mode $4 --target-near-pairs 5e9 ${@:5} 2>&1 | eval $F > $L/grfr_$2_$3_thecov_$4.log ;;
  run)    python -u -m desi_validation.${@:2} 2>&1 | eval $F > $L/run_$(echo ${@:2} | tr ' /' '__' | cut -c1-120).log ;;
  plot)   D=${3:-$OUT/$2/grf_randoms}; T=${4:-}; [ "$T" = "-" ] && T=""; NEED=${5:-4}
          if [ $(ls $D/thecov_$2_*$T_*.npz 2>/dev/null | wc -l) -ge $NEED ]; then
            python -u desi_validation/grf_randoms_plot.py --bin $2 --dir $D --tag "$T" --out $L/grf_randoms_$2$T.png > $L/grf_randoms_$2$T.txt 2>&1
          else
            n=$(grep -c "^$line\$" $L/grfr_requeued.txt 2>/dev/null || echo 0)
            [ "$n" -lt 4 ] && { echo "$line" >> $Q; echo "$line" >> $L/grfr_requeued.txt; echo "requeued $line"; }
          fi ;;
esac
echo "[$(date +%T)] finished: $line (exit ${PIPESTATUS[0]})"
[ -s $Q ] && sbatch -o $L/grfr_chain_%j.log $SELF
