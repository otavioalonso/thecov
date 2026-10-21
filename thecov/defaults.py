"""Numerical defaults, in one place. Every class takes these as keyword defaults; change them per call, not here.

Pair counts of the windows (pair_counts.py, covariance.py, ssc.py)
    N_SUB        randoms per window for the far pairs (s >= S_SPLIT): all pairs, cost ~ N_SUB^2
    N_NEAR       randoms per window for the near pairs (s < S_SPLIT): KD-tree, cost ~ N_NEAR x density
    S_SPLIT      separation [Mpc/h] between the two regimes (snapped to a pair-count bin edge)
    DS_PAIR      width [Mpc/h] of the pair-count s bins; Q(s) is smooth, coarse is fine
    MIN_PAIRS    s bins with fewer pairs are dropped from the spline of Q(s)
    N_SHELLS     r1 cells of the (r1, s, mu) histogram (Q depends weakly on r1)
    N_MU         mu cells: None = automatic, max(24, 6 Lambda_max) (S oscillates as Pbar_Lambda(mu))
    CHUNK_PAIRS  pairs per vectorised block (memory, not accuracy)
    SEED         seed of the random subsamples

Radial integrals of the Gaussian term (covariance.py)
    DS           step [Mpc/h] of the s grid of the shell-kernel integrals

SSC (ssc.py): its own pair counts weight the Lambda = 4 window components more, which needs a finer mu grid
    SSC_N_MU, SSC_N_NEAR

Kernel smoothing of the clustering window (smoothing.py)
    KERNEL_CELL, KERNEL_R_SPLIT, KERNEL_TAPER, KERNEL_R_MAX, KERNEL_DR, KERNEL_TOL, KERNEL_MAX_KERNELS,
    KERNEL_DAMPING, KERNEL_K_MAX, KERNEL_LOW_SLOPE: see WindowSmoothing
"""
N_SUB = 5000
N_NEAR = 200_000
S_SPLIT = 80.0
DS_PAIR = 10.0
MIN_PAIRS = 20
N_SHELLS = 16
N_MU = None
CHUNK_PAIRS = 200_000
SEED = 0

DS = 2.0

SSC_N_MU = 96
SSC_N_NEAR = 300_000

KERNEL_CELL = 3.0
KERNEL_R_SPLIT = 8.0
KERNEL_TAPER = 2.0
KERNEL_R_MAX = 200.0
KERNEL_DR = 0.25
KERNEL_TOL = 2e-3
KERNEL_MAX_KERNELS = 48
KERNEL_DAMPING = 1.0
KERNEL_K_MAX = 5.0
KERNEL_LOW_SLOPE = 1.0
