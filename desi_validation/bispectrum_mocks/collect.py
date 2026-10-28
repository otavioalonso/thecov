"""Collect Sugiyama-diagonal B000/B202 (+ P0/P2 up to k = 0.3) for every realisation of the chosen suites into one npz
per suite.  usage: python collect.py suite [suite ...]   (uses all CPUs given by SLURM_CPUS_PER_TASK)"""
import os, re, sys, json, time
from multiprocessing import Pool
import numpy as np, h5py
BASE = '/dvs_ro/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/full_shape/base'
TRACERS = [('LRG', 'z0.4-0.6'), ('LRG', 'z0.6-0.8'), ('LRG', 'z0.8-1.1'), ('ELG_LOPnotqso', 'z0.8-1.1'),
           ('ELG_LOPnotqso', 'z1.1-1.6'), ('QSO', 'z0.8-2.1'), ('ELG_LOP', 'z0.8-1.1'), ('ELG_LOP', 'z1.1-1.6')]
REGIONS = ('NGC', 'SGC', 'GCcomb')
WEIGHT = 'weight-default-FKP'
KMAX_P = 0.3


def read_one(args):
    path_b, path_p = args
    out = {}
    try:
        with h5py.File(path_b, 'r') as f:
            for ell in ('0_0_0', '2_0_2'):
                g = f[ell]
                out[f'B{ell.replace("_", "")}'] = g['value'][()]
                out[f'sn{ell.replace("_", "")}'] = g['num_shotnoise'][()]
                out[f'norm{ell.replace("_", "")}'] = g['norm'][()]
            out['wsum_data'] = float(np.ravel(f.attrs['wsum_data'])[0]) if 'wsum_data' in f.attrs else np.nan
            out['wsum_randoms'] = float(np.ravel(f.attrs['wsum_randoms'])[0]) if 'wsum_randoms' in f.attrs else np.nan
    except Exception as e:
        return None, f'B read error {path_b}: {e}'
    if path_p is not None and os.path.exists(path_p):
        try:
            with h5py.File(path_p, 'r') as f:
                k = f['0']['k'][()]
                sel = k < KMAX_P
                for ell in ('0', '2', '4'):
                    out[f'P{ell}'] = f[ell]['value'][()][sel]
                out['normP'] = float(np.ravel(f['0']['norm'][()])[0]); out['snP'] = float(np.ravel(f['0']['num_shotnoise'][()])[0])
        except Exception as e:
            out['P_error'] = str(e)
    return out, None


def meta_of(path_b, path_p):
    m = {}
    with h5py.File(path_b, 'r') as f:
        g = f['0_0_0']
        m['k_b'] = g['k'][()][:, 0]; m['k_edges_b'] = g['k_edges'][()][:, 0, :]; m['nmodes_b'] = g['nmodes'][()]
        m['boxsize'] = np.asarray(f.attrs.get('boxsize', np.nan)); m['meshsize'] = np.asarray(f.attrs.get('meshsize', np.nan))
    if path_p is not None and os.path.exists(path_p):
        with h5py.File(path_p, 'r') as f:
            k = f['0']['k'][()]; sel = k < KMAX_P
            m['k_p'] = k[sel]; m['k_edges_p'] = f['0']['k_edges'][()][sel]; m['nmodes_p'] = f['0']['nmodes'][()][sel]
    return m


def main(suite, nproc):
    t0 = time.time()
    root = os.path.join(BASE, suite)
    dirs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith('dubious'))
    files = {d: set(os.listdir(os.path.join(root, d))) for d in dirs}
    print(f'{suite}: {len(dirs)} realisation dirs listed in {time.time() - t0:.0f}s', flush=True)
    arrays, log = {}, {'suite': suite, 'groups': {}, 'errors': []}
    with Pool(nproc) as pool:
        for tr, z in TRACERS:
            for reg in REGIONS:
                fb = f'mesh3_spectrum_sugiyama-diagonal_poles_{tr}_{z}_{reg}_{WEIGHT}.h5'
                fp = f'mesh2_spectrum_poles_{tr}_{z}_{reg}_{WEIGHT}.h5'
                have = [d for d in dirs if fb in files[d]]
                if not have:
                    continue
                jobs = [(os.path.join(root, d, fb), os.path.join(root, d, fp) if fp in files[d] else None) for d in have]
                res = pool.map(read_one, jobs, chunksize=8)
                ok = [(d, r) for d, (r, err) in zip(have, res) if r is not None]
                log['errors'] += [err for r, err in res if err]
                key = f'{tr}_{z}_{reg}'
                meta = meta_of(*jobs[0])
                for k_, v in meta.items():
                    arrays[f'{key}__{k_}'] = v
                arrays[f'{key}__ids'] = np.array([d for d, _ in ok])
                for name in ('B000', 'B202', 'sn000', 'sn202', 'norm000', 'norm202'):
                    arrays[f'{key}__{name}'] = np.array([r[name] for _, r in ok], dtype=np.float64)
                for name in ('wsum_data', 'wsum_randoms'):
                    arrays[f'{key}__{name}'] = np.array([r[name] for _, r in ok])
                hasP = [d for d, r in ok if 'P0' in r]
                if len(hasP) == len(ok):
                    for name in ('P0', 'P2', 'P4'):
                        arrays[f'{key}__{name}'] = np.array([r[name] for _, r in ok], dtype=np.float32)
                    for name in ('normP', 'snP'):
                        arrays[f'{key}__{name}'] = np.array([r[name] for _, r in ok])
                log['groups'][key] = dict(n=len(ok), n_with_P=len(hasP), n_dirs=len(have))
                print(f'  {key}: {len(ok)} read ({len(hasP)} with P) {time.time() - t0:.0f}s', flush=True)
    np.savez_compressed(f'bispec_{suite}.npz', **arrays)
    json.dump(log, open(f'bispec_{suite}.json', 'w'), indent=1)
    print(f'{suite} done in {time.time() - t0:.0f}s, {len(log["errors"])} errors', flush=True)


if __name__ == '__main__':
    nproc = int(os.environ.get('SLURM_CPUS_PER_TASK', '8'))
    for s in sys.argv[1:]:
        main(s, nproc)
