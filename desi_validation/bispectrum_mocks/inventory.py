"""Inventory of Sugiyama bispectrum (mesh3) files per mock suite under the DR2 full_shape base tree."""
import os, re, json, time, collections
import h5py, numpy as np
BASE = '/dvs_ro/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/full_shape/base'
suites = [s for s in sorted(os.listdir(BASE)) if any(t in s for t in ('holi', 'glam', 'abacus', 'uchuu'))]
pat = re.compile(r'mesh3_spectrum_(?P<basis>[a-z\-]+)_poles_(?P<tracer>.+?)_(?P<z>z[0-9.]+-[0-9.]+)_(?P<region>NGC|SGC|GCcomb)_(?P<rest>.+)\.h5$')
out = {}
t0 = time.time()
for s in suites:
    root = os.path.join(BASE, s)
    try:
        subs = sorted(os.listdir(root))
    except Exception as e:
        out[s] = {'error': str(e)}; continue
    real = [d for d in subs if os.path.isdir(os.path.join(root, d))]
    counts = collections.Counter(); ids = collections.defaultdict(list); sample = {}
    has_pk = collections.Counter()
    for d in real:
        try:
            files = os.listdir(os.path.join(root, d))
        except Exception:
            continue
        fs = set(files)
        for f in files:
            m = pat.match(f)
            if not m:
                continue
            key = (m['basis'], m['tracer'], m['z'], m['region'], m['rest'])
            counts[key] += 1; ids[key].append(d)
            pk = f'mesh2_spectrum_poles_{m["tracer"]}_{m["z"]}_{m["region"]}_{m["rest"]}.h5'
            if pk in fs:
                has_pk[key] += 1
            if key not in sample:
                sample[key] = os.path.join(root, d, f)
    info = {'n_subdirs': len(real), 'subdir_examples': real[:5] + real[-3:], 'n_dubious': sum(d.startswith('dubious') for d in real), 'groups': []}
    for key, n in sorted(counts.items()):
        g = dict(basis=key[0], tracer=key[1], z=key[2], region=key[3], rest=key[4], n=n, n_with_pk=has_pk[key],
                 n_dubious=sum(x.startswith('dubious') for x in ids[key]), example=sample[key])
        try:
            with h5py.File(sample[key], 'r') as f:
                g['multipoles'] = [x.decode() if isinstance(x, bytes) else str(x) for x in f['labels_values'][()]]
                g0 = f[g['multipoles'][0]]
                ke = g0['k_edges'][()]
                g['k_edges_shape'] = list(ke.shape); g['k_first'] = ke[0].tolist(); g['k_last'] = ke[-1].tolist()
                g['boxsize'] = f.attrs['boxsize'].tolist(); g['meshsize'] = f.attrs['meshsize'].tolist(); g['los'] = str(f.attrs.get('los'))
        except Exception as e:
            g['read_error'] = str(e)
        info['groups'].append(g)
    out[s] = info
    print(f'{s}: {len(real)} subdirs, {len(counts)} mesh3 groups, {time.time() - t0:.0f}s', flush=True)
json.dump(out, open('inventory.json', 'w'), indent=1)
