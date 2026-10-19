"""On-disk cache of expensive, cosmology-independent results (pair counts, kernel smoothing).

One format for every module: an .npz with the arrays and a JSON 'meta' entry describing them.
    save(path, arrays, meta)      arrays: {name: ndarray}, meta: anything JSON-serialisable
    load(path) -> arrays, meta    arrays read into memory
    tag(*parts)                   short content hash, to recognise a cache made with the same settings
"""
import hashlib
import json

import numpy as np


def save(path, arrays, meta):
    np.savez(path, meta=np.array(json.dumps(meta)), **arrays)


def load(path):
    with np.load(path) as f:
        meta = json.loads(str(f['meta'].item()))
        arrays = {k: f[k] for k in f.files if k != 'meta'}
    return arrays, meta


def tag(*parts):
    return hashlib.md5(repr(parts).encode()).hexdigest()[:8]
