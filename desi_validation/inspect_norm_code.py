"""Print how the installed jaxpower / clustering_statistics compute the power-spectrum normalisation, and what one
spectrum file stores about it. Run at NERSC inside the cosmodesi environment (no job needed, ~1 min):

    source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
    cd ~/thecov && python -m desi_validation.inspect_norm_code [--bin LRG1] [--region NGC] [--mock 173]
        > /global/cfs/cdirs/desicollab/users/oalves/thecov_validation/norm_code.txt

jaxpower (github.com/adematti/jax-power, jaxpower/mesh2.py) has two conventions in compute_fkp2_normalization:
  split=None : norm = alpha x sum_cells D_c R_c / V_c  (CIC, cellsize 10; "the pypower normalization": data x randoms)
  split=seed : norm = alpha x sum_cells R1_c R2_c / V_c  (two disjoint random subsamples, rescaled: randoms only)
num_shotnoise = sum (w_d - alpha w_r)^2 over coinciding particles = sum_d w_d^2 + alpha^2 sum_r w_r^2.
Which one the DESI spectra used is decided by the call in clustering_statistics: this script prints it.

What to read in the output:
  1. every function whose name contains 'norm' in jaxpower and in clustering_statistics, with its source: look for
     which meshes are multiplied (data x randoms, randoms x randoms, or a sum over objects of w^2 nbar) and whether
     the data enter;
  2. the lines of clustering_statistics that call jaxpower with a normalisation option (the spectra pipeline's choice);
  3. the attributes stored in one mesh2_spectrum_poles file (norm, num_shotnoise, any 'norm_*' / 'attrs' entries).
"""
from __future__ import annotations

import argparse
import ast
import importlib
import os
import re
import sys

sys.path.insert(0, os.environ.get('THECOV_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

PATTERN = re.compile(r'norm', re.IGNORECASE)


def package_files(mod):
    root = os.path.dirname(os.path.abspath(mod.__file__))
    out = []
    for d, _, fns in os.walk(root):
        out += [os.path.join(d, f) for f in fns if f.endswith('.py')]
    return root, sorted(out)


def functions_named(fn, pattern=PATTERN):
    """(name, source) of every def / class method in fn whose name matches"""
    src = open(fn).read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    lines = src.splitlines()
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and pattern.search(node.name):
            end = getattr(node, 'end_lineno', node.lineno + 40)
            out.append((node.name, node.lineno, '\n'.join(lines[node.lineno - 1:end])))
    return out


def grep(fn, pattern, context=3, max_hits=40):
    lines = open(fn).read().splitlines()
    hits = [i for i, l in enumerate(lines) if pattern.search(l)]
    out = []
    for i in hits[:max_hits]:
        lo, hi = max(0, i - context), min(len(lines), i + context + 1)
        out.append(f'  {os.path.basename(fn)}:{i + 1}\n' + '\n'.join(f'    {j + 1:5d} {lines[j]}' for j in range(lo, hi)))
    return out


def report_package(name, call_pattern=None):
    print(f'\n{"=" * 100}\n{name}\n{"=" * 100}')
    try:
        mod = importlib.import_module(name)
    except Exception as e:
        print(f'  not importable: {type(e).__name__}: {e}')
        return
    root, fns = package_files(mod)
    print(f'  {mod.__file__}  version {getattr(mod, "__version__", "?")}; {len(fns)} files under {root}')
    for fn in fns:
        for fname, lineno, src in functions_named(fn):
            print(f'\n--- {os.path.relpath(fn, root)}:{lineno}  def {fname}\n{src}')
    if call_pattern is not None:
        print(f'\n--- lines matching {call_pattern.pattern!r}:')
        for fn in fns:
            for block in grep(fn, call_pattern):
                print(block)


def report_spectrum_file(args):
    from desi_validation import desi_compare as dc
    paths = dc.Paths(kind='holi_v3', mock=args.mock)
    tracer, zr = dc.TRACER_SPECS[args.bin]
    fns = [fn for fn in paths.spectra_fns(tracer, zr, args.region) if f'mock{args.mock}' in fn.split(os.sep)]
    fns = fns or paths.spectra_fns(tracer, zr, args.region)[:1]
    print(f'\n{"=" * 100}\nspectrum file {fns[0] if fns else "NOT FOUND"}\n{"=" * 100}')
    if not fns:
        return
    import lsstypes as types
    s = types.read(fns[0])
    print('  type', type(s), '\n  repr', repr(s)[:2000])
    for attr in ('attrs', 'meta', 'metadata', '__dict__'):
        v = getattr(s, attr, None)
        if v:
            print(f'  {attr}: {str(v)[:3000]}')
    p0 = s.get(0)
    print('  values stored for ell = 0:', getattr(p0, 'values', lambda: None)() if callable(getattr(p0, 'values', None)) else '?')
    for key in ('norm', 'num_shotnoise', 'shotnoise', 'nmodes', 'num_zero', 'norm_randoms', 'norm_data'):
        try:
            v = p0.values(key)
            import numpy as np
            print(f'    {key}: shape {np.shape(v)} first {np.ravel(v)[:3]}')
        except Exception as e:
            print(f'    {key}: not stored ({type(e).__name__})')
    for attr in ('attrs', 'meta'):
        v = getattr(p0, attr, None)
        if v:
            print(f'  ell = 0 {attr}: {str(v)[:3000]}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bin', default='LRG1')
    ap.add_argument('--region', default='NGC')
    ap.add_argument('--mock', type=int, default=173)
    ap.add_argument('--no-file', action='store_true', help='skip reading the spectrum file')
    args = ap.parse_args()
    report_package('jaxpower', call_pattern=re.compile(r'def compute_fkp2_normalization|def compute_normalization|split', re.IGNORECASE))
    # the spectra pipeline's call decides the convention: compute_fkp2_normalization(fkp, split=None) is
    # alpha sum_cells D_c R_c / V_c (data x randoms, 'data-randoms'); with split=<seed> it is alpha times the product
    # of two disjoint random subsamples (randoms only: alpha realised once, 'alpha')
    report_package('clustering_statistics', call_pattern=re.compile(r'norm|split', re.IGNORECASE))
    report_package('pypower', call_pattern=None)
    if not args.no_file:
        try:
            report_spectrum_file(args)
        except Exception as e:
            print(f'spectrum file: failed: {type(e).__name__}: {e}')


if __name__ == '__main__':
    main()
