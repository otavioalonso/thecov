import json, numpy as np
from data import run
R = json.load(open('results.json'))
out = {}
# chi2 table
rows = []
for name, lab in (('holi', 'holi'), ('complete', 'Ab.\\ compl.'), ('altmtl', 'Ab.\\ altmtl')):
    z, m = run(name)
    for b, mb in m['bins'].items():
        c = mb['chi2']
        for r in ('NGC', 'SGC', 'GCcomb'):
            pm = 'combined-regions' if r == 'GCcomb' else 'random-density'
            nxm = 'combined-regions [nx]' if r == 'GCcomb' else 'nx'
            f = lambda key: '%.3f' % c[key][0] if key in c else '--'
            N = 859 if name == 'holi' else 25
            err = np.sqrt(2 / (168 * N))
            rows.append(f"{lab} & {b} & {r} & {f(f'{r} | {pm}')} & {f(f'{r} | {nxm}')} & {f(f'{r} | none')} & "
                        f"{f(f'{r} | {pm}, bins x2')} & {f(f'{r} | {pm}, bins x4')} & {err:.3f}\\\\")
out['chi2'] = '\n'.join(rows)
# weights table
rows = []
for key, w in R['weights'].items():
    name, b, r = key.split()
    lab = {'holi': 'holi', 'complete': 'Ab.\\ compl.'}[name]
    rows.append(f"{lab} & {b} & {r} & {w['w2_data']:.3f} & {w['I_over_norm_smooth']:.3f} & {w['I_over_norm_naive']:.3f} & "
                f"{w['chi2_smooth']:.3f} & {w['chi2_naive']:.3f}\\\\")
out['weights'] = '\n'.join(rows)
# params table
rows = []
for key, v in R['params'].items():
    name, b, r, kmax = key.split('|')
    if name != 'holi' and r != 'GCcomb':
        continue
    lab = {'holi': 'holi', 'complete': 'Ab.\\ compl.', 'altmtl': 'Ab.\\ altmtl'}[name]
    j, s = v['joint'], v['single']
    rows.append(f"{lab} & {b} & {r} & {kmax} & " + ' & '.join('%.2f' % x for x in s[:3]) + ' & ' + ' & '.join('%.2f' % x for x in j) + '\\\\')
out['params'] = '\n'.join(rows)
for k, v in out.items():
    open(f'tab_{k}.tex', 'w').write(v + '\n')
a = R['amplitude']
for b in a:
    print(b, a[b]['norm'])
print(json.dumps({b: {k: [round(x['excess'], 4) for x in v] for k, v in a[b].items() if k != 'norm'} for b in a}, indent=0))
