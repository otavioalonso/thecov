"""Figure: amplitudes of the non-Gaussian templates preferred by the mocks (Wishart maximum likelihood, 1-sigma Fisher
errors), C = C_G + sum_a theta_a T_a, for the holi (859) and AbacusSummit complete (25) mocks. theta = 1 is the
prediction. Three fits: SSC + discreteness; + tree-level T0; + response T0."""
from common import *

FITS = [('SSC + disc', ('ssc', 'disc')), (r'+ tree $T_0$', ('ssc', 'disc', 'T0_tree')),
        (r'+ response $T_0$', ('ssc', 'disc', 'T0_response'))]
TCOL = {'ssc': ORANGE, 'disc': AQUA, 'T0_tree': RED, 'T0_response': VIOLET}
TMARK = {'ssc': 'o', 'disc': 'o', 'T0_tree': 'D', 'T0_response': 's'}
XL = (-1.0, 3.0)
TLAB = {'ssc': 'SSC', 'disc': 'discreteness', 'T0_tree': r'tree $T_0$', 'T0_response': r'response $T_0$'}
sets = []
for b in ('LRG1', 'QSO'):
    for cap in ('NGC', 'SGC'):
        d = holi(b, cap)
        sets.append((f'holi {b} {cap}', d['V'], d['C']['G'], d['terms']))
z = load('abacus_LRG1')
for cap in ('NGC', 'SGC'):
    p = f'kernel/{cap}/x1/'
    terms = {key: z[p + 'C_' + key] for key in ('ssc', 'disc', 'T0_tree', 'T0_response') if p + 'C_' + key in z.files}
    sets.append((f'Abacus LRG1 {cap}', z[p + 'V'], z[p + 'C_G'], terms))

out = {}
fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.9), sharey=True)
ys = np.arange(len(sets))[::-1]
for ax, (title, keys) in zip(axes, FITS):
    for y, (name, V, CG, terms) in zip(ys, sets):
        if not all(t in terms for t in keys):
            continue
        th, sg, dchi = fit_templates(V, CG, {t: terms[t] for t in keys})
        out[f'{name}/{title}'] = dict(theta=th, sigma=sg, m2dlnL=dchi)
        for j, t in enumerate(keys):
            off = (j - (len(keys) - 1) / 2) * 0.22
            x = float(np.clip(th[t], XL[0] + 0.08, XL[1] - 0.08))
            if x != th[t]:     # outside the axis: an arrow at the edge, value printed
                ax.annotate('', (x - 0.05 * np.sign(th[t]), y + off), (x + 0.25 * np.sign(th[t]) * -1, y + off),
                            arrowprops=dict(arrowstyle='->', color=TCOL[t], lw=1.0))
                ax.text(x + 0.3 * (1 if th[t] < 0 else -1), y + off + 0.05, f'{th[t]:.1f}', fontsize=6, color=INK2,
                        ha='left' if th[t] < 0 else 'right', va='center')
                continue
            ax.errorbar(th[t], y + off, xerr=sg[t], fmt=TMARK[t], ms=3.2, lw=1.0, color=TCOL[t],
                        label=TLAB[t] if y == ys[0] else None)
    ax.axvline(1, color=INK2, lw=0.7)
    ax.axvline(0, color=BASE, lw=0.6)
    ax.set_title(title, fontsize=8)
    ax.set_xlim(*XL)
    ax.set_xlabel(r'amplitude $\theta$')
    ax.grid(axis='y', visible=False)
axes[0].set_yticks(ys, [s[0] for s in sets])
handles = {}
for ax in axes:
    for h, l in zip(*ax.get_legend_handles_labels()):
        handles.setdefault(l, h)
fig.legend(handles.values(), handles.keys(), loc='upper center', ncol=4, bbox_to_anchor=(0.55, 1.06), fontsize=7)
fig.tight_layout(w_pad=0.4)
save(fig, 'templates')
write_numbers('templates', {k: {kk: (vv if not isinstance(vv, dict) else {a: float(b) for a, b in vv.items()})
                                for kk, vv in v.items()} for k, v in out.items()})
