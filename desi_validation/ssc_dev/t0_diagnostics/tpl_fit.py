import numpy as np, sys
from scipy.optimize import minimize
from desi_validation.compare_kernel import get, chi2
from desi_validation.ssc_check import summarize
S='/tmp/claude-0/-home-user-thecov/90e4caa1-064c-5b2e-b2e3-4ff61a970c03/scratchpad'
z=np.load(S+'/t0run/report_data_holi-kcore2-LRG1.npz')
V,C,k=get(z,'LRG1','NGC',1,'kernel'); nb=len(k); n=3*nb; N=len(V); Cm=np.cov(V.T)
A=0.725; t0=np.load(S+'/t0_fog_NGC_s2.09.npz')
T0=A**3*(t0['snake']+t0['star']); Tb3=A**3*t0['star_b3']
amp=np.load(S+'/sd_amp_NGC.npz'); SSC,DISC=amp['ssc'],amp['disc']
ks=float(sys.argv[1]) if len(sys.argv)>1 else 0.06
Lm=np.tile(k<ks,3); H=~Lm
blk=lambda X,a,b: np.where(np.outer(a,b)|np.outer(b,a),X,0.0)
T_LL=blk(T0,Lm,Lm); T_LH=blk(T0,Lm,H); T_HH=np.where(np.outer(H,H),T0,0.0)
SLL=(C+SSC+DISC)[np.ix_(Lm,Lm)]
comp=np.zeros((n,n)); X=T0[np.ix_(Lm,H)]; comp[np.ix_(H,H)]=X.T@np.linalg.solve(SLL,X)
tpl={'ssc':SSC,'disc':DISC,'T_LL':T_LL,'T_LH':T_LH,'T_HH':T_HH,'dstar_db3':Tb3}
def model(p):
    d=dict(zip(tpl,p)); M=C.copy()
    for kname,T in tpl.items(): M=M+d[kname]*T
    return M + d['T_LH']**2*comp
def nll(p):
    M=model(p)
    try: L=np.linalg.cholesky(M)
    except np.linalg.LinAlgError: return 1e30
    Mi=np.linalg.inv(M)
    return 0.5*N*(np.sum(Mi*Cm)+2*np.sum(np.log(np.diag(L))))
for name,free,p0 in [('physical (all 1, b3 Lazeyras)',[],np.ones(6)*np.r_[1,1,1,1,1,0]),
                     ('b3 free',['dstar_db3'],np.r_[1,1,1,1,1,0.]),
                     ('all free',list(tpl),np.r_[1,1,1,1,1,0.])]:
    idx=[list(tpl).index(f) for f in free]
    f=lambda q: nll(np.where(np.isin(np.arange(6),idx), np.put(p0.copy(),idx,q) or 0, p0)) if False else None
    def obj(q):
        p=p0.copy(); p[idx]=q; return nll(p)
    if idx:
        r=minimize(obj,p0[idx],method='Nelder-Mead',options=dict(maxiter=4000,xatol=1e-3,fatol=1e-2))
        p=p0.copy(); p[idx]=r.x
    else: p=p0
    print(f'== {name}: ', ', '.join(f'{a}={b:.2f}' for a,b in zip(tpl,p)), f'(d b3 = {p[5]:+.2f})')
    summarize(f'{name[:28]:28s}', V, C, model(p)-C, k, nb)

print('\n-2 ln L relative to Gaussian-only (lower is better; each extra parameter should buy > ~2-10):')
ref=nll(np.zeros(6))  # careful: model(0)=C
def fit(free, p0):
    idx=[list(tpl).index(f) for f in free]
    def obj(q):
        p=p0.copy(); p[idx]=q; return nll(p)
    best=None
    for start in (p0[idx], p0[idx]+0.3, p0[idx]*0.5+0.5):
        r=minimize(obj,start,method='Powell',options=dict(maxiter=20000,xtol=1e-4,ftol=1e-6))
        if best is None or r.fun<best.fun: best=r
    p=p0.copy(); p[idx]=best.x; return p, best.fun
cases=[('SSC+disc fixed 1',[],np.r_[1,1,0,0,0,0.]),
       ('SSC+disc+T0 fixed 1',[],np.r_[1,1,1,1,1,0.]),
       ('ssc, disc free',['ssc','disc'],np.r_[1,1,0,0,0,0.]),
       ('ssc, disc, T0 (one amp) free',None,None),
       ('ssc, disc, b3 free + T0 fixed',['ssc','disc','dstar_db3'],np.r_[1,1,1,1,1,0.]),
       ('all free',list(tpl),np.r_[1,1,1,1,1,0.])]
for name,free,p0 in cases:
    if free is None:
        def obj(q):
            p=np.r_[q[0],q[1],q[2],q[2],q[2],0.]; return nll(p)
        r=minimize(obj,[1,1,1],method='Powell'); p=np.r_[r.x[0],r.x[1],r.x[2],r.x[2],r.x[2],0.]; v=r.fun
    elif free: p,v=fit(free,p0)
    else: p,v=p0,nll(p0)
    M=model(p)
    print(f'  {name:32s} -2dlnL {2*(v-ref):9.1f}  chi2/n {chi2(V,M,np.arange(n)):.4f}  params', ', '.join(f'{a}={b:.2f}' for a,b in zip(tpl,p)))
