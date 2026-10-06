import numpy as np, camb
def plin(z, kmin=1e-4, kmax=10, n=3000):
    p = camb.CAMBparams()
    p.set_cosmology(H0=67.66, ombh2=0.02242, omch2=0.11933, mnu=0.06, omk=0)
    p.InitPower.set_params(As=2.105e-9, ns=0.9665)
    p.set_matter_power(redshifts=[z], kmax=kmax * 1.2)
    r = camb.get_results(p)
    kh, _, pk = r.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints=n)
    f = r.get_fsigma8()[0] / r.get_sigma8()[0]
    return kh, pk[0], f
if __name__ == '__main__':
    k, P, f = plin(0.51); np.savez('/tmp/claude-0/-home-user-thecov/90e4caa1-064c-5b2e-b2e3-4ff61a970c03/scratchpad/plin_z051.npz', k=k, P=P, f=f); print(f, P[np.searchsorted(k, 0.1)])
