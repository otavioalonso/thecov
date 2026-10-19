"""Covariance templates with free amplitudes, for fitting non-Gaussian terms to mocks or subvolumes (validation tool)."""
import numpy as np


class CovarianceTemplates:
    """C(amplitudes) = C_fixed + sum_i A_i C_i: the container for fitting the amplitudes of non-Gaussian
    terms (e.g. to subvolumes of the data). Templates are added by name; unspecified amplitudes are 1."""

    def __init__(self, fixed=None):
        self.fixed = fixed
        self.templates = {}

    def add(self, name, C):
        self.templates[str(name)] = np.asarray(C, float)
        return self

    def update(self, comps, prefix=''):
        for name, C in comps.items():
            self.add(prefix + name, C)
        return self

    @property
    def names(self):
        return list(self.templates)

    def __call__(self, **amplitudes):
        unknown = set(amplitudes) - set(self.templates)
        if unknown:
            raise KeyError(f'unknown templates {sorted(unknown)}; have {self.names}')
        C = 0.0 if self.fixed is None else np.array(self.fixed, float)
        for name, T in self.templates.items():
            C = C + amplitudes.get(name, 1.0) * T
        return C

    def loglike(self, S, n_samples, **amplitudes):
        """-2 ln L (up to a constant) of a sample covariance S of n_samples vectors under C(amplitudes)
        (Wishart: n [tr(C^-1 S) + ln det C]); +inf if C is not positive definite."""
        C = self(**amplitudes)
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            return np.inf
        Li = np.linalg.inv(L)
        return float(n_samples * (np.sum((Li @ np.asarray(S) @ Li.T).diagonal()) + 2 * np.sum(np.log(np.diag(L)))))

    def fit(self, S, n_samples, free=None, start=None, fixed=None, restarts=3):
        """Maximum-likelihood amplitudes of the templates `free` (default: all) for a sample covariance S
        (e.g. of mocks or of sub-volumes); the others are held at `fixed` (default 1).
        Returns (amplitudes dict, -2 ln L)."""
        from scipy.optimize import minimize
        free = self.names if free is None else list(free)
        base = {n: 1.0 for n in self.names}
        base.update(fixed or {})
        x0 = np.array([(start or {}).get(n, base[n]) for n in free], float)

        def f(x):
            amps = dict(base, **dict(zip(free, x)))
            v = self.loglike(S, n_samples, **amps)
            return v if np.isfinite(v) else 1e300
        if f(x0) >= 1e300:                                     # start not positive definite: start small
            x0 = np.full_like(x0, 0.1)
        best = None
        for i in range(max(1, restarts)):
            r = minimize(f, x0 if i == 0 else x0 * (1 + 0.3 * i) + 0.1 * i, method='Powell',
                         options=dict(maxiter=20000, xtol=1e-4, ftol=1e-8))
            if best is None or r.fun < best.fun:
                best = r
        amps = dict(base, **dict(zip(free, np.atleast_1d(best.x))))
        return amps, float(best.fun)

    def save(self, fn):
        np.savez(fn, fixed=np.asarray(self.fixed if self.fixed is not None else np.nan), names=np.array(self.names),
                 **{f't_{n}': T for n, T in self.templates.items()})

    @classmethod
    def load(cls, fn):
        with np.load(fn) as z:
            fixed = z['fixed']
            out = cls(None if fixed.ndim == 0 else fixed)
            for n in z['names']:
                out.add(str(n), z[f't_{n}'])
        return out
