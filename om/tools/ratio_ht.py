"""Clyde--Ghosh (2012, Biometrika 99:981--988), Section 4 ratio HT.

The caller supplies every eighth primary state and one independent auxiliary
state per block. No burn-in, thinning, sampling or budget policy is hidden here.
Invalid plug-in probabilities are reported, never clipped or replaced.
"""
from collections import Counter
import numpy as np


def inclusion_probabilities(probabilities, draws):
    p = np.asarray(probabilities, dtype=float)
    if draws < 1 or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Inclusion approximation requires draws >= 1 and probabilities in [0, 1].')
    with np.errstate(divide='ignore'):
        return -np.expm1(draws * np.log1p(-p))


def ratio_weights(target_weights, probabilities, draws):
    """Return normalized ratio-HT weights and approximate inclusion probabilities."""
    u = np.asarray(target_weights, dtype=float)
    if not np.isfinite(u).all() or (u <= 0).any():
        raise ValueError('Observed states must have finite, positive target weights.')
    pi = inclusion_probabilities(probabilities, draws)
    if (pi <= 0).any():
        raise ValueError('Zero inclusion probability on the observed support.')
    logw = np.log(u) - np.log(pi)
    w = np.exp(logw - logw.max())
    return w / w.sum(), pi


class RatioHT:
    """Incrementally maintain unique supports and the primary overlap frequency."""
    def __init__(self, target_weights, function_values):
        self.u = np.asarray(target_weights, dtype=float)
        self.f = np.asarray(function_values, dtype=float)
        if self.u.shape != self.f.shape:
            raise ValueError('Target and function arrays must have identical shapes.')
        self.primary_counts = Counter()
        self.auxiliary = set()
        self.auxiliary_mass = 0.
        self.overlap_count = 0
        self.draws = 0

    def update(self, retained_primary, auxiliary):
        if len(retained_primary) != len(auxiliary):
            raise ValueError('Each block requires one retained primary and one auxiliary state.')
        for p, a in zip(retained_primary, auxiliary):
            p, a = int(p), int(a)
            if a not in self.auxiliary:
                self.auxiliary.add(a)
                self.auxiliary_mass += self.u[a]
                self.overlap_count += self.primary_counts[a]
            self.primary_counts[p] += 1
            self.overlap_count += int(p in self.auxiliary)
            self.draws += 1

    def estimate(self):
        ids = np.fromiter(self.primary_counts, dtype=np.int64)
        c_hat = self.overlap_count / self.draws / self.auxiliary_mass
        p = c_hat * self.u[ids]
        status = 'valid'
        if self.overlap_count == 0:
            status = 'zero_overlap'
        elif not np.isfinite(p).all() or (p > 1).any() or (p < 0).any():
            status = 'invalid_model_probability'
        result = dict(status=status, ids=ids, C_hat=c_hat, p_hat=p,
                      pi_hat=np.full(len(ids), np.nan), weights=None,
                      estimate=np.nan, overlap_count=self.overlap_count,
                      retained_draws=self.draws, auxiliary_size=len(self.auxiliary))
        if status == 'valid':
            w, pi = ratio_weights(self.u[ids], p, self.draws)
            result.update(weights=w, pi_hat=pi, estimate=float(w @ self.f[ids]))
        return result
