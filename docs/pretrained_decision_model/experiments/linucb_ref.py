"""Reference LinUCB implementing exactly the update rules in BanditDB's src/math.rs.

    score(x)  = theta.x + alpha * sqrt(x' A^-1 x)
    update    : Sherman-Morrison rank-one, then symmetry enforcement
                A^-1 <- (A^-1 + A^-1') / 2
    warm start: A^-1 = I/lambda, b = lambda*mu0  =>  theta = mu0

Validated against the live engine in exp_validate.py before it is used for any
sweep, so results obtained here describe the shipped algorithm, not an idealised
cousin of it.
"""
import numpy as np


class Arm:
    def __init__(self, d, mu0=None, lam=1.0):
        if mu0 is None:
            self.a_inv = np.eye(d)
            self.b = np.zeros(d)
            self.theta = np.zeros(d)
        else:
            self.a_inv = np.eye(d) / lam
            self.b = lam * np.asarray(mu0, float)
            self.theta = np.asarray(mu0, float).copy()
        self.n = 0

    def score(self, x, alpha):
        var = max(float(x @ (self.a_inv @ x)), 0.0)
        return float(self.theta @ x) + alpha * math_sqrt(var)

    def update(self, x, r):
        ax = self.a_inv @ x
        denom = 1.0 + float(x @ ax)
        self.a_inv = self.a_inv - np.outer(ax, ax) / denom
        self.a_inv = (self.a_inv + self.a_inv.T) * 0.5      # symmetry enforcement
        self.b = self.b + r * x
        self.theta = self.a_inv @ self.b
        self.n += 1


def math_sqrt(v):
    return v ** 0.5


class LinUCB:
    def __init__(self, d, arm_ids, alpha=1.0, priors=None, lam=1.0):
        priors = priors or {}
        self.d = d
        self.alpha = alpha
        self.arms = {a: Arm(d, priors.get(a), lam) for a in arm_ids}

    def select(self, x, eligible=None):
        cands = eligible if eligible is not None else list(self.arms)
        scores = {a: self.arms[a].score(x, self.alpha) for a in cands}
        return max(scores, key=scores.get), scores

    def propensities(self, scores):
        """Softmax over the scores actually used — mirrors softmax_propensities()."""
        m = max(scores.values())
        ex = {a: np.exp(s - m) for a, s in scores.items()}
        z = sum(ex.values())
        return {a: v / z for a, v in ex.items()}

    def update(self, arm, x, r):
        self.arms[arm].update(x, r)
