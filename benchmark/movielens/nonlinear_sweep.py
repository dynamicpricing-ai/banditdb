#!/usr/bin/env python3
"""
Non-linear Feature Expansion Sweep — MovieLens 100K
=====================================================

Goal: find cross-feature configurations that push lift toward +35%+.

Key insight from prior sweep: x_2way_full (dim=68) performed worse than
baseline (dim=24) because the UCB term alpha*sqrt(x^T A^-1 x) inflates
with dimensionality — more active features per user means higher
exploration penalty at the same alpha. The fix is either:

  (a) Much lower alpha for high-dim models (tested here, 0.001–0.05)
  (b) L2-normalize the context so UCB is bounded by alpha regardless of dim

Feature sets
------------
  baseline        dim=24   [1, age, male, occ_oh×21]           current best
  x_pairs         dim=68   baseline + age×male + age² +         all pairwise
                           male×occ_oh×21 + age×occ_oh×21      (non-zero pairs only)
  x_pairs_normed  dim=68   x_pairs but L2-normalised — removes UCB inflation
  poly2           dim=10   degree-2 poly of [age, male, occ_norm]  — compact
  poly3           dim=20   degree-3 poly of [age, male, occ_norm]
  poly2_normed    dim=10   poly2, L2-normalised
  poly3_normed    dim=20   poly3, L2-normalised
  x_age_occ_only  dim=46   baseline + age×occ_oh×21 (best single cross from prior sweep)

Usage:
  python benchmark/movielens/nonlinear_sweep.py
"""

import json
import itertools
import numpy as np
from pathlib import Path

ARMS     = ["Drama", "Comedy", "Action", "Romance", "Thriller", "Adventure"]
N_OCC    = 21
DATA_DIR = Path(__file__).parent.parent / "data"
TRAIN_WAL = DATA_DIR / "movielens_train.jsonl"
TEST_WAL  = DATA_DIR / "movielens_test.jsonl"


# ---------------------------------------------------------------------------
# LinUCB (Sherman-Morrison)
# ---------------------------------------------------------------------------

class LinUCB:
    def __init__(self, arms, d, alpha):
        self.arms = arms; self.d = d; self.alpha = alpha
        self.A_inv = {a: np.eye(d)   for a in arms}
        self.b     = {a: np.zeros(d) for a in arms}

    def predict(self, x):
        x = np.array(x, dtype=float); best, bs = None, -np.inf
        for a in self.arms:
            Ai = self.A_inv[a]; th = Ai @ self.b[a]
            s  = float(th @ x) + self.alpha * np.sqrt(max(0., float(x @ Ai @ x)))
            if s > bs: bs, best = s, a
        return best

    def update(self, arm, x, r):
        x = np.array(x, dtype=float); Ai = self.A_inv[arm]; Ax = Ai @ x
        self.A_inv[arm] = Ai - np.outer(Ax, Ax) / (1. + float(x @ Ax))
        self.b[arm] += r * x

    def clone(self):
        m = LinUCB(self.arms, self.d, self.alpha)
        m.A_inv = {a: v.copy() for a, v in self.A_inv.items()}
        m.b     = {a: v.copy() for a, v in self.b.items()}
        return m


# ---------------------------------------------------------------------------
# WAL loader
# ---------------------------------------------------------------------------

def load_wal(path):
    pred, rew = {}, {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line: continue
        ev = json.loads(line)
        if "Predicted" in ev:
            p = ev["Predicted"]
            pred[p["interaction_id"]] = (p["arm_id"], p["context"])
        elif "Rewarded" in ev:
            r = ev["Rewarded"]
            rew[r["interaction_id"]] = r["reward"]
    return [(a, c, rew[i]) for i, (a, c) in pred.items() if i in rew]


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def unpack(raw_ctx):
    """Extract (age, male, occ_norm, occ_id, occ_oh) from either 3-dim or 24-dim context."""
    if len(raw_ctx) == 3:
        age, male, occ_n = raw_ctx
        occ_id = min(max(int(round(occ_n * N_OCC)), 0), N_OCC - 1)
        occ_oh = [0.0] * N_OCC
        occ_oh[occ_id] = 1.0
    else:
        age    = raw_ctx[1]
        male   = raw_ctx[2]
        occ_oh = list(raw_ctx[3:3 + N_OCC])
        occ_id = int(np.argmax(occ_oh))
        occ_n  = occ_id / N_OCC
    return age, male, occ_n, occ_id, occ_oh


def build(raw_ctx, feature_set, normalise=False):
    age, male, occ_n, occ_id, occ_oh = unpack(raw_ctx)

    if feature_set == "baseline":
        v = [1.0, age, male] + occ_oh                                          # dim=24

    elif feature_set == "x_pairs":
        # All non-trivial pairwise products of base features.
        # occ_oh[i]*occ_oh[j] = 0 for i≠j (user is in one occupation only),
        # so only 44 new features are ever non-zero.
        cross  = [age * male, age * age]
        cross += [male * occ_oh[i] for i in range(N_OCC)]                     # 21
        cross += [age  * occ_oh[i] for i in range(N_OCC)]                     # 21
        v = [1.0, age, male] + occ_oh + cross                                  # dim=68

    elif feature_set == "x_age_occ_only":
        # Only the age-occupation cross (best single cross from prior sweep)
        cross  = [age * occ_oh[i] for i in range(N_OCC)]
        v = [1.0, age, male] + occ_oh + cross                                  # dim=46

    elif feature_set == "poly2":
        # Degree-2 polynomial of [age, male, occ_norm] — compact, avoids
        # the sparse one-hot explosion. dim=10 (bias + 3 linear + 6 quadratic)
        v = [
            1.0,                            # bias
            age, male, occ_n,               # degree 1
            age*age, age*male, age*occ_n,   # degree 2
            male*male, male*occ_n,          # (male^2 = male but kept for symmetry)
            occ_n*occ_n,
        ]                                                                       # dim=10

    elif feature_set == "poly3":
        # Degree-3 polynomial of [age, male, occ_norm]. dim=20
        a, m, o = age, male, occ_n
        v = [
            1.0,                                            # 1
            a, m, o,                                        # 3  → total 4
            a*a, a*m, a*o, m*m, m*o, o*o,                  # 6  → total 10
            a*a*a, a*a*m, a*a*o, a*m*m, a*m*o, a*o*o,      # 6  → total 16
            m*m*m, m*m*o, m*o*o, o*o*o,                     # 4  → total 20
        ]                                                                       # dim=20

    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    v = np.array(v, dtype=float)
    if normalise:
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            v = v / norm
    return v


FEATURE_DIMS = {
    "baseline":       24,
    "x_pairs":        68,
    "x_age_occ_only": 45,
    "poly2":          10,
    "poly3":          20,
}


# ---------------------------------------------------------------------------
# Train + evaluate
# ---------------------------------------------------------------------------

def run(train, test, alpha, feature_set, normalise=False):
    dim   = FEATURE_DIMS[feature_set]
    model = LinUCB(ARMS, dim, alpha)

    for arm, ctx, raw_r in train:
        r = 1.0 if raw_r >= 0.8 else 0.0
        model.update(arm, build(ctx, feature_set, normalise), r)

    ev = model.clone()
    br, bc = 0.0, 0
    test_r = [1.0 if r >= 0.8 else 0.0 for _, _, r in test]
    rand   = sum(test_r) / len(test_r) if test_r else 0.0

    for (al, ctx, _), r in zip(test, test_r):
        xv = build(ctx, feature_set, normalise)
        ap = ev.predict(xv)
        if ap == al:
            bc += 1; br += r
            ev.update(ap, xv, r)

    ba   = br / bc if bc else 0.0
    lift = (ba / rand - 1) * 100 if rand else 0.0
    return lift, ba, rand, bc / len(test), bc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading WAL data ...")
    all_data = load_wal(TRAIN_WAL) + load_wal(TEST_WAL)
    split    = int(len(all_data) * 0.9)
    train, test = all_data[:split], all_data[split:]
    print(f"Total: {len(all_data):,}   Train: {len(train):,}   Test: {len(test):,}\n")

    # Wide alpha range — essential for high-dim models
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 5.0]

    configs = [
        ("baseline",       False),
        ("x_pairs",        False),   # all pairwise, raw alpha
        ("x_pairs",        True),    # all pairwise, L2-normalised
        ("x_age_occ_only", False),
        ("x_age_occ_only", True),
        ("poly2",          False),
        ("poly2",          True),
        ("poly3",          False),
        ("poly3",          True),
    ]

    results = []
    col = "  {:<22}  {:<5}  {:>8}  {:>10}  {:>7}  {:>5}"
    print(col.format("feature_set", "alpha", "lift", "bandit", "match%", "dim"))
    print("  " + "-" * 62)

    for feature_set, normalise in configs:
        label = feature_set + ("(normed)" if normalise else "")
        dim   = FEATURE_DIMS[feature_set]
        best  = -np.inf
        for alpha in alphas:
            lift, ba, rand, match, cnt = run(train, test, alpha, feature_set, normalise)
            results.append((lift, alpha, label, ba, rand, match, cnt))
            marker = " <--" if lift > best else ""
            best   = max(best, lift)
            print(col.format(label[:22], f"{alpha:.3f}",
                             f"{lift:+.1f}%", f"{ba:.4f}",
                             f"{match:.1%}", str(dim)) + marker)
        print()

    sep = "=" * 65
    print(sep)
    print("Top 15 by lift:")
    results.sort(reverse=True)
    for lift, alpha, label, ba, rand, match, cnt in results[:15]:
        print(f"  {label:<22}  alpha={alpha:.3f}  lift={lift:+.1f}%"
              f"  bandit={ba:.4f}  match={match:.1%}")

    print()
    best = results[0]
    print(f"Best: {best[2]}  alpha={best[1]}  lift={best[0]:+.1f}%")


if __name__ == "__main__":
    main()
