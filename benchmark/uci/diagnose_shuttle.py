#!/usr/bin/env python3
"""
Investigate the shuttle LinUCB gap: BanditDB ~2,026 vs Zhang et al. 966.6 at T=10,000.

Two independent questions, answered separately:

  A) Is it preprocessing? Measure the full-information linear ceiling under
     several feature scalings. LinUCB fits one-vs-rest ridge on 0/1 rewards, so
     a RidgeClassifier with full labels is the ceiling no bandit can exceed.
     If no scaling reaches the paper's 90.3% accuracy, their preprocessing
     differs from all of ours.

  B) Is it the implementation? Run a reference LinUCB written directly against
     the update rule in src/math.rs, on the identical shuffle order the HTTP
     harness uses. If the reference matches BanditDB, the engine is faithful and
     the gap lives in the protocol. If the reference is much better, the engine
     has a bug.

Usage:
    python benchmark/uci/diagnose_shuttle.py
"""

import json
import random
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score

DATA = Path(__file__).parent.parent / "data" / "uci_bandit"
PAPER_REGRET = 966.6
PAPER_ACC = 1.0 - PAPER_REGRET / 10000
ROUNDS = 10000
SEEDS = [0, 1, 2]


def load_raw():
    """Reconstruct pre-standardisation features straight from the UCI source."""
    import pandas as pd
    src = Path(__file__).parent.parent / "data" / "uci" / "statlog+shuttle"
    frames = [pd.read_csv(p, header=None, sep=r"\s+")
              for p in (src / "shuttle.trn", src / "shuttle.tst") if p.exists()]
    df = pd.concat(frames, ignore_index=True).dropna()
    y_raw = df[9].to_numpy()
    uniq = np.sort(np.unique(y_raw))
    remap = {v: i for i, v in enumerate(uniq)}
    y = np.array([remap[v] for v in y_raw])
    x = df.drop(columns=[9]).to_numpy(dtype=np.float64)
    return x, y


def scalings(x_raw):
    z = (x_raw - x_raw.mean(0)) / np.where(x_raw.std(0) < 1e-12, 1.0, x_raw.std(0))

    def l2(a):
        n = np.linalg.norm(a, axis=1, keepdims=True)
        return a / np.where(n < 1e-12, 1.0, n)

    rng = x_raw.max(0) - x_raw.min(0)
    minmax = (x_raw - x_raw.min(0)) / np.where(rng < 1e-12, 1.0, rng)
    return {
        "raw":            x_raw,
        "zscore (ours)":  z,
        "l2 only":        l2(x_raw),
        "zscore + l2":    l2(z),
        "minmax":         minmax,
        "minmax + l2":    l2(minmax),
    }


def linucb_reference(x, y, n_arms, alpha, seed, rounds=ROUNDS):
    """
    Disjoint LinUCB mirroring src/math.rs:
        A_inv init I  (ridge lambda = 1)
        score = theta.x + alpha*sqrt(x' A_inv x)
        Sherman-Morrison rank-1 update, symmetry enforced
        theta = A_inv b
    Shuffle order replicates evaluate.py exactly.
    """
    d = x.shape[1]
    a_inv = [np.eye(d) for _ in range(n_arms)]
    b = [np.zeros(d) for _ in range(n_arms)]
    theta = [np.zeros(d) for _ in range(n_arms)]

    rng = random.Random(seed)
    order = list(range(len(y)))
    rng.shuffle(order)
    order = order[:rounds]

    regret = 0
    for idx in order:
        ctx = x[idx]
        scores = [theta[a] @ ctx + alpha * np.sqrt(max(ctx @ (a_inv[a] @ ctx), 0.0))
                  for a in range(n_arms)]
        arm = int(np.argmax(scores))
        reward = 1.0 if arm == y[idx] else 0.0
        regret += 1 - reward

        ai_x = a_inv[arm] @ ctx
        a_inv[arm] = a_inv[arm] - np.outer(ai_x, ai_x) / (1.0 + ctx @ ai_x)
        a_inv[arm] = (a_inv[arm] + a_inv[arm].T) * 0.5
        b[arm] = b[arm] + ctx * reward
        theta[arm] = a_inv[arm] @ b[arm]
    return regret


def main():
    x_raw, y = load_raw()
    n_arms = len(np.unique(y))
    print(f"shuttle: {x_raw.shape[0]:,} rows, d={x_raw.shape[1]}, {n_arms} classes")
    print(f"paper LinUCB: regret {PAPER_REGRET} at T=10,000  ->  accuracy {PAPER_ACC:.4f}\n")

    sub = np.random.RandomState(0).permutation(len(y))[:ROUNDS]

    print("=" * 78)
    print("A) FULL-INFORMATION LINEAR CEILING (one-vs-rest ridge = LinUCB's own model)")
    print("=" * 78)
    print(f"{'scaling':<16}{'||x|| mean':>12}{'||x|| max':>12}{'ridge acc':>12}"
          f"{'equiv regret':>14}{'>= paper?':>11}")
    print("-" * 78)
    variants = scalings(x_raw)
    for name, xv in variants.items():
        norms = np.linalg.norm(xv, axis=1)
        acc = cross_val_score(RidgeClassifier(), xv[sub], y[sub], cv=3).mean()
        flag = "YES" if acc >= PAPER_ACC else "no"
        print(f"{name:<16}{norms.mean():>12.3f}{norms.max():>12.3f}{acc:>12.4f}"
              f"{ROUNDS * (1 - acc):>14,.0f}{flag:>11}")

    print()
    print("=" * 78)
    print("B) REFERENCE LinUCB (numpy, mirrors src/math.rs) vs BanditDB HTTP engine")
    print("=" * 78)
    print(f"{'scaling':<16}{'alpha':>7}{'ref regret':>13}{'ref acc':>10}   note")
    print("-" * 78)
    for name in ("zscore (ours)", "raw", "minmax", "zscore + l2"):
        xv = variants[name]
        for alpha in (1.0, 0.1, 0.01):
            rs = [linucb_reference(xv, y, n_arms, alpha, s) for s in SEEDS]
            mean = float(np.mean(rs))
            note = ""
            if name == "zscore (ours)" and alpha == 1.0:
                note = "<- BanditDB measured 2,026 here"
            print(f"{name:<16}{alpha:>7}{mean:>13,.0f}{1 - mean / ROUNDS:>10.4f}   {note}")
    print()


if __name__ == "__main__":
    main()
