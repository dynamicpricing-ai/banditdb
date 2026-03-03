#!/usr/bin/env python3
"""
LinUCB Offline Hyperparameter Sweep — MovieLens 100K
=====================================================

Sweeps alpha × reward_type × feature_set using the exact replay method from
evaluate.py, but entirely offline (no BanditDB server needed).

Feature sets
------------
  basic       dim=3  [age_norm, is_male, occ_ordinal]
  bias        dim=4  [1.0, age_norm, is_male, occ_ordinal]   arm-level intercepts
  onehot      dim=23 [age_norm, is_male, occ_onehot×21]      non-ordinal occupation
  onehot+bias dim=24 [1.0, age_norm, is_male, occ_onehot×21]

Occupation IDs are losslessly recovered from the stored occ_norm = occ_id/21.

Usage:
  python benchmark/movielens/offline_sweep.py
"""

import json
import numpy as np
from pathlib import Path
from itertools import product

ARMS      = ["Drama", "Comedy", "Action", "Romance", "Thriller", "Adventure"]
N_OCC     = 21          # MovieLens 100K u.occupation has 21 occupations
DATA_DIR  = Path(__file__).parent.parent / "data"
TRAIN_WAL = DATA_DIR / "movielens_train.jsonl"
TEST_WAL  = DATA_DIR / "movielens_test.jsonl"


# ---------------------------------------------------------------------------
# Pure-Python LinUCB — mirrors BanditDB's Sherman-Morrison engine
# ---------------------------------------------------------------------------

class LinUCB:
    def __init__(self, arms, feature_dim, alpha):
        self.arms  = arms
        self.d     = feature_dim
        self.alpha = alpha
        self.A_inv = {a: np.eye(feature_dim) for a in arms}
        self.b     = {a: np.zeros(feature_dim) for a in arms}

    def predict(self, context):
        x = np.array(context, dtype=float)
        best, best_score = None, -np.inf
        for arm in self.arms:
            A_inv = self.A_inv[arm]
            theta = A_inv @ self.b[arm]
            ucb   = self.alpha * np.sqrt(max(0.0, float(x @ A_inv @ x)))
            score = float(theta @ x) + ucb
            if score > best_score:
                best_score, best = score, arm
        return best

    def update(self, arm, context, reward):
        x     = np.array(context, dtype=float)
        A_inv = self.A_inv[arm]
        Ax    = A_inv @ x
        denom = 1.0 + float(x @ Ax)
        self.A_inv[arm] = A_inv - np.outer(Ax, Ax) / denom
        self.b[arm] += reward * x

    def clone(self):
        m = LinUCB(self.arms, self.d, self.alpha)
        m.A_inv = {a: arr.copy() for a, arr in self.A_inv.items()}
        m.b     = {a: v.copy()   for a, v   in self.b.items()}
        return m


# ---------------------------------------------------------------------------
# WAL loader
# ---------------------------------------------------------------------------

def load_wal(path: Path):
    """Return list of (arm, raw_context_3d, continuous_reward)."""
    predicted, rewarded = {}, {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        event = json.loads(line)
        if "Predicted" in event:
            p = event["Predicted"]
            predicted[p["interaction_id"]] = (p["arm_id"], p["context"])
        elif "Rewarded" in event:
            r = event["Rewarded"]
            rewarded[r["interaction_id"]] = r["reward"]
    return [
        (arm, ctx, rewarded[iid])
        for iid, (arm, ctx) in predicted.items()
        if iid in rewarded
    ]


# ---------------------------------------------------------------------------
# Feature transforms
# ---------------------------------------------------------------------------

def expand_context(ctx, feature_set):
    """
    ctx is always the stored 3-dim vector [age_norm, is_male, occ_norm].
    Losslessly recover occupation_id = round(occ_norm * N_OCC).
    """
    age_norm, is_male, occ_norm = ctx

    if feature_set == "basic":
        return ctx                                         # dim=3

    if feature_set == "bias":
        return [1.0, age_norm, is_male, occ_norm]          # dim=4

    # One-hot occupation
    occ_id = min(max(int(round(occ_norm * N_OCC)), 0), N_OCC - 1)
    occ_oh = [0.0] * N_OCC
    occ_oh[occ_id] = 1.0

    if feature_set == "onehot":
        return [age_norm, is_male] + occ_oh                # dim=23

    if feature_set == "onehot+bias":
        return [1.0, age_norm, is_male] + occ_oh           # dim=24

    raise ValueError(f"Unknown feature_set: {feature_set}")


FEATURE_DIMS = {
    "basic":       3,
    "bias":        4,
    "onehot":      23,
    "onehot+bias": 24,
}


def to_reward(r, reward_type):
    if reward_type == "binary":
        return 1.0 if r >= 0.8 else 0.0   # rating >= 4 / 5
    return r                               # continuous: rating / 5.0


# ---------------------------------------------------------------------------
# Train + evaluate one configuration
# ---------------------------------------------------------------------------

def run_config(train, test, alpha, reward_type, feature_set):
    dim   = FEATURE_DIMS[feature_set]
    model = LinUCB(ARMS, dim, alpha)

    for arm, ctx, raw_r in train:
        model.update(arm, expand_context(ctx, feature_set), to_reward(raw_r, reward_type))

    # Replay evaluation with online learning on matches
    model_eval = model.clone()
    bandit_reward, bandit_count = 0.0, 0
    test_r = [to_reward(r, reward_type) for _, _, r in test]
    random_avg = sum(test_r) / len(test_r) if test_r else 0.0

    for (arm_logged, ctx, _), r in zip(test, test_r):
        xvec     = expand_context(ctx, feature_set)
        arm_pred = model_eval.predict(xvec)
        if arm_pred == arm_logged:
            bandit_count  += 1
            bandit_reward += r
            model_eval.update(arm_pred, xvec, r)

    bandit_avg = bandit_reward / bandit_count if bandit_count else 0.0
    lift       = (bandit_avg / random_avg - 1) * 100 if random_avg else 0.0
    match_rate = bandit_count / len(test) if test else 0.0
    return lift, bandit_avg, random_avg, match_rate, bandit_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not TRAIN_WAL.exists() or not TEST_WAL.exists():
        print("WAL files not found. Run convert.py first.")
        return

    print("Loading WAL data ...")
    train = load_wal(TRAIN_WAL)
    test  = load_wal(TEST_WAL)
    print(f"Train: {len(train):,}   Test: {len(test):,}\n")

    # Finer alpha grid around the known peak (1.0) + full range
    alphas        = [0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
    reward_types  = ["binary"]          # continuous already confirmed weak
    feature_sets  = ["basic", "bias", "onehot", "onehot+bias"]

    results = []
    col = "  {:<12}  {:<6}  {:>8}  {:>10}  {:>10}  {:>7}"
    print(col.format("features", "alpha", "lift", "bandit", "random", "match%"))
    print("  " + "-" * 60)

    for feature_set in feature_sets:
        for alpha in alphas:
            lift, b_avg, r_avg, match, matched = run_config(
                train, test, alpha, "binary", feature_set
            )
            results.append((lift, alpha, "binary", feature_set, b_avg, r_avg, match, matched))
            lift_s = f"{lift:+.1f}%"
            print(col.format(
                feature_set, f"{alpha:.3f}", lift_s,
                f"{b_avg:.4f}", f"{r_avg:.4f}", f"{match:.1%}"
            ))
        print()

    sep = "=" * 64
    print(sep)
    print("Top 10 by lift:")
    results.sort(reverse=True)
    for lift, alpha, rtype, fset, b_avg, r_avg, match, matched in results[:10]:
        print(
            f"  {fset:<12}  alpha={alpha:.3f}  lift={lift:+.1f}%"
            f"  bandit={b_avg:.4f}  random={r_avg:.4f}  match={match:.1%}"
        )
    print()

    best = results[0]
    lift, alpha, rtype, fset = best[0], best[1], best[2], best[3]
    print(f"Best config: features={fset}  alpha={alpha}  lift={lift:+.1f}%")
    if fset in ("onehot", "onehot+bias"):
        print()
        print("NOTE: one-hot features require regenerating the WAL.")
        print("Update convert.py to write the expanded context and set FEATURE_DIM accordingly.")


if __name__ == "__main__":
    main()
