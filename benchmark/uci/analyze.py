#!/usr/bin/env python3
"""
Learning-curve analysis for the UCI benchmark results.

Endpoint regret alone does not show that a policy learned anything: a policy
that instantly locks onto the majority class and never adapts also produces a
finite, respectable-looking number. What distinguishes learning is the *shape*
of the cumulative regret curve.

Metrics computed per (dataset, algorithm):

  beta        Slope of log R(t) vs log t. Regret grows as R(t) ~ t^beta.
              beta ~ 0.5  matches the sqrt(T) theory for LinUCB / TS
              beta < 1    sublinear -> the policy is still improving
              beta ~ 1    linear    -> constant error rate, no learning

  acc_first   Accuracy over the first 10% of rounds.
  acc_last    Accuracy over the last 10% of rounds.
  lift        acc_last - acc_first. Positive = measurably learned.

  t_major     First round after which the running accuracy stays above the
              majority-class rate. None = never beat the trivial baseline.

Usage:
    python benchmark/uci/analyze.py
    python benchmark/uci/analyze.py --results benchmark/uci/results
"""

import argparse
import glob
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

MIN_POINTS = 20  # curves coarser than this cannot support a slope fit

# Final total regret reported in Zhang, Zhou, Li & Gu, "Neural Thompson
# Sampling", ICLR 2021, Table 1 (mean over 20 runs). Their protocol differs
# from ours -- see the caveats printed at the bottom of this report.
PAPER = {
    "adult":    {"round": 10000, "dim": "2x15", "random": 5000,
                 "Linear UCB": 2097.5, "Linear TS": 2154.7,
                 "NeuralUCB": 2061.8, "NeuralTS": 2092.5},
    "magic":    {"round": 10000, "dim": "2x12", "random": 5000,
                 "Linear UCB": 2604.4, "Linear TS": 2700.5,
                 "NeuralUCB": 2033.0, "NeuralTS": 2037.4},
    "mushroom": {"round": 8124, "dim": "2x23", "random": 4062,
                 "Linear UCB": 562.7, "Linear TS": 643.3,
                 "NeuralUCB": 160.4, "NeuralTS": 115.0},
    "shuttle":  {"round": 10000, "dim": "7x9", "random": 8571,
                 "Linear UCB": 966.6, "Linear TS": 1020.9,
                 "NeuralUCB": 338.6, "NeuralTS": 232.0},
}

OURS_TO_PAPER = {
    "linucb": "Linear UCB",
    "thompson_sampling": "Linear TS",
    "neural_lin_ucb": "NeuralUCB",
    "neural_thompson_sampling": "NeuralTS",
}

ALGOS = list(OURS_TO_PAPER)


def loglog_slope(curve: list[dict]) -> float | None:
    """Least-squares slope of log(regret) against log(t), ignoring zero regret."""
    pts = [(math.log(p["t"]), math.log(p["regret"]))
           for p in curve if p["t"] > 0 and p["regret"] > 0]
    if len(pts) < MIN_POINTS:
        return None
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    num = sum((x - mx) * (y - my) for x, y in pts)
    den = sum((x - mx) ** 2 for x, _ in pts)
    return num / den if den > 0 else None


def window_accuracy(curve: list[dict], frac: float, tail: bool) -> float | None:
    """Accuracy over the first or last `frac` of the curve, from regret deltas."""
    if len(curve) < MIN_POINTS:
        return None
    k = max(2, int(len(curve) * frac))
    seg = curve[-k:] if tail else curve[:k]
    t0 = 0 if not tail else seg[0]["t"]
    r0 = 0.0 if not tail else seg[0]["regret"]
    dt = seg[-1]["t"] - t0
    dr = seg[-1]["regret"] - r0
    return 1.0 - dr / dt if dt > 0 else None


def rounds_to_beat_majority(curve: list[dict], majority_rate: float) -> int | None:
    """First t after which running accuracy never drops back below majority rate."""
    best = None
    for p in reversed(curve):
        acc = 1.0 - p["regret"] / p["t"] if p["t"] else 0.0
        if acc >= majority_rate:
            best = p["t"]
        else:
            break
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()

    runs = defaultdict(list)
    for f in sorted(glob.glob(f"{args.results}/uci_benchmark_*.json")):
        for r in json.load(open(f)):
            if len(r.get("curve", [])) >= MIN_POINTS:
                runs[(r["dataset"], r["algorithm"])].append(r)

    if not runs:
        raise SystemExit("no results with dense curves — rerun evaluate.py with --log-every 100")

    datasets = [d for d in PAPER if any(k[0] == d for k in runs)]

    print("=" * 100)
    print("LEARNING DIAGNOSTICS   (beta = slope of log-regret vs log-t; "
          "beta<1 sublinear = learning, beta~1 = flat error rate)")
    print("=" * 100)
    print(f"{'dataset':<10}{'algorithm':<26}{'beta':>7}{'acc_first':>11}"
          f"{'acc_last':>10}{'lift':>8}{'t_major':>9}{'runs':>6}")
    print("-" * 100)

    for ds in datasets:
        maj_rate = None
        for algo in ALGOS:
            rs = runs.get((ds, algo))
            if not rs:
                continue
            if maj_rate is None:
                maj_rate = 1.0 - rs[0]["majority_regret"] / rs[0]["rounds"]

            betas = [b for b in (loglog_slope(r["curve"]) for r in rs) if b is not None]
            first = [a for a in (window_accuracy(r["curve"], 0.1, False) for r in rs) if a]
            last = [a for a in (window_accuracy(r["curve"], 0.1, True) for r in rs) if a]
            tmaj = [rounds_to_beat_majority(r["curve"], maj_rate) for r in rs]
            tmaj = [t for t in tmaj if t is not None]

            beta = st.mean(betas) if betas else float("nan")
            af = st.mean(first) if first else float("nan")
            al = st.mean(last) if last else float("nan")
            tm = f"{int(st.mean(tmaj)):,}" if len(tmaj) == len(rs) else "never"
            print(f"{ds:<10}{algo:<26}{beta:>7.3f}{af:>11.4f}{al:>10.4f}"
                  f"{al - af:>+8.4f}{tm:>9}{len(rs):>6}")
        print()

    print("=" * 100)
    print("VS ZHANG ET AL. (ICLR 2021) TABLE 1 — final cumulative regret")
    print("=" * 100)
    print(f"{'dataset':<10}{'algorithm':<26}{'ours':>10}{'paper':>10}{'ratio':>9}"
          f"{'ourT':>8}{'paperT':>8}  comparable?")
    print("-" * 100)
    for ds in datasets:
        for algo in ALGOS:
            rs = runs.get((ds, algo))
            if not rs:
                continue
            ours = st.mean(r["cumulative_regret"] for r in rs)
            ref = PAPER[ds][OURS_TO_PAPER[algo]]
            our_t = rs[0]["rounds"]
            ref_t = PAPER[ds]["round"]
            same_t = our_t == ref_t
            linear = algo in ("linucb", "thompson_sampling")
            note = "yes" if (same_t and linear and ds == "shuttle") else \
                   "no - horizon" if not same_t else \
                   "no - architecture" if not linear else "no - features"
            print(f"{ds:<10}{algo:<26}{ours:>10,.0f}{ref:>10,.1f}"
                  f"{ours / ref:>9.2f}{our_t:>8,}{ref_t:>8,}  {note}")
        print()

    print("CAVEATS — why most cells above are NOT apples-to-apples:")
    print("  * alpha/nu: they grid-search nu in {1, 0.1, 0.01} and report the BEST;")
    print("    we ran a single untuned alpha=1.0. This favours the paper.")
    print("  * features: they L2-normalise to unit norm; we standardise per column.")
    print("  * encoding: their input dims are adult 15, magic 12, mushroom 23;")
    print("    ours are 96, 10, 76 -- different preprocessing, so different problems.")
    print("  * mushroom: they keep all 8,124 rows; we drop rows with '?' -> 5,644.")
    print("  * neural: they run ONE shared network over a k*d disjoint block encoding;")
    print("    BanditDB shares an MLP and keeps per-arm linear heads in embed space.")
    print("    These are different architectures; the neural rows are not comparable.")
    print("  * runs: they average 20; we ran 3 seeds per configuration.")
    print("  * shuttle linear is the one genuinely comparable cell: same d=9, same")
    print("    7 arms, same T=10,000, same random baseline of 8,571.")


if __name__ == "__main__":
    main()
