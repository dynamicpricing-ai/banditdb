"""Experiment 4 — allocation under capacity constraints.

Payoffs are known exactly here, which deliberately removes learning from the
picture: the question is not "can the bandit estimate?" but "given estimates,
does argmax allocate well when arms are scarce?"

Three policies:
  greedy   — serve the highest-value arm that still has capacity (what a bandit
             plus a hard inventory check does today)
  dual     — serve argmax of (r - lambda_k), with lambda from online dual mirror
             descent on capacity consumption
  offline  — the hindsight LP optimum, solved with HiGHS; an upper bound no
             online policy can exceed
"""
import json, os
import numpy as np
from scipy.optimize import linprog

T, K = 2000, 5
SEEDS = list(range(10))
SCARCE_SHARE = 0.15      # the popular arm can serve only 15% of requests
ABUNDANT     = 0.40      # every other arm can serve 40% each (total > T, always feasible)


def instance(seed):
    rng = np.random.default_rng(60_000 + seed)
    # Request-by-arm payoff matrix. Arm 0 is broadly strong (the "popular" item),
    # so scarcity binds on it; the others are specialists.
    r = np.zeros((T, K))
    for t in range(T):
        base = rng.uniform(0.1, 0.4)
        r[t, 0] = base + rng.uniform(0.25, 0.45)
        for k in range(1, K):
            r[t, k] = base + rng.uniform(0.0, 0.5)
    cap = np.full(K, ABUNDANT * T)
    cap[0] = SCARCE_SHARE * T      # only arm 0 is scarce
    return r, cap


def greedy(r, cap):
    left = cap.copy(); total = 0.0; served = 0
    for t in range(T):
        avail = [k for k in range(K) if left[k] >= 1]
        k = max(avail, key=lambda k: r[t, k])
        left[k] -= 1; total += r[t, k]; served += 1
    return total, served


def dual(r, cap, eta=0.02):
    left = cap.copy(); lam = np.zeros(K); total = 0.0; served = 0
    target = cap / T                      # per-step consumption budget
    for t in range(T):
        avail = [k for k in range(K) if left[k] >= 1]
        k = max(avail, key=lambda k: r[t, k] - lam[k])
        consumed = np.zeros(K); consumed[k] = 1.0
        lam = np.maximum(0.0, lam + eta * (consumed - target))
        left[k] -= 1; total += r[t, k]; served += 1
    return total, served


def offline(r, cap):
    # max sum r*x  s.t.  sum_k x_tk <= 1 (each request served at most once),
    #                    sum_t x_tk <= cap_k
    from scipy.sparse import coo_matrix
    c = -r.reshape(-1)
    er, ec, ed = [], [], []
    for t in range(T):                       # each request served exactly once
        for k in range(K):
            er.append(t); ec.append(t * K + k); ed.append(1.0)
    A_eq = coo_matrix((ed, (er, ec)), shape=(T, T * K))
    ur, uc, ud = [], [], []
    for k in range(K):                       # per-arm capacity
        for t in range(T):
            ur.append(k); uc.append(t * K + k); ud.append(1.0)
    A_ub = coo_matrix((ud, (ur, uc)), shape=(K, T * K))
    res = linprog(c, A_ub=A_ub, b_ub=cap, A_eq=A_eq, b_eq=np.ones(T),
                  bounds=(0, 1), method="highs")
    return -res.fun


def main():
    rows = []
    for s in SEEDS:
        r, cap = instance(s)
        g, gs = greedy(r, cap)
        d, ds = dual(r, cap)
        o = offline(r, cap)
        rows.append({"seed": s, "greedy": g, "dual": d, "offline": o,
                     "greedy_served": gs, "dual_served": ds,
                     "greedy_gap": (o - g) / o, "dual_gap": (o - d) / o})
        print(f"  seed {s}: greedy {g:.1f}  dual {d:.1f}  LP {o:.1f}  "
              f"gaps {rows[-1]['greedy_gap']*100:.2f}% / {rows[-1]['dual_gap']*100:.2f}%", flush=True)
    json.dump(rows, open(os.environ.get("OUT", "capacity.json"), "w"))
    import statistics as st
    for k in ("greedy_gap", "dual_gap"):
        v = [x[k] * 100 for x in rows]
        print(f"{k:11} mean {st.mean(v):.2f}%  (SE {st.stdev(v)/len(v)**0.5:.2f})")


if __name__ == "__main__":
    main()
