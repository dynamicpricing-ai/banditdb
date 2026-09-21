"""Experiment 2 — prior error and prior strength.

Theorem 1 says a warm start on theta* is a cold run on theta* - mu0, so regret
should depend on the prior error e = ||theta* - mu0||, not on ||theta*||.

Two falsifiable predictions:
  P1. Regret is monotone increasing in e.
  P2. A warm start whose error equals ||theta*|| performs the SAME as a cold
      start, because cold IS the mu0 = 0 case. Here ||theta*|| = 0.45 for every
      arm by construction, so the e = 0.45 curve must sit on the cold baseline.
  P3. Larger lambda amplifies the prior: better when e is small, worse when large,
      with the curves crossing near e = ||theta*||.
"""
import json, os, itertools
import numpy as np
from linucb_ref import LinUCB

D, K, T, ALPHA = 6, 5, 3000, 1.0
NORM = 0.45
EPS = [0.0, 0.1, 0.2, 0.45, 0.9, 1.8]
LAMS = [1.0, 5.0, 25.0]
SEEDS = list(range(20))


def world(seed):
    rng = np.random.default_rng(10_000 + seed)
    arms = [f"a{i}" for i in range(K)]
    true = {}
    for a in arms:
        v = np.abs(rng.normal(size=D))
        true[a] = NORM * v / np.linalg.norm(v)
    ctxs, coins = [], []
    for _ in range(T):
        x = np.abs(rng.normal(size=D))
        ctxs.append(x / np.linalg.norm(x))
        coins.append(rng.random())
    return arms, true, ctxs, coins


def run(seed, eps, lam, cold):
    arms, true, ctxs, coins = world(seed)
    rng = np.random.default_rng(20_000 + seed)
    if cold:
        model = LinUCB(D, arms, ALPHA)
    else:
        priors = {}
        for a in arms:
            u = rng.normal(size=D)
            u = u / np.linalg.norm(u)
            priors[a] = true[a] + eps * u          # ||mu0 - theta*|| = eps exactly
        model = LinUCB(D, arms, ALPHA, priors=priors, lam=lam)

    regret = 0.0
    curve = []
    for t in range(T):
        x = ctxs[t]
        arm, _ = model.select(x)
        p = float(np.clip(true[arm] @ x, 0, 1))
        y = 1.0 if coins[t] < p else 0.0
        model.update(arm, x, y)
        best = max(float(np.clip(true[a] @ x, 0, 1)) for a in arms)
        regret += best - p
        if (t + 1) % 250 == 0:
            curve.append(regret)
    theta_err = float(np.mean([np.linalg.norm(model.arms[a].theta - true[a]) for a in arms]))
    return regret, curve, theta_err


def main():
    rows = []
    for seed in SEEDS:
        r, c, e = run(seed, None, None, True)
        rows.append({"cond": "cold", "eps": NORM, "lam": 1.0, "seed": seed,
                     "regret": r, "curve": c, "theta_err": e})
        for eps, lam in itertools.product(EPS, LAMS):
            r, c, e = run(seed, eps, lam, False)
            rows.append({"cond": "warm", "eps": eps, "lam": lam, "seed": seed,
                         "regret": r, "curve": c, "theta_err": e})
        print(f"  seed {seed} done", flush=True)
    json.dump(rows, open(os.environ.get("OUT", "prior.json"), "w"))

    # Console summary
    import statistics as st
    cold = [x["regret"] for x in rows if x["cond"] == "cold"]
    print(f"\ncold: {st.mean(cold):.1f} +- {st.stdev(cold)/len(cold)**0.5:.1f}")
    for lam in LAMS:
        print(f"lambda={lam}")
        for eps in EPS:
            v = [x["regret"] for x in rows if x["cond"] == "warm" and x["eps"] == eps and x["lam"] == lam]
            print(f"   eps={eps:<5} regret {st.mean(v):7.1f} +- {st.stdev(v)/len(v)**0.5:4.1f}")


if __name__ == "__main__":
    main()
