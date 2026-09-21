"""Experiment 3 — off-policy evaluation under per-request eligibility.

A logging policy serves only eligible arms. We estimate the value of a fixed
target policy by inverse propensity scoring, twice:

  (a) filter-aware  : propensity = softmax over the ELIGIBLE set (what BanditDB logs)
  (b) naive         : propensity = softmax over ALL arms (what it would log if the
                      filter were applied after scoring)

Ground truth is computable because we know the payoff function, so the bias of
each estimator is measurable rather than argued.
"""
import json, os
import numpy as np
from linucb_ref import LinUCB

D, K, T = 6, 6, 40_000
ALPHA = 1.0
SEEDS = list(range(10))


def softmax(d):
    m = max(d.values())
    ex = {k: np.exp(v - m) for k, v in d.items()}
    z = sum(ex.values())
    return {k: v / z for k, v in ex.items()}


def run(seed):
    rng = np.random.default_rng(30_000 + seed)
    arms = [f"a{i}" for i in range(K)]
    true = {a: 0.45 * (lambda v: v / np.linalg.norm(v))(np.abs(rng.normal(size=D))) for a in arms}

    model = LinUCB(D, arms, ALPHA)
    # Target policy: a fixed, known stochastic policy over eligible arms
    # (softmax over a fixed scoring vector). Its true value is computed exactly.
    w = np.abs(rng.normal(size=D))

    num_a = den_a = num_n = den_n = 0.0     # SNIPS accumulators
    truth_sum = 0.0

    for t in range(T):
        x = np.abs(rng.normal(size=D)); x = x / np.linalg.norm(x)
        # Random eligibility: each arm independently available with prob 0.6,
        # at least one guaranteed.
        elig = [a for a in arms if rng.random() < 0.6] or [arms[rng.integers(K)]]

        # --- target policy pi(a|x): softmax over eligible arms on fixed weights
        tgt_scores = {a: float(w @ x) * (1 + 0.3 * int(a.endswith("0"))) + 0.5 * float(true[a] @ x)
                      for a in elig}
        pi = softmax(tgt_scores)

        # --- logging policy: LinUCB restricted to eligible arms
        arm, scores_elig = model.select(x, eligible=elig)
        p_aware = model.propensities(scores_elig)                    # (a) over eligible
        scores_all = {a: model.arms[a].score(x, ALPHA) for a in arms}
        p_naive = model.propensities(scores_all)                     # (b) over all arms

        payoff = float(np.clip(true[arm] @ x, 0, 1))
        r = 1.0 if rng.random() < payoff else 0.0
        model.update(arm, x, r)

        # Ground truth value of the target policy on this context
        truth_sum += sum(pi[a] * float(np.clip(true[a] @ x, 0, 1)) for a in elig)

        wgt_a = pi.get(arm, 0.0) / max(p_aware[arm], 1e-12)
        wgt_n = pi.get(arm, 0.0) / max(p_naive[arm], 1e-12)
        num_a += wgt_a * r; den_a += wgt_a
        num_n += wgt_n * r; den_n += wgt_n

    return {
        "seed": seed,
        "truth": truth_sum / T,
        "snips_aware": num_a / den_a,
        "snips_naive": num_n / den_n,
    }


def main():
    rows = [run(s) for s in SEEDS]
    for r in rows:
        r["err_aware"] = r["snips_aware"] - r["truth"]
        r["err_naive"] = r["snips_naive"] - r["truth"]
    json.dump(rows, open(os.environ.get("OUT", "ope.json"), "w"))

    import statistics as st
    for k in ("err_aware", "err_naive"):
        v = [r[k] for r in rows]
        print(f"{k:11} mean bias {st.mean(v):+.5f}  (SE {st.stdev(v)/len(v)**0.5:.5f})  "
              f"|rel| {abs(st.mean(v))/st.mean([r['truth'] for r in rows])*100:5.2f}%")
    print("truth mean", st.mean([r["truth"] for r in rows]))


if __name__ == "__main__":
    main()
