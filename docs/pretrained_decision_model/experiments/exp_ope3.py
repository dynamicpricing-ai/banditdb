"""Experiment 3 (final) — isolating the effect of WHERE the filter is applied.

To measure the filtering effect alone, the logging policy is a softmax sampler:
it draws its action from softmax(scores) over the eligible set, so its action
probabilities are known exactly, with no Monte Carlo error. Plain IPS is then
unbiased by construction, and any residual bias is attributable to the
propensities themselves.

  aware : p(a) = softmax over ELIGIBLE arms      (filter applied before scoring)
  naive : p(a) = softmax over ALL arms           (filter applied after scoring)

Theorem 3 predicts the naive weights are inflated by exactly 1/(1 - q_t), where
q_t is the softmax mass on ineligible arms, giving a positive bias of
E[q/(1-q)] * V in expectation.
"""
import json, os
import numpy as np

D, K, T = 5, 6, 60_000
TEMP = 3.0
SEEDS = list(range(10))


def softmax(v, temp=TEMP):
    z = v * temp
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def run(seed):
    rng = np.random.default_rng(50_000 + seed)
    true = []
    for _ in range(K):
        v = np.abs(rng.normal(size=D))
        true.append(0.45 * v / np.linalg.norm(v))
    # A fixed, imperfect scoring model stands in for the learned one. Keeping it
    # fixed removes the learning dynamics, which are irrelevant to this claim and
    # would only add variance.
    est = [t + 0.15 * rng.normal(size=D) for t in true]

    ips_a = ips_n = 0.0
    snips_na = snips_da = snips_nn = snips_dn = 0.0
    truth = 0.0
    leak = 0.0

    for _ in range(T):
        x = np.abs(rng.normal(size=D)); x = x / np.linalg.norm(x)
        elig = [a for a in range(K) if rng.random() < 0.6] or [int(rng.integers(K))]

        s_all = np.array([float(e @ x) for e in est])
        p_all = softmax(s_all)                       # naive: over every arm
        p_el_full = np.zeros(K)
        p_el = softmax(s_all[elig])                  # aware: over eligible only
        for i, a in enumerate(elig):
            p_el_full[a] = p_el[i]

        q = float(1.0 - p_all[elig].sum())           # mass on ineligible arms
        leak += q

        # Action really is drawn from the aware distribution.
        arm = int(rng.choice(elig, p=p_el))

        # Target policy: uniform over eligible arms.
        pi = 1.0 / len(elig)

        payoff = float(np.clip(true[arm] @ x, 0, 1))
        r = 1.0 if rng.random() < payoff else 0.0

        truth += sum(pi * float(np.clip(true[a] @ x, 0, 1)) for a in elig)

        wa = pi / p_el_full[arm]
        wn = pi / p_all[arm]
        ips_a += wa * r
        ips_n += wn * r
        snips_na += wa * r; snips_da += wa
        snips_nn += wn * r; snips_dn += wn

    return {"seed": seed, "truth": truth / T,
            "ips_aware": ips_a / T, "ips_naive": ips_n / T,
            "snips_aware": snips_na / snips_da, "snips_naive": snips_nn / snips_dn,
            "mean_q": leak / T}


def main():
    rows = [run(s) for s in SEEDS]
    json.dump(rows, open(os.environ.get("OUT", "ope3.json"), "w"))
    import statistics as st
    truth = st.mean([r["truth"] for r in rows])
    q = st.mean([r["mean_q"] for r in rows])
    print(f"true target value {truth:.5f}    mean ineligible mass q = {q:.4f}")
    print(f"predicted naive inflation 1/(1-q) = {1/(1-q):.4f}\n")
    for est in ("ips_aware", "ips_naive", "snips_aware", "snips_naive"):
        v = [r[est] for r in rows]
        bias = st.mean(v) - truth
        print(f"{est:12} {st.mean(v):.5f}   bias {bias:+.5f} "
              f"(SE {st.stdev(v)/len(v)**0.5:.5f})   relative {bias/truth*100:+6.2f}%")


if __name__ == "__main__":
    main()
