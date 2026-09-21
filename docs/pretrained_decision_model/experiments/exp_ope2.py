"""Experiment 3 (corrected) — off-policy evaluation under per-request eligibility,
with a genuinely stochastic logging policy.

Thompson Sampling is used as the logger because its propensity is a real action
probability: P(arm = argmax of a posterior draw). BanditDB estimates it by Monte
Carlo over the ELIGIBLE arms. We compare that against the counterfactual system
that scores first and filters afterwards, whose propensities are normalised over
ALL arms.

Theory (Theorem 3): writing q_t for the probability mass the unrestricted policy
puts on ineligible arms, the naive weights are inflated by exactly 1/(1 - q_t),
so the estimator overstates the target value. The filter-aware estimator is
unbiased up to Monte Carlo error.
"""
import json, os
import numpy as np

D, K, T = 5, 6, 20_000
V = 0.25                     # posterior scale for TS draws
MC = 400                     # Monte Carlo draws per step for propensity estimates
SEEDS = list(range(8))


class TSArm:
    def __init__(self, d):
        self.a_inv = np.eye(d); self.b = np.zeros(d); self.theta = np.zeros(d)

    def draw(self, rng, n):
        L = np.linalg.cholesky(self.a_inv + 1e-12 * np.eye(len(self.b)))
        z = rng.normal(size=(n, len(self.b)))
        return self.theta + V * (z @ L.T)

    def update(self, x, r):
        ax = self.a_inv @ x
        self.a_inv -= np.outer(ax, ax) / (1.0 + float(x @ ax))
        self.a_inv = (self.a_inv + self.a_inv.T) * 0.5
        self.b += r * x
        self.theta = self.a_inv @ self.b


def run(seed):
    rng = np.random.default_rng(40_000 + seed)
    arms = list(range(K))
    true = []
    for _ in arms:
        v = np.abs(rng.normal(size=D))
        true.append(0.45 * v / np.linalg.norm(v))
    model = [TSArm(D) for _ in arms]

    num_a = den_a = num_n = den_n = 0.0
    truth_sum = 0.0
    leak_sum = 0.0

    for t in range(T):
        x = np.abs(rng.normal(size=D)); x = x / np.linalg.norm(x)
        elig = [a for a in arms if rng.random() < 0.6] or [int(rng.integers(K))]

        # One posterior draw per arm, shared by the action and the MC estimate.
        draws = np.stack([model[a].draw(rng, MC) @ x for a in arms])     # (K, MC)

        # --- filter-aware: restrict to eligible, then take argmax per draw
        sub = draws[elig, :]
        win_a = np.array(elig)[np.argmax(sub, axis=0)]
        p_aware = {a: float(np.mean(win_a == a)) for a in elig}
        arm = int(win_a[0])                                   # action = first draw

        # --- naive: argmax over ALL arms, normalised over all arms
        win_n = np.argmax(draws, axis=0)
        p_all = {a: float(np.mean(win_n == a)) for a in arms}
        leak = sum(p_all[a] for a in arms if a not in elig)    # mass on ineligible
        leak_sum += leak

        # Target policy: uniform over eligible arms (full support => overlap holds)
        pi = {a: 1.0 / len(elig) for a in elig}

        payoff = float(np.clip(true[arm] @ x, 0, 1))
        r = 1.0 if rng.random() < payoff else 0.0
        model[arm].update(x, r)

        truth_sum += sum(pi[a] * float(np.clip(true[a] @ x, 0, 1)) for a in elig)

        wa = pi[arm] / max(p_aware.get(arm, 0.0), 1e-9)
        wn = pi[arm] / max(p_all.get(arm, 0.0), 1e-9)
        num_a += wa * r; den_a += wa
        num_n += wn * r; den_n += wn

    return {"seed": seed, "truth": truth_sum / T,
            "snips_aware": num_a / den_a, "snips_naive": num_n / den_n,
            "mean_leak": leak_sum / T}


def main():
    rows = []
    for s in SEEDS:
        rows.append(run(s))
        print("  seed", s, "done", flush=True)
    for r in rows:
        r["err_aware"] = r["snips_aware"] - r["truth"]
        r["err_naive"] = r["snips_naive"] - r["truth"]
    json.dump(rows, open(os.environ.get("OUT", "ope2.json"), "w"))

    import statistics as st
    truth = st.mean([r["truth"] for r in rows])
    print(f"\ntrue target value      {truth:.5f}")
    for k in ("aware", "naive"):
        e = [r[f"err_{k}"] for r in rows]
        v = [r[f"snips_{k}"] for r in rows]
        print(f"SNIPS {k:6} estimate {st.mean(v):.5f}   bias {st.mean(e):+.5f} "
              f"(SE {st.stdev(e)/len(e)**0.5:.5f})   relative {st.mean(e)/truth*100:+.2f}%")
    print(f"mean ineligible mass q {st.mean([r['mean_leak'] for r in rows]):.4f}")


if __name__ == "__main__":
    main()
