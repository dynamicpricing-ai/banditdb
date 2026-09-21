"""Experiment 1 — does the reference implementation reproduce the shipped engine?

Same contexts, same reward coins, same alpha, same arms. We compare the action
sequence and the learned parameters. Any divergence means the reference cannot be
used to stand in for the engine in later experiments.

Ties are the known source of divergence: when several arms have identical state
(as at t=0) the engine's argmax falls to HashMap iteration order. We therefore
report agreement after the first reward has broken symmetry, and we report the
final parameter distance, which is the quantity that matters for the sweeps.
"""
import json, os, sys
import numpy as np
import requests
from linucb_ref import LinUCB

DB, KEY = "http://localhost:8099", "sim-admin-key"
H = {"X-Api-Key": KEY, "Content-Type": "application/json"}
D, T, ALPHA = 6, 1500, 1.0
ARMS = [f"a{i}" for i in range(4)]

rng = np.random.default_rng(99)
TRUE = {a: np.abs(rng.normal(size=D)) for a in ARMS}
for a in TRUE:
    TRUE[a] = 0.45 * TRUE[a] / np.linalg.norm(TRUE[a])

ctxs = []
coins = []
for _ in range(T):
    x = np.abs(rng.normal(size=D))
    ctxs.append(x / np.linalg.norm(x))
    coins.append(rng.random())

s = requests.Session()


def api(method, path, **kw):
    r = s.request(method, f"{DB}{path}", headers=H, timeout=60, **kw)
    if r.status_code >= 400:
        raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:200]}")
    return r


def main():
    cid = "validate_ref"
    api("POST", "/campaign", json={"campaign_id": cid, "arms": ARMS, "feature_dim": D,
                                   "alpha": ALPHA, "algorithm": "linucb"})
    ref = LinUCB(D, ARMS, alpha=ALPHA)

    agree = 0
    compared = 0
    engine_regret = ref_regret = 0.0

    for t in range(T):
        x = ctxs[t]
        # Engine decides; the reference is asked what it would have done, then both
        # are updated with the SAME (arm, reward) pair so their states stay aligned.
        r = api("POST", "/predict", json={"campaign_id": cid, "context": x.tolist()}).json()
        arm = r["arm_id"]
        ref_arm, _ = ref.select(x)

        if t > 0:
            compared += 1
            agree += int(arm == ref_arm)

        p = float(np.clip(TRUE[arm] @ x, 0, 1))
        y = 1.0 if coins[t] < p else 0.0
        api("POST", "/reward", json={"interaction_id": r["interaction_id"], "reward": y})
        ref.update(arm, x, y)

        best = max(float(np.clip(TRUE[a] @ x, 0, 1)) for a in ARMS)
        engine_regret += best - p
        ref_regret += best - float(np.clip(TRUE[ref_arm] @ x, 0, 1))

    info = api("GET", f"/campaign/{cid}").json()
    dists = {}
    for a in ARMS:
        te = np.array(info["arms"][a]["theta"])
        tr = ref.arms[a].theta
        dists[a] = float(np.linalg.norm(te - tr))

    out = {
        "steps": T,
        "action_agreement": agree / compared,
        "max_theta_distance": max(dists.values()),
        "theta_distances": dists,
        "engine_regret": engine_regret,
        "ref_counterfactual_regret": ref_regret,
    }
    print(json.dumps(out, indent=2))
    json.dump(out, open(os.environ.get("OUT", "validate.json"), "w"))


if __name__ == "__main__":
    main()
