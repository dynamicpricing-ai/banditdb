#!/usr/bin/env python3
"""
Head-to-head: BanditDB's neural architecture vs the paper's block encoding.

The question this settles: is BanditDB's shared-MLP-with-per-arm-linear-heads
design representationally weaker than the k*d disjoint block encoding used by
NeuralUCB (Zhou et al. 2019) and NeuralTS (Zhang et al. ICLR 2021)?

Both arms run in this one harness so the only variable is the architecture.
Everything else is held identical: data, shuffle order, network width, gradient
budget, exploration grid, seeds.

  ARM A -- "banditdb"  (NeuralLinUCB, Xu et al.)
      shared MLP  phi_W : R^d -> R^m
      per-arm ridge head theta_a in R^m, A_a in R^{m x m}
      score_a = phi(x).theta_a + nu * sqrt(phi(x)' A_a^-1 phi(x))
      periodic retrain of W on sum (phi(x_t).theta_{a_t} - r_t)^2,
      then arm statistics re-accumulated in the new embedding space.
      This mirrors src/neural.rs Algorithm 2.

  ARM B -- "block"  (NeuralUCB / NeuralTS)
      one network  f_W : R^{k*d} -> R
      block encoding x_a = [0; ...; x; ...; 0]
      score_a = f(x_a) + nu * sqrt(sum_i g_a[i]^2 / U[i]),  g_a = grad_W f(x_a)
      U is the diagonal approximation of the gradient outer-product matrix --
      the same approximation the paper uses ("we use the inverse of the diagonal
      elements of U as an approximation of U^-1").

Fairness notes:
  * Both architectures get the SAME total number of gradient steps. Arm B trains
    `steps_per_round` steps every round; arm A retrains every `retrain_every`
    rounds with a proportionally larger step count so the products match.
  * Both grid-search nu over the same values and the BEST is reported, matching
    the paper's protocol.
  * T is reduced from 10,000 because arm B needs one backward pass per arm per
    round to form its gradient features. This is stated in the output.

Usage:
    python benchmark/uci/block_encoding_ab.py --dataset shuttle
    python benchmark/uci/block_encoding_ab.py --dataset shuttle --rounds 5000 --seeds 3
"""

import argparse
import json
import random
import statistics as st
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

DATA = Path(__file__).parent.parent / "data" / "uci_bandit"
RESULTS = Path(__file__).parent / "results"
torch.set_num_threads(4)


def load(name):
    meta = json.loads((DATA / f"{name}.meta.json").read_text())
    rows = [json.loads(l) for l in (DATA / f"{name}.jsonl").read_text().splitlines()]
    x = np.array([r["context"] for r in rows], dtype=np.float32)
    y = np.array([r["label"] for r in rows], dtype=np.int64)
    return meta, x, y


def stream(x, y, rounds, seed):
    rng = random.Random(seed)
    order = list(range(len(y)))
    rng.shuffle(order)
    if rounds > len(order):
        order = [order[i % len(order)] for i in range(rounds)]
    return order[:rounds]


def mlp(d_in, hidden, d_out, seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, d_out))


# --- ARM A: BanditDB architecture ---------------------------------------

def run_banditdb(x, y, k, nu, seed, rounds, hidden, embed_dim,
                 retrain_every, steps_per_retrain, lam=1.0):
    d = x.shape[1]
    net = mlp(d, hidden, embed_dim, seed)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    a_inv = [np.eye(embed_dim, dtype=np.float64) for _ in range(k)]
    b = [np.zeros(embed_dim) for _ in range(k)]
    theta = [np.zeros(embed_dim) for _ in range(k)]

    buf_x, buf_a, buf_r = [], [], []
    regret = 0

    for t, idx in enumerate(stream(x, y, rounds, seed), 1):
        ctx = torch.from_numpy(x[idx]).unsqueeze(0)
        with torch.no_grad():
            phi = net(ctx).squeeze(0).double().numpy()

        scores = [theta[a] @ phi + nu * np.sqrt(max(phi @ (a_inv[a] @ phi), 0.0))
                  for a in range(k)]
        arm = int(np.argmax(scores))
        r = 1.0 if arm == y[idx] else 0.0
        regret += 1 - r

        ai = a_inv[arm] @ phi
        a_inv[arm] -= np.outer(ai, ai) / (1.0 + phi @ ai)
        a_inv[arm] = (a_inv[arm] + a_inv[arm].T) * 0.5
        b[arm] += phi * r
        theta[arm] = a_inv[arm] @ b[arm]

        buf_x.append(x[idx]); buf_a.append(arm); buf_r.append(r)

        if t % retrain_every == 0:
            bx = torch.from_numpy(np.array(buf_x))
            bt = torch.from_numpy(np.array([theta[a] for a in buf_a], dtype=np.float32))
            br = torch.from_numpy(np.array(buf_r, dtype=np.float32)).unsqueeze(1)
            for _ in range(steps_per_retrain):
                pred = (net(bx) * bt).sum(1, keepdim=True)
                loss = 0.5 * ((pred - br) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            # Re-accumulate arm statistics in the new embedding space.
            with torch.no_grad():
                emb = net(bx).double().numpy()
            a_inv = [np.eye(embed_dim) / lam for _ in range(k)]
            b = [np.zeros(embed_dim) for _ in range(k)]
            for e, a, rr in zip(emb, buf_a, buf_r):
                ai = a_inv[a] @ e
                a_inv[a] -= np.outer(ai, ai) / (1.0 + e @ ai)
                a_inv[a] = (a_inv[a] + a_inv[a].T) * 0.5
                b[a] += e * rr
            theta = [a_inv[a] @ b[a] for a in range(k)]

    return regret


# --- ARM B: paper block encoding ----------------------------------------

def run_block(x, y, k, nu, seed, rounds, hidden, steps_per_round,
              buffer_cap=1000, lam=1.0):
    d = x.shape[1]
    net = mlp(k * d, hidden, 1, seed)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    params = [p for p in net.parameters() if p.requires_grad]
    p_count = sum(p.numel() for p in params)
    u = np.full(p_count, lam)          # diagonal approximation of U

    buf_xa, buf_r = [], []
    regret = 0

    def block(ctx_np, a):
        v = np.zeros(k * d, dtype=np.float32)
        v[a * d:(a + 1) * d] = ctx_np
        return v

    for t, idx in enumerate(stream(x, y, rounds, seed), 1):
        ctx_np = x[idx]
        cand = np.stack([block(ctx_np, a) for a in range(k)])
        cand_t = torch.from_numpy(cand)

        scores, grads = [], []
        for a in range(k):
            net.zero_grad()
            out = net(cand_t[a:a + 1]).squeeze()
            g = torch.autograd.grad(out, params, retain_graph=False)
            gv = torch.cat([gi.reshape(-1) for gi in g]).detach().numpy().astype(np.float64)
            bonus = nu * np.sqrt(np.sum(gv * gv / u))
            scores.append(float(out.detach()) + bonus)
            grads.append(gv)

        arm = int(np.argmax(scores))
        r = 1.0 if arm == y[idx] else 0.0
        regret += 1 - r

        u += grads[arm] ** 2
        buf_xa.append(cand[arm]); buf_r.append(r)
        if len(buf_xa) > buffer_cap:
            buf_xa.pop(0); buf_r.pop(0)

        bx = torch.from_numpy(np.array(buf_xa))
        br = torch.from_numpy(np.array(buf_r, dtype=np.float32)).unsqueeze(1)
        for _ in range(steps_per_round):
            loss = 0.5 * ((net(bx) - br) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    return regret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="shuttle")
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=100, help="paper uses 100")
    ap.add_argument("--embed-dim", type=int, default=0, help="0 = max(8, n_arms)")
    ap.add_argument("--nu-grid", default="1.0,0.1,0.01")
    ap.add_argument("--steps-per-round", type=int, default=2,
                    help="arm B gradient steps per round; arm A is matched to the same total")
    ap.add_argument("--retrain-every", type=int, default=100)
    args = ap.parse_args()

    meta, x, y = load(args.dataset)
    k = meta["n_arms"]
    embed_dim = args.embed_dim or max(8, k)
    nus = [float(v) for v in args.nu_grid.split(",")]

    # Equal gradient budget: A does (rounds/retrain_every) retrains of S steps,
    # B does rounds*steps_per_round steps. Match the products.
    total_steps = args.rounds * args.steps_per_round
    steps_per_retrain = max(1, total_steps // max(1, args.rounds // args.retrain_every))

    print(f"dataset={args.dataset}  rows={len(y):,}  d={x.shape[1]}  arms={k}")
    print(f"T={args.rounds:,}  seeds={args.seeds}  hidden={args.hidden}  "
          f"embed_dim={embed_dim}  nu grid={nus}")
    print(f"gradient budget: both arms ~{total_steps:,} steps "
          f"(A: {args.rounds // args.retrain_every} retrains x {steps_per_retrain}; "
          f"B: {args.rounds} rounds x {args.steps_per_round})")
    print(f"random baseline regret = {args.rounds * (1 - 1/k):,.0f}   "
          f"majority = {args.rounds * (1 - max(meta['class_balance'])/sum(meta['class_balance'])):,.0f}\n")

    results = {}
    for label, fn in (("banditdb (shared MLP + per-arm heads)", "A"),
                      ("block encoding (one net over k*d)", "B")):
        print(f"--- {label}")
        best = None
        for nu in nus:
            runs = []
            for s in range(args.seeds):
                t0 = time.time()
                if fn == "A":
                    reg = run_banditdb(x, y, k, nu, s, args.rounds, args.hidden,
                                       embed_dim, args.retrain_every, steps_per_retrain)
                else:
                    reg = run_block(x, y, k, nu, s, args.rounds, args.hidden,
                                    args.steps_per_round)
                runs.append(reg)
                print(f"    nu={nu:<6} seed={s}  regret={reg:>6,.0f}  ({time.time()-t0:.0f}s)")
            m = st.mean(runs)
            if best is None or m < best[1]:
                best = (nu, m, runs)
        results[label] = best
        print(f"  best: nu={best[0]}  regret={best[1]:,.1f}  runs={best[2]}\n")

    print("=" * 74)
    print(f"{'architecture':<42}{'best nu':>9}{'regret':>11}")
    print("-" * 74)
    for label, (nu, m, _) in results.items():
        print(f"{label:<42}{nu:>9}{m:>11,.1f}")
    vals = [v[1] for v in results.values()]
    a, b = vals[0], vals[1]
    verdict = "block encoding WINS" if b < a else "banditdb architecture WINS"
    print("-" * 74)
    print(f"{verdict}  (ratio block/banditdb = {b/a:.2f})")
    print("=" * 74)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"block_ab_{args.dataset}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(
        {"dataset": args.dataset, "rounds": args.rounds, "seeds": args.seeds,
         "hidden": args.hidden, "embed_dim": embed_dim,
         "results": {kk: {"nu": v[0], "mean_regret": v[1], "runs": v[2]}
                     for kk, v in results.items()}}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
