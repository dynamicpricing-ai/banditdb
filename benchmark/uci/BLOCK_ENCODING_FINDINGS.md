# Block encoding head-to-head

**Question:** is BanditDB's neural architecture — a shared MLP with per-arm linear heads — representationally weaker than the `k*d` disjoint block encoding used by NeuralUCB (Zhou et al. 2019) and NeuralTS (Zhang et al. ICLR 2021)?

**Answer: no.** There is no consistent architectural advantage in either direction.

Run it: `python benchmark/uci/block_encoding_ab.py --dataset shuttle --rounds 3000 --seeds 2 --steps-per-round 20`

## Method

Both architectures run inside `block_encoding_ab.py`, so the architecture is the only variable. Held identical: dataset, shuffle order, seeds, network width (100 hidden units, matching the paper), total gradient budget, and the exploration grid (best of ν ∈ {1.0, 0.1} reported for both, matching the paper's protocol).

* **Arm A (BanditDB)** — shared MLP `φ_W: R^d → R^m`, per-arm ridge heads. `score_a = φ(x)·θ_a + ν·√(φ(x)ᵀA_a⁻¹φ(x))`. Periodic retrain of W on `Σ(φ(x_t)·θ_{a_t} − r_t)²` followed by re-accumulation of arm statistics in the new embedding space. Mirrors `src/neural.rs` Algorithm 2.
* **Arm B (paper)** — one network `f_W: R^{k·d} → R` over `x_a = [0;…;x;…;0]`. `score_a = f(x_a) + ν·√(Σ g_a[i]²/U[i])` with `g_a = ∇_W f(x_a)` and `U` the diagonal approximation the paper uses.

T = 3,000 rather than 10,000 because arm B needs one backward pass per arm per round to form gradient features. `embed_dim` follows the server's rule `min(32, max(8, d/2))`.

## Results

Cumulative regret, T=3,000, 2 seeds, 20 gradient steps/round, best ν:

| dataset | arms | d | BanditDB arch | block encoding | winner | ratio |
|---|---|---|---|---|---|---|
| shuttle | 7 | 9 | **56.5** | 97.5 | BanditDB | 1.73x |
| mushroom | 2 | 76 | 47.5 | **19.0** | block | 2.50x |
| magic | 2 | 10 | 527.0 | 516.5 | tie | 1.02x |
| adult | 2 | 96 | 700.5 | 707.5 | tie | 1.01x |

Block encoding wins decisively on one dataset, loses decisively on one, ties on two. **Nothing here supports replacing BanditDB's architecture.**

This also matches the theory: `f(x,a) = φ_W(x)ᵀθ_a` with `embed_dim >= k` is fully general — take `φ(x) = [f(x,1),…,f(x,k)]` and `θ_a = e_a`. There was never a representational gap to close.

## What actually dominates: training budget

Raising gradient steps from 2/round to 20/round on shuttle:

| architecture | 2 steps/round | 20 steps/round | improvement |
|---|---|---|---|
| BanditDB | 122.5 | 56.5 | 2.2x |
| block encoding | 969.0 | 97.5 | 9.9x |

Training budget moves regret by up to 10x. Architecture moves it by at most 2.5x, in either direction. Block encoding is the more budget-hungry of the two, which is expected — its first layer has `k·d·hidden` parameters against BanditDB's `d·hidden`.

## Block encoding is unstable

At the lower budget, arm B collapsed on individual seeds: ν=0.1 produced regret 2,512 and 233 on two consecutive seeds of the same configuration (2,512 out of 3,000 rounds is barely above random). BanditDB's arm never showed this.

This reproduces the paper's own reported variance — NeuralUCB shuttle is **338.6 ±386.4**, a standard deviation larger than the mean, and NeuralTS is **232.0 ±149.5**. Their headline neural numbers come from a 20-point (λ, ν) grid search over a high-variance estimator.

## Actionable: the server under-trains

The server's neural retrain only fires from `checkpoint()`, gated by `should_retrain()`. Default settings give roughly 0.1 gradient steps per round.

Measured on shuttle, T=10,000, `neural_lin_ucb`:

| configuration | grad steps | regret |
|---|---|---|
| `checkpoint_every=1000, retrain_steps=100` (default-ish) | ~1,000 | 669 |
| `checkpoint_every=100, retrain_steps=2000` | ~200,000 | **525** |

A 22% improvement from configuration alone, no code change. Two follow-ups worth considering:

1. **Decouple retrain from checkpoint.** Tying MLP training to a durability operation means the training cadence is set by an unrelated concern. Checkpointing also does Parquet export and WAL rotation, so raising retrain frequency drags that cost along.
2. **`BUFFER_CAP = 5,000`** (`src/neural.rs:62`) — at T=10,000 the network trains on only the most recent half of the data.

## Conclusion

The earlier hypothesis that block encoding explains BanditDB's gap to the paper's neural numbers is **not supported**. The gap is better explained by training budget and hyperparameter search than by architecture. Recommendation: do not implement block encoding in the engine. Tune retrain cadence instead.
