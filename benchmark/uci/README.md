# UCI Contextual Bandit Benchmark

Reproduces the benchmark suite from **Zhang, Zhou, Li & Gu, "Neural Thompson Sampling", ICLR 2021** — the paper BanditDB's `NeuralThompsonSampling` implements — against a live BanditDB server.

```bash
python benchmark/uci/convert.py                     # download already done; encodes to bandit format
cargo build --release --features neural
DATA_DIR=/tmp/bd BANDITDB_API_KEY=k BANDITDB_RATE_LIMIT_PER_SEC=100000 ./target/release/banditdb &
BANDITDB_URL=http://localhost:8080 BANDITDB_API_KEY=k \
  python benchmark/uci/evaluate.py --all --rounds 10000 --repeats 3 --log-every 100
python benchmark/uci/analyze.py
```

> The default 1,000 rps rate limit will silently fail the sweep. Raise `BANDITDB_RATE_LIMIT_PER_SEC`.

## What is being measured

These are classification datasets converted to bandits by the standard reduction:

```
k classes -> k arms,  feature vector -> context,  reward = 1 if chosen arm is the true class else 0
```

The algorithm is **never shown the correct answer**. It picks one class and learns only whether its own guess was right — bandit (partial) feedback, not supervised feedback.

**Cumulative regret = total mistakes over T rounds.** A perfect policy makes zero.

Unlike a logged-feedback replay, this evaluation is **exact**: the label is known for every arm, so nothing is estimated, importance-weighted, or discarded. (For contrast, the MovieLens replay in `benchmark/movielens/` discards ~93% of events and can only estimate.)

## The datasets

| dataset | rows (ours) | dim | arms | what it is |
|---|---|---|---|---|
| **mushroom** | 5,644 | 76 | 2 | Mushroom specimens described by 22 categorical traits (cap shape, **odor**, gill size, spore print, habitat). Edible or poisonous. Near-linearly-separable — odor alone nearly decides it. Sanity check. |
| **magic** | 19,020 | 10 | 2 | Events from a ground-based gamma-ray telescope. 10 continuous features describing the geometry of the Cherenkov light ellipse (length, width, size, concentration, asymmetry, angle). Gamma ray (signal) or hadron (background). Classes genuinely overlap. |
| **shuttle** | 58,000 | 9 | 7 | NASA space shuttle radiator sensor readings. 9 integer sensors, 7 operating states. ~79% of rows are a single class. The only multi-class set here. |
| **adult** | 45,222 | 96 | 2 | 1994 US Census records. Age, workclass, education, marital status, occupation, race, sex, capital gain/loss, hours worked. Earns >$50K/year or not. |

Preprocessing: numeric columns min-max scaled to [0,1], categoricals one-hot encoded, then **every row L2-normalised to unit norm**. That last step is not cosmetic — see [Context normalisation](#context-normalisation).

## Results

T = 10,000 (mushroom 5,644 — all rows available), 3 seeds, α = 1.0 untuned.

| dataset | LinUCB | ThompsonSampling | NeuralLinUCB | NeuralTS | random | majority class |
|---|---|---|---|---|---|---|
| mushroom | **35** | 208 | 268 | 465 | 2,822 | 2,157 |
| magic | 2,062 | 2,134 | **2,009** | 2,186 | 5,000 | 3,516 |
| shuttle | 709 | 1,102 | **669** | 1,147 | 8,571 | 2,140 |
| adult | **1,716** | 2,002 | 1,937 | 2,166 | 5,000 | 2,478 |

### Learning diagnostics

`beta` is the slope of log(regret) against log(t). Regret grows as t^beta: **beta ≈ 0.5** matches the √T theory, **beta < 1** means the policy is still improving, **beta ≈ 1** means a flat error rate — no learning.

| dataset | algorithm | beta | acc first 10% | acc last 10% | rounds to beat majority |
|---|---|---|---|---|---|
| mushroom | linucb | **0.232** | 0.9553 | 1.0000 | 100 |
| mushroom | thompson_sampling | 0.425 | 0.8307 | 0.9967 | 100 |
| mushroom | neural_ts | 0.572 | 0.7653 | 0.9492 | 166 |
| shuttle | neural_ts | 0.660 | 0.7467 | 0.9093 | 1,733 |
| shuttle | neural_lin_ucb | 0.661 | 0.8420 | 0.9496 | 366 |
| shuttle | thompson_sampling | 0.685 | 0.7733 | 0.9193 | 1,400 |
| shuttle | linucb | 0.704 | 0.8568 | 0.9307 | 316 |
| adult | thompson_sampling | 0.856 | 0.7273 | 0.8156 | 2,533 |
| adult | neural_ts | 0.897 | 0.7273 | 0.7856 | 2,700 |
| magic | all four | 0.91–0.96 | ~0.75 | ~0.80 | 100–166 |
| adult | linucb / neural_lin_ucb | 0.93–0.94 | ~0.80 | ~0.82 | 166–233 |

Reading it: **mushroom is solved** (beta 0.23, converges to 100%). **shuttle is genuinely learned** by everything (beta 0.66–0.70). **magic and adult are barely learned** — beta near 1 and accuracy climbing only 2–6 points over 10,000 rounds. Magic is worth treating as a negative control.

## Comparison with the paper

Zhang et al. report final regret in Table 1 (appendix), averaged over 20 runs.

| dataset | algorithm | BanditDB | paper | ratio | like-for-like? |
|---|---|---|---|---|---|
| **shuttle** | **LinUCB** | **709** | **966.6 ±39.0** | **0.73** | **yes** |
| shuttle | LinTS | 1,102 | 1,020.9 ±42.8 | 1.08 | yes |
| shuttle | Neural*UCB | 669 | 338.6 ±386.4 | 1.98 | no — architecture |
| shuttle | NeuralTS | 1,147 | 232.0 ±149.5 | 4.94 | no — architecture |
| magic | LinUCB | 2,062 | 2,604.4 ±34.6 | 0.79 | no — features |
| magic | Neural*UCB | 2,009 | 2,033.0 ±48.6 | 0.99 | no — architecture |
| adult | LinUCB | 1,716 | 2,097.5 ±50.3 | 0.82 | no — features |
| mushroom | LinUCB | 35 | 562.7 ±23.1 | 0.06 | no — horizon + features |

### What is and isn't comparable

**Linear models are legitimately comparable.** The paper uses disjoint block encoding — one shared θ ∈ ℝ^{kd} scoring `x_a = [0;…;x;…;0]`. That expands to exactly `θ_a · x`, which is BanditDB's per-arm model. Mathematically identical. Verified empirically: an independent numpy LinUCB written against `src/math.rs` reproduces BanditDB's shuttle number exactly (709 = 709).

**Neural models are not.** The paper feeds the k·d block vector through *one* network, so it sees which arm it is scoring and can learn arm-specific nonlinearity. BanditDB shares an MLP over the d-dim context and keeps per-arm **linear** heads in embedding space — architecturally closer to "Neural-Linear" (Riquelme et al. 2018) than to NeuralUCB/NeuralTS.

**Other differences, all favouring the paper:** they grid-search ν ∈ {1, 0.1, 0.01} for linear (λ ∈ {1…1e-3}, ν ∈ {1e-1…1e-5} for neural) and report the *best*; we run a single untuned α=1.0. They average 20 runs; we run 3. Their feature encodings differ from ours (adult 15 dims vs our 96, magic 12 vs 10, mushroom 23 vs 76), and they keep all 8,124 mushroom rows where we drop rows with missing `stalk-root` (5,644).

### Verdict

**Shuttle linear is the one true apples-to-apples cell** — same 9 raw sensor features, same 7 classes, same T, same 8,571 random baseline. There BanditDB's LinUCB scores **709 vs the published 966.6, 27% better**, without the ν tuning the paper used.

That is a statement about correctness and input conditioning, not algorithmic superiority — LinUCB is LinUCB, and BanditDB implements it faithfully. The magic and adult margins largely reflect richer feature encodings, not a better solver.

**The neural path is more nuanced.** BanditDB's neural variants help where the target is nonlinear (shuttle 669 vs 709, magic 2,009 vs 2,062) and hurt where it is near-linear (mushroom 268 vs 35, adult 1,937 vs 1,716). That is ordinary bias–variance: extra capacity costs estimation variance when the true function is simple. It is not evidence of a defect.

Two hypotheses for the residual gap to the paper's neural numbers were tested and **both failed**:

* *Architecture.* BanditDB scores `f(x,a) = φ_W(x)ᵀθ_a`. With `embed_dim >= k` this form is fully general — take `φ(x) = [f(x,1),…,f(x,k)]` and `θ_a = e_a`. Every configuration here satisfies that (k=2 with embed_dim 32; shuttle k=7 with embed_dim 8). There is no representational gap.
* *Training compute.* The paper runs 100 gradient steps per round (~1e6 total); BanditDB retrains only at checkpoint. Raising retrains 10x (checkpoint every 100 rounds, `retrain_every=50`) moved mushroom 250 -> 214 and shuttle 579 -> 598. Retrain logs show `initial_loss` drifting 0.0155 -> 0.0151 across consecutive retrains: the MLP already fits its buffer and is not underfitting.

**The open gap:** the paper's NeuralTS reaches 232.0 on shuttle against BanditDB's best neural 579. That is real and currently unexplained. Context: their reported deviations are very large (NeuralTS shuttle 232.0 ±149.5; NeuralUCB 338.6 ±386.4, std exceeding the mean) and they select the best of a 20-point (λ, ν) grid, where BanditDB ran untuned. Settling it requires implementing block encoding and comparing head-to-head — until then, treat any claim that BanditDB's neural architecture is weaker as unproven.

## Context normalisation

The single largest effect measured in this whole exercise, and it is a client-side preprocessing choice, not a code path.

LinUCB scores `θ·x + α·√(xᵀA⁻¹x)`. The exploration term scales with `‖x‖`, and the regret analysis assumes `‖x‖ ≤ 1`. With per-column standardisation, shuttle contexts reached `‖x‖ = 123` against a mean of 2.09 — rare high-norm rows produced enormous exploration bonuses that dominated arm selection.

Same code, same α, shuttle only:

| context scaling | LinUCB regret |
|---|---|
| per-column standardisation | 2,026 |
| standardisation then L2 | 2,381 |
| raw | 1,230 |
| L2 on raw | 981 |
| **min-max then L2** | **709** |

Note that standardisation and L2 **do not compose** — applying L2 on top of standardised features is worse than either alone. Scale into a bounded range first, then normalise the row.

## Files

| file | purpose |
|---|---|
| `convert.py` | UCI source → bandit JSONL. `--no-l2` restores the old standardisation-only behaviour. |
| `evaluate.py` | Replays a dataset against a live server, all four algorithms. |
| `analyze.py` | Learning diagnostics (beta, accuracy lift, time-to-beat-baseline) and the paper comparison. |
| `diagnose_shuttle.py` | Root-cause script for the shuttle gap: reference numpy LinUCB plus a preprocessing sweep. |
| `results/` | Timestamped run output. `archive_zscore/` holds pre-normalisation runs for reference. |
