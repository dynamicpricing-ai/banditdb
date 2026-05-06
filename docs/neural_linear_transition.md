# NeuralLinear Phase Transition — Warm Start Design

## The Problem: "Same Result" Is Not Achievable Exactly

Phase 1 (LinUCB on raw context) and Phase 2 (LinUCB on neural embedding) operate in completely
different vector spaces:

```
Phase 1:  score_k = θ_k(d) · x_d  +  α √(x_dᵀ A_k(d×d)⁻¹ x_d)
Phase 2:  score_k = θ_k(e) · φ(x)  +  α √(φ(x)ᵀ A_k(e×e)⁻¹ φ(x))
```

Where `d = context_dim` (e.g. 10) and `e = embed_dim` (e.g. 387). The matrices live in
different spaces — there is no algebraic path from `A(d×d)` to `A(e×e)`. Exact score
preservation across the transition is impossible.

What *is* achievable: **behavioral continuity** — the arm that was winning before the transition
still scores highest after, and uncertainty is calibrated rather than reset to zero-knowledge.

---

## Warm Start via Re-accumulation

The WAL stores raw context vectors for every interaction. At transition time the LinUCB
statistics are re-derived in the new embedding space using all historical data:

```
For each arm k:

  1. Collect all completed interactions for arm k from the buffer:
       { (x_i, r_i) : arm_i == k }

  2. Compute embeddings through the just-trained MLP:
       φ_i = embed(x_i)              ← same historical contexts, new representation

  3. Re-accumulate LinUCB statistics in embed_dim space:
       A_new_k = Σ φ_i φ_iᵀ  +  I   ← identical formula to online updates
       b_new_k = Σ r_i φ_i
       θ_new_k = A_new_k⁻¹ b_new_k

  4. Replace arm state with (A_new_k, b_new_k, θ_new_k)
```

This is mathematically equivalent to running LinUCB from scratch in embedding space on all
historical data — not a heuristic, the correct initialization.

---

## What Warm Start Guarantees

**Calibrated uncertainty.** `A_new` has accumulated the correct number of outer products, so
UCB exploration bonuses reflect actual data coverage. No artificial over-exploration on
transition day.

**Arm ordering preserved.** If Arm A genuinely outperformed Arm B in Phase 1, that advantage
survives re-accumulation — both models estimate the same underlying reward function, just with
different capacity. The winning arm stays the winning arm.

**No catastrophic forgetting.** Cold reset (A = I, b = 0) would trigger massive exploration
immediately because every arm looks equally uncertain. Warm start eliminates this entirely.

---

## The Two Scenarios — and Why They Matter

### Scenario A: Same raw input, richer representation

The caller sends the same underlying data throughout. Phase 1 uses 10 features directly. Phase 2
feeds the same 10 features through an MLP that maps to 387-dim (learned nonlinear basis
functions — essentially a kernel trick in disguise).

Warm start works perfectly here: all raw inputs are already in the WAL buffer and can be
re-embedded through the new network.

### Scenario B: Different input sources across phases

Phase 1: caller sends 10 hand-crafted features.
Phase 2: caller sends a 387-dim LLM embedding of raw text.

These are genuinely different data sources. You cannot re-embed Phase 1 interactions through
the Phase 2 network because the raw text was never logged — only the 10-dim features were.

**This is a real design constraint.** If Scenario B is the intent, the WAL must log the richer
raw input from day one, even if Phase 1 only uses 10 dimensions of it. Otherwise warm start is
impossible and the system falls back to cold reset.

---

## The Transition Invariant

```
The interaction buffer stores raw context at context_dim_phase2 resolution
from campaign creation, even during Phase 1. Arm matrices run at embed_dim
throughout. When the MLP initialises at the Phase 1 → Phase 2 boundary,
warm start re-derives arm matrices from buffered raw contexts via embed().
```

This single constraint prevents the Scenario B ambiguity from ever becoming a problem at
transition time.

---

## The Expansion Direction Warning

Going from 10-dim → 387-dim via MLP is atypical. Neural embeddings normally *compress*
high-dimensional inputs. Expanding is more like a learned kernel method.

Practical concern: with 500 interactions and a `10 → 128 → 387` network you have far more
parameters than data — overfitting risk is high.

Mitigations:
- Use a shallow architecture: `10 → 32 → 387` or even `10 → 387` (single linear layer)
- Apply dropout and weight decay aggressively
- Reconsider whether context_dim should be larger from the start

**Rule of thumb:**

| Direction | Example | Classification |
|---|---|---|
| context_dim > embed_dim | 1536 → 32 | Compression — typical, recommended |
| context_dim < embed_dim | 10 → 387 | Expansion — kernel-like, needs caution |
| context_dim == embed_dim | 32 → 32 | Nonlinear re-representation |

If `context_dim < embed_dim`, treat the MLP as a kernel method, not a feature extractor.
The recommended and typical case is `context_dim >> embed_dim`.

---

## Cold Reset Fallback

Warm start requires sufficient data per arm. If the buffer has fewer than `min_samples_per_arm`
interactions for a given arm at transition time, fall back to cold reset for that arm:

```
A_k = I (embed_dim × embed_dim)
b_k = 0 (embed_dim)
θ_k = 0 (embed_dim)
```

That arm will explore aggressively until it accumulates data in the new embedding space.
Arms with sufficient data get warm start; sparse arms get cold reset. Both are correct
behaviors for their respective data situations.

Suggested `min_samples_per_arm` default: 20 interactions.

---

## Transition Checklist

- [ ] Buffer stores context at `context_dim_phase2` resolution from campaign creation
- [ ] MLP trained on buffer before re-accumulation begins
- [ ] Re-accumulation runs per arm, not globally
- [ ] Arms below `min_samples_per_arm` receive cold reset, not warm start
- [ ] New `A_new`, `b_new`, `theta_new` written atomically (swap, not in-place update)
- [ ] Transition event emitted to WAL so recovery knows which phase to resume in
- [ ] Checkpoint written immediately after transition (avoids replaying Phase 1 WAL through Phase 2 embed on next recovery)
