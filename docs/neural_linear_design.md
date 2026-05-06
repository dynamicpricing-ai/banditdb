# NeuralLinear Design — BanditDB

## Strategic Motivation

LinUCB and Thompson Sampling are well-understood algorithms. BanditDB's current moat is
infrastructure (WAL durability, checkpoint/recovery, HTTP API), not the algorithms themselves.
Neural methods are the path to relevance in the 2026 LLM-native application stack.

### The mid-market gap

| Capability | Tech Giants | Mid-Market / Enterprise |
|---|---|---|
| A/B testing | ✓ | ✓ (Optimizely, LaunchDarkly) |
| Multi-armed bandit | ✓ | Partial |
| Contextual bandit (linear) | ✓ | Almost nobody |
| Neural contextual bandit | ✓ (internal only) | **Nobody** |
| Bandit on LLM embeddings | ✓ (internal only) | **Nobody** |

### Why now

LLM applications produce dense embedding vectors (768–3072 dim) as their natural context.
These break LinUCB's assumptions:
- A 1536×1536 A⁻¹ matrix per arm ≈ 9 MB; 100 arms ≈ 900 MB
- Reward does not vary linearly with embedding components
- Useful signal lives in nonlinear subspaces

NeuralLinear solves both: it compresses context into a small embed space where linear methods
work well, and lets the network discover nonlinear feature interactions.

### Agentic use cases

- **Prompt template selection** — context: query embedding, arms: prompt variants, reward: quality score
- **Model/provider routing** — context: task embedding, arms: GPT-4o / Claude / Gemini, reward: quality/cost
- **RAG strategy selection** — context: query type embedding, arms: retrieval strategies
- **Agent tool routing** — context: conversation state, arms: available tools

---

## Core Insight: NeuralLinear is LinUCB with a preprocessing step

From `math.rs`'s perspective — `score()`, `score_ts()`, `update()` — nothing changes. Those
functions receive a feature vector. They do not need to know where it came from.

```
// Today
score(arm, context, alpha)           // context is raw, dim-dimensional

// NeuralLinear
score(arm, embed(context), alpha)    // embed() is identity for linear, MLP for neural
```

One line change in the hot path. Everything else is additive.

---

## What Changes vs. What Stays Frozen

### Frozen — zero changes
- `math.rs` entirely: `score()`, `score_ts()`, `update()`, `cholesky()`
- WAL event types and serialization
- HTTP API (except campaign creation params)
- `ArmState` struct — `a_inv`, `b`, `theta` remain correct, just sized to `embed_dim`

### Additive — new code alongside existing
- `embed()` method on Campaign (identity for linear algorithms)
- `Option<EmbeddingNetwork>` field on Campaign state
- Per-campaign interaction buffer for training data
- Training loop triggered at checkpoint time

### Modified — existing things that grow
- `CheckpointData` gains optional serialized network weights (safetensors bytes)
- `Algorithm` enum gains `NeuralLinear` variant
- `dim` concept splits into `context_dim` and `embed_dim`

---

## The Dimension Split

Currently `dim` means one thing everywhere. NeuralLinear requires two:

```rust
struct Campaign {
    context_dim: usize,  // what the caller sends (e.g. 1536)
    embed_dim: usize,    // what ArmState matrices are sized to (e.g. 32)
    // Linear algorithms: context_dim == embed_dim, embed() is identity
    // NeuralLinear:      context_dim > embed_dim, embed() runs MLP
}
```

`ArmState` matrices are always `embed_dim × embed_dim`. For existing linear campaigns nothing
changes — `embed_dim` equals `dim` today. Fully backward compatible.

---

## Auto-Upgrade: The Smart Switch

Rather than forcing customers to choose NeuralLinear upfront (cold start problem), campaigns
self-upgrade when enough data has accumulated.

```
Campaign created with Algorithm::NeuralLinear
│
├── Phase 1: 0 → N interactions
│   └── Runs pure LinUCB in context_dim space
│       embed() returns context unchanged
│       Accumulates completed interactions in buffer
│
└── Phase 2: N interactions crossed (at next checkpoint)
    └── Train initial MLP on buffer
        Re-initialize arm matrices in embed_dim space
        embed() now runs MLP forward pass
        Future scoring uses neural features
```

`N` is a campaign config param (suggested default: 500). Below threshold the customer gets
standard LinUCB for free. Above threshold they get neural features automatically — no manual
intervention.

**Customer pitch:** "Starts working immediately. Upgrades itself as your data grows."

---

## Training Loop Design

Training runs at checkpoint time, not online. This preserves the WAL hot path entirely.

```
Checkpoint event fires
│
├── Existing flow (unchanged)
│   └── flush barrier → snapshot matrices → write checkpoint.json
│       → parquet export → WAL rotation
│
└── New: if NeuralLinear && buffer.len() >= threshold
    └── Train MLP on interaction buffer (async, non-blocking)
        Update campaign.embedding in memory
        Serialize weights into checkpoint
        Next scoring cycle picks up new embedding
```

Training is purely a checkpoint-time batch operation. No new WAL event types required.

---

## Interaction Buffer

Training needs `(context, arm_id, reward)` tuples — already present in `CompletedInteraction`
records flowing through the WAL.

Campaigns with `NeuralLinear` maintain an in-memory ring buffer of recent completed interactions
(cap: e.g. 10k entries). At checkpoint time this is the training set. No new WAL events.

---

## Recovery Story

On restart:
1. Load checkpoint — network weights are safetensors bytes in checkpoint (sidecar or inline)
2. Deserialize `EmbeddingNetwork` from weights
3. Replay WAL tail events since last checkpoint
4. WAL events contain **raw context vectors, never embeddings** — re-embed on replay

**Key invariant:** WAL always stores raw context. If the network changes at next checkpoint,
old WAL events replay correctly through the new network. This keeps recovery deterministic.

---

## MLP Architecture

Opinionated defaults, minimal config surface:

```
Input:   context_dim  (e.g. 1536)
Hidden:  hidden_dim   (default: 128, configurable)
Hidden:  hidden_dim
Output:  embed_dim    (default: 32, configurable)
Activation: ReLU
```

Campaign creation gains two optional params: `embed_dim` (default 32) and `hidden_dim`
(default 128). Layer count fixed at 2 for V1 — avoids overfitting on mid-market data volumes.

Loss: MSE between predicted reward (linear head θ_arm · embed(context)) and actual reward.
The network learns a representation where linear prediction of reward works well.

---

## Feature Flag: Zero Cost for Existing Users

```toml
[features]
default = []
neural = ["candle", "candle-nn"]
```

All MLP code lives behind `#[cfg(feature = "neural")]`. Default builds: same binary size, same
performance, no candle dependency. `embed()` compiles to an identity function — eliminated by
the compiler.

Neural builds opt in: `cargo build --features neural`.

GPU (CUDA/Metal) is a further opt-in via candle's own feature flags, not required for CPU-only
neural inference.

---

## Expected KPI Improvements

| KPI | Mechanism | Expected gain |
|---|---|---|
| Cumulative regret | Better feature extraction for nonlinear reward-context | 20–40% reduction vs LinUCB on LLM embedding contexts |
| Memory at scale | embed_dim matrices instead of context_dim matrices | 1536-dim → 32-dim = 2300× smaller per-arm state |
| Time to convergence | Shared embedding transfers signal across arms | Faster convergence with many arms, uneven traffic |
| Cold start for new arms | New arms inherit pretrained embedding | Meaningful from first interaction |
| Scoring latency (GPU) | Batch all-arm forward pass in one kernel | ~100× faster vs sequential CPU at K=1000 arms |

---

## Open Questions

1. **Warm start on Phase 1 → Phase 2 transition** — reset arm matrices cold (simple) or
   project existing LinUCB weights through the new embedding (better regret, more complex)?
   Suggest: cold reset for V1, warm start as V2 improvement.

2. **embed_dim sizing** — fixed at campaign creation or auto-sized from context_dim? Suggest:
   fixed at creation, document a rule of thumb (embed_dim ≈ context_dim / 16, min 16, max 64).

3. **Buffer eviction policy** — FIFO ring buffer (simple, recency bias) or reservoir sampling
   (unbiased)? Reservoir sampling is better for training but adds complexity.

4. **Training on GPU vs CPU** — for V1, CPU training is acceptable (small MLP, batch at
   checkpoint). GPU training is a follow-on once the CPU path is validated.

5. **Checkpoint weight storage** — inline in checkpoint.json as base64 (simple) or sidecar
   `.safetensors` file (cleaner, standard format)? Sidecar preferred for large networks.

---

## Implementation Order

1. Prototype `embed()` injection point and dimension split — everything else follows from this
2. `EmbeddingNetwork` struct behind feature flag, CPU forward pass only
3. Auto-upgrade state machine (Phase 1 → Phase 2 at checkpoint)
4. Training loop at checkpoint time
5. Checkpoint serialization of weights
6. Recovery: deserialize weights + re-embed WAL tail
7. GPU support via candle CUDA/Metal features
