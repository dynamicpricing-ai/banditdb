# Multi-Agent Game Environments — BanditDB as Shared Decision Infrastructure

In both scenarios below BanditDB acts as the **shared memory of what works in what context** — something no single agent could build alone, but a swarm naturally produces through accumulated interactions.

The core pattern is identical in both cases:

```
agent selects arm via get_intuition(context)
  → executes action
  → receives outcome score
  → records reward via record_outcome(iid, reward)
  → shared model improves for all agents
```

---

## Scenario 1: Negotiation Tournament

N agents negotiate contracts against each other in rounds. Each agent must pick a
negotiation tactic per encounter. No agent is told which tactic works — they learn
collectively through BanditDB which strategies beat which opponent types.

### Campaign

| Field | Value |
|---|---|
| Campaign ID | `negotiation_tactics` |
| Arms | `anchoring`, `collaborative`, `competitive`, `mirroring`, `delay` |
| Feature dim | 9 |
| Algorithm | LinUCB |

### Reward Signal

```
reward = (achieved_value − reservation_price) / (ideal_value − reservation_price)
```

How close to the best possible outcome, normalised 0–1. A deal at reservation price
scores 0.0; a deal at ideal value scores 1.0.

### Context Vector

| Index | Feature | Description | Calculation |
|---|---|---|---|
| 0 | `opponent_aggression` | How combative this opponent has been historically | `competitive_moves / total_moves` across all prior rounds with this opponent |
| 1 | `deal_value_norm` | Relative size of this deal in the game | `deal_value / max_deal_value_in_session` |
| 2 | `time_pressure` | How close to the deadline | `elapsed_rounds / max_rounds`, capped at 1.0 |
| 3 | `my_leverage` | Strength of my outside option (BATNA) | `my_best_alternative_value / current_deal_value`, capped at 1.0 |
| 4 | `relationship_history` | Quality of prior outcomes with this specific opponent | Rolling mean of past reward scores against this opponent; 0.5 if first encounter |
| 5 | `opponent_consistency` | How predictable the opponent's style is | `1 − normalised_stddev(opponent_last_5_moves)` — high = predictable |
| 6 | `information_asymmetry` | How much of the deal terms I can see vs are hidden | `known_fields / total_fields` in the deal struct |
| 7 | `tournament_stage` | Early rounds vs elimination rounds | `current_round / total_rounds` |
| 8 | `my_standing` | My current win rate in the tournament | `wins / games_played`; 0.5 at start |

### What Emerges

LinUCB learns that `anchoring` works against passive opponents but `mirroring`
beats aggressive ones. No single agent discovers this — the swarm does, because
every deal outcome feeds the shared model. After enough rounds a meta-strategy
emerges from the data without anyone programming it.

A useful experiment: run two swarms in parallel — one sharing a BanditDB campaign,
one with isolated agents — and measure convergence speed and final tournament
standings. The shared swarm will consistently outperform.

---

## Scenario 2: LLM Debate Arena

AI agents argue opposing positions on topics in front of a judge LLM. Each agent
selects an argument style per turn. The judge scores each exchange and the reward
flows back to BanditDB, which learns which styles prevail in which debate contexts.

### Campaign

| Field | Value |
|---|---|
| Campaign ID | `argument_strategy` |
| Arms | `rhetorical`, `data_driven`, `socratic`, `emotional`, `narrative` |
| Feature dim | 9 |
| Algorithm | LinUCB |

### Reward Signal

The judge LLM scores each argument on three dimensions and the mean is reported:

- **Persuasiveness** — did it shift the position?
- **Logical coherence** — was the reasoning sound?
- **Direct rebuttal quality** — did it address the opponent's last claim?

Score range: 0.0–1.0

### Context Vector

| Index | Feature | Description | Calculation |
|---|---|---|---|
| 0 | `topic_controversy` | How polarising the topic is | Pre-scored 0–1 per topic; or stddev of judge scores across prior debates on the same topic |
| 1 | `opponent_last_style` | Encoded style of opponent's most recent argument | `0.0=rhetorical` · `0.25=data_driven` · `0.5=socratic` · `0.75=emotional` · `1.0=narrative` |
| 2 | `debate_position` | Which side I'm arguing | `0.0=against` · `1.0=for` |
| 3 | `round_norm` | Where we are in the debate | `current_round / max_rounds` |
| 4 | `audience_sentiment` | Current judge leaning toward my position | Running sentiment score from judge LLM after each exchange; updated each round |
| 5 | `momentum` | Whether I've been winning recent exchanges | `mean(my_last_3_scores) − mean(opponent_last_3_scores)`, rescaled to 0–1 |
| 6 | `rebuttal_pressure` | How directly the opponent's last argument attacked mine | LLM-scored 0–1: *"How directly does this argument rebut the prior claim?"* |
| 7 | `claim_density` | How many distinct claims are in play | `total_claims_made / expected_max_claims`, capped at 1.0 — high = crowded debate |
| 8 | `opponent_consistency` | How predictable the opponent's style choices have been | `1 − entropy(opponent_style_distribution)` normalised — high = they keep using the same arm |

### What Emerges

BanditDB learns that `socratic` dismantles `data_driven` opponents but `emotional`
outperforms it on ethical topics. Agents develop genuine counter-strategy reflexes
— not through explicit programming but through accumulated judgment stored in the
shared model.

The `opponent_last_style` and `opponent_consistency` features are particularly
valuable for LinUCB: they create distinct enough contexts that the model learns
counter-strategies, not just "what works in general" but "what works against this
type of opponent in this situation."

---

## Shared Infrastructure Pattern

Both scenarios expose the same architectural insight: **BanditDB is not per-agent
state, it is collective intelligence**.

```
agent_1 ──get_intuition──► BanditDB ──► arm selection
agent_2 ──get_intuition──►    │     ──► arm selection
agent_N ──get_intuition──►    │     ──► arm selection
                               │
agent_1 ──record_outcome──►   ▼
agent_2 ──record_outcome──► shared model updates
agent_N ──record_outcome──► all agents benefit
```

The more agents participate, the faster the model converges. This is the
property that makes BanditDB useful as game infrastructure rather than just
a per-agent decision tool.
