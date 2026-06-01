# 07 · Prompt Optimisation (LLM Evals in Production)

> A continuous prompt evaluation engine that learns which prompt strategy produces the highest-quality responses for each task type — replacing static offline benchmarks with a live, contextual feedback loop running in production.

---

## Problem

Prompt engineering today follows an offline cycle: write several variants, run them against a fixed test set, pick the winner, deploy it uniformly to all users, and revisit it months later when someone notices degradation.

This approach has three fundamental flaws:

- **The winning prompt on average is rarely best for every task.** A chain-of-thought prompt that excels at mathematical reasoning adds unnecessary verbosity to a simple conversational query.
- **Evals live in a spreadsheet, not in production.** The test set you benchmarked against does not represent the distribution of real user requests — and that distribution shifts over time.
- **You stop learning after deployment.** Once a prompt is shipped, you collect no signal about whether it was actually the right choice for each specific request.

BanditDB replaces the offline benchmark with a continuous production loop. Every request is an experiment. Every judge score is a data point. The system learns — from real traffic — which prompt strategy wins for which task context, and routes accordingly from the very next request.

---

## Arms

| Arm | Description |
|-----|-------------|
| `zero_shot` | Bare instruction with no examples or reasoning scaffolding. Fast, low-token, effective for simple and conversational tasks. |
| `chain_of_thought` | Instructs the model to reason step by step before answering. Strong for complex reasoning, mathematics, and multi-step problems. Adds tokens and latency. |
| `few_shot` | Includes 2–3 worked examples in the prompt. Improves format consistency and domain calibration. Best for technical and structured tasks. |
| `structured_output` | Instructs the model to respond in a defined schema (JSON, bullet points, table). Correct for enterprise and API consumers who parse the response downstream. |

---

## Context Vector (`feature_dim = 5`)

| Index | Feature | Range | How to compute |
|-------|---------|-------|----------------|
| 0 | `task_complexity` | `[0.0, 1.0]` | Score from a lightweight classifier on the input: simple Q&A `0.1`, summarisation `0.3`, analysis `0.6`, multi-step reasoning `0.9` |
| 1 | `domain` | `[0.0, 1.0]` | Domain type encoded as float: conversational `0.0`, creative `0.2`, factual `0.4`, technical `0.7`, mathematical `1.0` |
| 2 | `input_length_norm` | `[0.0, 1.0]` | `input_token_count / context_window_size` (e.g. divide by 4096) |
| 3 | `session_turn_norm` | `[0.0, 1.0]` | `current_turn / 20` — early in a conversation vs late, where prior context has accumulated |
| 4 | `user_expertise` | `[0.0, 1.0]` | Inferred from session history or account data: novice `0.0`, intermediate `0.5`, expert `1.0` |

---

## Reward Design

The reward is a quality score produced by an LLM judge, run asynchronously after the response is served. The judge evaluates the response against the original query and returns a score in `[0, 1]`.

```python
def judge_response(query: str, response: str) -> float:
    """Score a response using an LLM judge. Returns 0.0–1.0."""
    result = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                f"Rate this response to the query on a scale of 0 to 10.\n\n"
                f"Query: {query}\n\n"
                f"Response: {response}\n\n"
                f"Consider accuracy, clarity, and usefulness. "
                f"Reply with only a number between 0 and 10."
            )
        }]
    )
    score = float(result.choices[0].message.content.strip())
    return round(score / 10.0, 2)  # normalise to [0, 1]
```

Because the judge runs after the response is already served to the user, BanditDB holds the context in its TTL cache until the score arrives. This is the same delayed-reward pattern used in clinical trials and sleep improvement — the prediction and the reward are decoupled in time.

Human feedback can optionally override or blend with the judge score:

```
reward = judge_score                        # automated path (always available)
reward = 1.0  if thumbs up                 # human override — strong positive signal
reward = 0.0  if user regenerates          # human override — strong negative signal
```

---

## Code

```python
from banditdb import Client
import openai

db  = Client("http://localhost:8080", api_key="your-secret-key")
llm = openai.OpenAI()

PROMPT_VARIANTS = {
    "zero_shot": (
        "Answer the user's question directly and concisely."
    ),
    "chain_of_thought": (
        "Think through the problem step by step before giving your final answer. "
        "Show your reasoning, then state your conclusion clearly."
    ),
    "few_shot": (
        "Here are two examples of high-quality responses:\n\n"
        "Q: What is the time complexity of binary search?\n"
        "A: O(log n) — each step halves the search space.\n\n"
        "Q: When should you use a hash map over a list?\n"
        "A: When you need O(1) lookups by key; lists require O(n) search.\n\n"
        "Now answer the following question in the same style."
    ),
    "structured_output": (
        "Respond in valid JSON with the following fields: "
        '{"answer": "...", "confidence": 0.0–1.0, "sources": []}. '
        "Do not include any text outside the JSON object."
    ),
}

# ── 1. Create the campaign (run once at startup) ──────────────────────────────
db.create_campaign(
    campaign_id="prompt_strategy",
    arms=["zero_shot", "chain_of_thought", "few_shot", "structured_output"],
    feature_dim=5,
)

# ── 2. A request arrives. Build the context vector from request metadata. ──────
# Context: [task_complexity, domain, input_length_norm, session_turn_norm, user_expertise]
context = [0.80, 0.75, 0.35, 0.10, 0.50]
#           ^      ^      ^     ^     ^
#           high   tech   mid   early  intermediate

# ── 3. Ask BanditDB which prompt strategy to use ──────────────────────────────
strategy, interaction_id = db.predict("prompt_strategy", context)

# ── 4. Apply the chosen strategy and serve the response ───────────────────────
response = llm.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": PROMPT_VARIANTS[strategy]},
        {"role": "user",   "content": user_query},
    ]
)
answer = response.choices[0].message.content

# ── 5. Judge asynchronously and close the feedback loop ───────────────────────
# Run in a background task — do not block the user response on this
judge_score = judge_response(user_query, answer)  # → e.g. 0.83
db.reward(interaction_id, judge_score)
```

---

## What the Model Learns

After enough production traffic with judge scores recorded, the model converges on prompt-task pairings that experienced prompt engineers know intuitively but can rarely systematise across an entire organisation:

| Task Profile | Context | Arm that emerges | Why the bandit learns this |
|---|---|---|---|
| Simple conversational query | `[0.1, 0.0, 0.1, 0.1, 0.3]` | `zero_shot` | Low complexity, conversational domain — extra scaffolding hurts more than it helps. Chain-of-thought adds verbose reasoning the user did not ask for. |
| Complex mathematical reasoning | `[0.9, 1.0, 0.4, 0.1, 0.7]` | `chain_of_thought` | High complexity, mathematical domain — step-by-step reasoning consistently raises judge scores by surfacing intermediate errors before they propagate. |
| Technical question, expert user | `[0.7, 0.75, 0.5, 0.5, 1.0]` | `few_shot` | Expert users in technical domains expect format-consistent, convention-aligned answers. Few-shot examples calibrate tone and depth more reliably than instructions alone. |
| API consumer, any task | `[*, *, *, *, *]` with downstream parsing | `structured_output` | When the caller parses the response programmatically, unstructured prose causes failures. The bandit learns to identify these sessions from context signals. |
| Late in a long conversation | `[0.5, 0.4, 0.6, 0.9, 0.5]` | `few_shot` | Late turns have accumulated context. Few-shot examples help maintain stylistic consistency with earlier exchanges that zero-shot and chain-of-thought do not preserve. |

The model also surfaces interactions that are hard to pre-specify: chain-of-thought hurts on simple tasks even when the domain is technical; few-shot loses to zero-shot for novice users who find the examples distracting rather than clarifying.

---

## Why This Is Different From Offline Evals

| Offline benchmark | BanditDB in production |
|---|---|
| Fixed test set | Real user traffic |
| One winner deployed to all users | Context-dependent routing |
| Run once, revisit manually | Continuous — learns every request |
| Measures average performance | Measures performance per context |
| Human or LLM judge runs offline | Judge runs asynchronously in production |
| Stops learning after deployment | Never stops learning |

---

## Convergence Estimate

**~400 production requests** — assuming the judge runs on every interaction (giving ~400 continuous reward signals) — to measurably outperform random prompt selection.

| Phase | Requests | What the model knows |
|---|---|---|
| Exploration | 0–100 | Strategies selected roughly equally. No meaningful pattern yet. |
| Early signal | 100–250 | Chain-of-thought advantage on high-complexity tasks begins to emerge. Zero-shot preference for conversational queries appears. |
| Measurable lift | ~400 | Cumulative judge score from BanditDB-routed requests exceeds random prompt selection. |
| Stable routing | ~800 | All four strategies are reliably triggered by appropriate context combinations. User expertise × domain interaction effects become visible. |

**Assumptions:** The judge runs asynchronously on every interaction within the TTL window. This gives near-100% reward observability — the best condition of all examples. Set `BANDITDB_REWARD_TTL_SECS` to cover the judge's maximum latency plus a safety buffer (e.g. `300` for a 5-minute window).

If human feedback is the primary reward signal instead of an automated judge, expect reward observability to drop to 5–15% (only users who actively give feedback). In that case, target ~3,000 requests to collect 300 reward signals.

---

## Related Examples

- [`01_dynamic_pricing_sell_through.md`](./01_dynamic_pricing_sell_through.md) — continuous reward with near-100% observability (same convergence profile)
- [`04_adaptive_clinical_trials.md`](./04_adaptive_clinical_trials.md) — delayed rewards held in TTL cache until outcome arrives
