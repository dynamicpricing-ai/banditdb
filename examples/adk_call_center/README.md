# Call Centre Agent Swarm — BanditDB + Google ADK

A multi-agent system that learns which prompt strategy produces the best customer
service responses for each type of enquiry. BanditDB acts as the shared decision
layer: every agent consults the same model, and every scored response feeds the
reward back so the whole swarm improves together.

## What this example shows

Standard LLM agents are stateless — if `chain_of_thought` works poorly for billing
complaints today, the agent will try it again tomorrow. BanditDB fixes this by
accumulating judgment across every interaction.

After ~50–100 rewarded interactions you will see the model start routing different
query types to different prompt strategies based on what actually worked:

- **Billing disputes** → may converge on `structured_output` (clear, step-by-step resolution)
- **Technical issues** → may converge on `chain_of_thought` (systematic diagnosis)
- **Simple FAQs** → may converge on `zero_shot` (fast, direct answer)
- **Angry customers** → may converge on `few_shot` (follows empathetic examples)

## Architecture

```
root_agent  (orchestrator)
├── context_agent     extracts 5D context vector from the customer query
├── csr_agent         generates response using the BanditDB-selected template
└── evaluator_agent   scores response quality → reward for BanditDB
```

### BanditDB campaign

| Field | Value |
|---|---|
| Campaign ID | `prompt_strategy` |
| Arms | `zero_shot`, `chain_of_thought`, `few_shot`, `structured_output` |
| Feature dim | 5 |
| Algorithm | LinUCB |
| Sandbox | `https://sandbox.banditdb.com` |

### Context vector

| Index | Feature | Range |
|---|---|---|
| 0 | `task_complexity` | 0 = simple FAQ, 1 = complex multi-step |
| 1 | `domain_code` | 0=billing, 0.25=technical, 0.5=account, 0.75=complaint, 1.0=general |
| 2 | `input_length_norm` | `len(query) / 500`, capped at 1.0 |
| 3 | `session_turn_norm` | `turn / 10`, capped at 1.0 |
| 4 | `user_expertise` | 0 = everyday language, 1 = technical vocabulary |

### Reward signal

The `evaluator_agent` acts as an LLM-as-judge, scoring the CSR response on
empathy, accuracy, clarity, resolution, and conciseness. The score (0.0–1.0)
is reported to BanditDB via `record_response_quality`. This is the signal that
drives learning.

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Google API key (for Gemini)
export GOOGLE_API_KEY=your-google-api-key

# 3. Optional — point at your own BanditDB instance
export BANDITDB_URL=http://localhost:8080
export BANDITDB_API_KEY=your-api-key
# Default: sandbox.banditdb.com with the demo key
```

## Run

```bash
python main.py
```

Expected output for each interaction:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Query 1: I was charged twice for my plan this month…
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Strategy : structured_output
  Response : I'm sorry for the double charge — that should not have
             happened. I've raised a refund for the duplicate payment
             and you'll see it back within 3–5 business days. Is there
             anything else I can help you with today?
  Score    : 0.91 — empathetic, specific action, clear timeline
  Reward   : recorded ✓
```

After all interactions, a diagnostics summary shows which arms the model has
explored, how many rewards each received, and which strategy it currently favours.

## How BanditDB drives the learning

```
context_agent                    BanditDB
─────────────  ──get_intuition──►  ──────────────────────────
 extracts 5D    ◄─"chain_of_thought"─  prompt_strategy campaign
 context vector                    LinUCB learns per-context arm

csr_agent + evaluator_agent
─────────────────────────────────────────────────────────────
 generate response → score → record_outcome(iid, 0.84)
 reward updates model for ALL agents in the swarm
```

After enough interactions, BanditDB will start routing different task types to
different strategies based on empirical quality, not guesswork.

## File structure

```
adk_call_center/
├── agent.py          Agent definitions, prompt templates, BanditDB tools
├── main.py           Demo runner — 7 sample queries + diagnostics
├── requirements.txt
└── README.md
```
