"""
Call Centre Agent Swarm — powered by BanditDB
==============================================
A multi-agent system built with Google ADK that learns which prompt strategy
works best for each type of customer service interaction.

BanditDB acts as shared infrastructure: every agent in the swarm consults the
same learned model, and every scored response feeds the reward back so the whole
swarm gets smarter together.

Architecture
------------
  root_agent  (orchestrator)
  ├── context_agent    extracts a 5D context vector from the customer query
  ├── csr_agent        generates the response using the BanditDB-selected template
  └── evaluator_agent  scores the response quality (LLM-as-judge → reward)

BanditDB campaign: prompt_strategy
Arms: zero_shot | chain_of_thought | few_shot | structured_output
Context: [task_complexity, domain_code, input_length_norm, session_turn_norm, user_expertise]
Sandbox: https://sandbox.banditdb.com  (API key: banditdb-demo)
"""

import os
import httpx
from google.adk.agents import Agent

# ── Configuration ─────────────────────────────────────────────────────────────

BANDITDB_URL = os.getenv("BANDITDB_URL", "https://sandbox.banditdb.com")
BANDITDB_API_KEY = os.getenv("BANDITDB_API_KEY", "banditdb-demo")
CAMPAIGN_ID = "prompt_strategy"

_headers = {
    "Content-Type": "application/json",
    "X-Api-Key": BANDITDB_API_KEY,
}

# ── Prompt templates (one per arm) ────────────────────────────────────────────

PROMPT_TEMPLATES = {
    "zero_shot": (
        "You are a professional call centre agent for a telecom company. "
        "Be empathetic, concise, and solution-focused.\n\n"
        "Customer: {query}\n\nAgent:"
    ),
    "chain_of_thought": (
        "You are a professional call centre agent for a telecom company.\n"
        "Work through this step by step before replying:\n"
        "  1. What is the customer's core issue?\n"
        "  2. What information or action resolves it?\n"
        "  3. How do you communicate this empathetically?\n\n"
        "Customer: {query}\n\nAgent:"
    ),
    "few_shot": (
        "You are a professional call centre agent for a telecom company.\n\n"
        "Example 1\n"
        "Customer: My bill is £20 higher than usual.\n"
        "Agent: I'm sorry for the confusion. I can see a one-off add-on was activated "
        "on the 3rd — I'll remove that charge and apply a credit within 24 hours.\n\n"
        "Example 2\n"
        "Customer: My broadband drops every evening.\n"
        "Agent: I apologise — that's disruptive. I can see congestion on your local "
        "exchange between 6–9 pm. I'm flagging this to our network team and will send "
        "you updates by text. Restarting your router at 5:45 pm may help in the meantime.\n\n"
        "Customer: {query}\n\nAgent:"
    ),
    "structured_output": (
        "You are a professional call centre agent for a telecom company.\n"
        "Structure your response using these four elements (no headings, just natural flow):\n"
        "  • Acknowledge  — validate the customer's situation\n"
        "  • Explain      — provide the key facts or diagnosis\n"
        "  • Resolve      — state the action taken or next steps\n"
        "  • Confirm      — check the customer is satisfied\n\n"
        "Customer: {query}\n\nAgent:"
    ),
}


# ── BanditDB tools ─────────────────────────────────────────────────────────────

def ensure_campaign() -> dict:
    """
    Create the prompt_strategy campaign on BanditDB if it does not already exist.
    Safe to call at session start — returns 'created' or 'exists' without error.
    """
    with httpx.Client() as client:
        r = client.post(
            f"{BANDITDB_URL}/campaign",
            headers=_headers,
            json={
                "campaign_id": CAMPAIGN_ID,
                "arms": list(PROMPT_TEMPLATES.keys()),
                "feature_dim": 5,
                "algorithm": "linucb",
                "alpha": 1.0,
            },
            timeout=10,
        )
    if r.status_code == 200:
        return {"status": "created", "campaign_id": CAMPAIGN_ID}
    # Non-200 on sandbox means it already exists — that is fine
    return {"status": "exists", "campaign_id": CAMPAIGN_ID}


def get_prompt_strategy(
    task_complexity: float,
    domain_code: float,
    input_length_norm: float,
    session_turn_norm: float,
    user_expertise: float,
) -> dict:
    """
    Ask BanditDB which prompt strategy to use for this customer interaction.
    Returns the strategy name, the filled prompt template, and an interaction_id
    that must be passed to record_response_quality once the response is scored.

    Args:
        task_complexity:   0.0 = simple FAQ, 1.0 = complex multi-step problem
        domain_code:       0.0=billing, 0.25=technical, 0.5=account, 0.75=complaint, 1.0=general
        input_length_norm: len(customer_message) / 500, capped at 1.0
        session_turn_norm: current turn number / 10, capped at 1.0
        user_expertise:    0.0 = everyday language, 1.0 = technical vocabulary detected
    """
    context = [
        task_complexity,
        domain_code,
        input_length_norm,
        session_turn_norm,
        user_expertise,
    ]
    with httpx.Client() as client:
        r = client.post(
            f"{BANDITDB_URL}/predict",
            headers=_headers,
            json={"campaign_id": CAMPAIGN_ID, "context": context},
            timeout=10,
        )
    r.raise_for_status()
    data = r.json()
    arm = data["arm_id"]
    return {
        "strategy": arm,
        "interaction_id": data["interaction_id"],
        "prompt_template": PROMPT_TEMPLATES[arm],
    }


def record_response_quality(interaction_id: str, quality_score: float) -> dict:
    """
    Report the quality of the CSR response back to BanditDB.
    This reward updates the shared model so future interactions route better
    for all agents in the swarm.

    Args:
        interaction_id: the ID returned by get_prompt_strategy for this interaction
        quality_score:  evaluator score — 0.0 (poor) to 1.0 (excellent)
    """
    with httpx.Client() as client:
        r = client.post(
            f"{BANDITDB_URL}/reward",
            headers=_headers,
            json={
                "interaction_id": interaction_id,
                "reward": round(float(quality_score), 4),
            },
            timeout=10,
        )
    r.raise_for_status()
    return {
        "status": "recorded",
        "interaction_id": interaction_id,
        "reward": quality_score,
    }


def get_campaign_diagnostics() -> dict:
    """
    Retrieve the current learning state of the prompt_strategy campaign.
    Shows per-arm prediction counts, reward counts, and theta norms —
    useful for understanding which strategy the model currently favours.
    """
    with httpx.Client() as client:
        r = client.get(
            f"{BANDITDB_URL}/campaign/{CAMPAIGN_ID}",
            headers=_headers,
            timeout=10,
        )
    if r.status_code == 404:
        return {"status": "campaign_not_found"}
    r.raise_for_status()
    return r.json()


# ── Sub-agents ─────────────────────────────────────────────────────────────────

context_agent = Agent(
    name="context_agent",
    model="gemini-2.0-flash",
    description=(
        "Analyses a customer service query and returns a 5-dimensional numeric "
        "context vector that BanditDB uses to select the best prompt strategy."
    ),
    instruction="""
You extract a structured context vector from a customer service query.

Return ONLY a valid JSON object with exactly these five float fields — no markdown,
no explanation, nothing else:

{
  "task_complexity":   <0.0–1.0>,
  "domain_code":       <0.0–1.0>,
  "input_length_norm": <0.0–1.0>,
  "session_turn_norm": <0.0–1.0>,
  "user_expertise":    <0.0–1.0>
}

Scoring guide:
  task_complexity   0 = single-answer question, 1 = multi-step problem requiring investigation
  domain_code       0.0=billing  0.25=technical  0.5=account management  0.75=complaint  1.0=general
  input_length_norm character count of query divided by 500, capped at 1.0
  session_turn_norm default to 0.1 (first turn) unless a turn number is provided
  user_expertise    0.0 = everyday language, 1.0 = network/technical jargon detected
""",
)


csr_agent = Agent(
    name="csr_agent",
    model="gemini-2.0-flash",
    description=(
        "Generates a professional customer service response using the prompt "
        "template selected by BanditDB. Behaves as a trained telecom call centre agent."
    ),
    instruction="""
You are a professional call centre agent for a telecom company.

You will receive a prompt that already contains the customer's message and a
response structure chosen by BanditDB. Follow that structure and generate a
warm, professional, empathetic response.

Keep it concise (2–4 sentences) unless the complexity demands more.
Never add meta-commentary. Produce only the agent response text.
""",
)


evaluator_agent = Agent(
    name="evaluator_agent",
    model="gemini-2.0-flash",
    description=(
        "Scores a customer service response on a 0.0–1.0 quality scale "
        "for use as a reward signal in BanditDB."
    ),
    instruction="""
You are a quality assurance specialist for a telecom call centre.

You will receive a customer query and the agent response. Score the response on
a 0.0–1.0 scale considering:
  • Empathy    — does it acknowledge the customer's situation?
  • Accuracy   — does it address the actual issue raised?
  • Clarity    — is it easy to understand?
  • Resolution — does it offer a concrete action or next step?
  • Conciseness — appropriately brief without being dismissive?

Return ONLY a valid JSON object — no markdown, no explanation:
{"score": <0.00–1.00>, "reason": "<one short sentence>"}
""",
)


# ── Root agent (orchestrator) ──────────────────────────────────────────────────

root_agent = Agent(
    name="call_centre_orchestrator",
    model="gemini-2.0-flash",
    description=(
        "Orchestrates the AI call centre swarm. Routes each customer query through "
        "BanditDB-powered prompt selection, generates a response, evaluates quality, "
        "and feeds the reward back so the shared model improves with every interaction."
    ),
    instruction="""
You orchestrate an AI call centre that uses BanditDB to learn which prompt strategy
works best for each type of customer interaction. Every reward you record improves
the routing for the entire agent swarm.

Follow these steps for every customer query:

STEP 1 — SETUP
Call ensure_campaign to confirm the BanditDB campaign is ready.

STEP 2 — CONTEXT EXTRACTION
Transfer to context_agent. Pass the customer query and ask it to return the
5-dimensional context vector as JSON.
Extract the five float values from its response.

STEP 3 — STRATEGY SELECTION
Call get_prompt_strategy with the five context values.
Note the returned strategy name, interaction_id, and prompt_template.

STEP 4 — RESPONSE GENERATION
Transfer to csr_agent. Provide the prompt_template with {query} replaced by
the actual customer query. Ask it to generate the agent response.

STEP 5 — QUALITY EVALUATION
Transfer to evaluator_agent. Provide:
  - The original customer query
  - The response from csr_agent
Ask it to return a JSON with "score" (float) and "reason" (string).

STEP 6 — REWARD
Call record_response_quality with the interaction_id from Step 3
and the score from Step 5.

STEP 7 — SUMMARY
Reply with a clean summary:

  Strategy : <name>
  Response : <csr response>
  Score    : <X.XX> — <reason>
  Reward   : recorded ✓
""",
    tools=[
        ensure_campaign,
        get_prompt_strategy,
        record_response_quality,
        get_campaign_diagnostics,
    ],
    sub_agents=[context_agent, csr_agent, evaluator_agent],
)
