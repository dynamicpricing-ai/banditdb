"""
Demo runner — BanditDB Call Centre Agent Swarm
==============================================
Runs a series of customer service interactions through the multi-agent swarm.
After each interaction, BanditDB's shared model is updated with the quality score.
At the end, diagnostics show which strategy the model has converged towards.

Usage:
    export GOOGLE_API_KEY=your-google-api-key
    python main.py
"""

import asyncio
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agent import root_agent, get_campaign_diagnostics

APP_NAME = "banditdb_call_centre"

# ── Demo queries — a realistic spread across domains and complexity ────────────

DEMO_QUERIES = [
    # Billing — moderate complexity
    "I was charged twice for my plan this month. I want one of the charges refunded.",

    # Technical — high complexity
    "My internet has been dropping every evening for the past week. "
    "I work from home and this is costing me money.",

    # Account — simple
    "How do I update my payment method online?",

    # Complaint — high complexity
    "I've been a loyal customer for 8 years and you're offering new customers "
    "the same plan I pay £45 for at £29. That feels disrespectful.",

    # Technical — medium, expert vocabulary
    "My router shows all LEDs green but I'm getting packet loss above 15% "
    "on every ping to 8.8.8.8. MTU issue or something on your end?",

    # General — simple
    "What's the difference between your 100 Mbps and 500 Mbps plans "
    "for a household of four with two people gaming?",

    # Account — medium complexity
    "I want to upgrade my plan but your website keeps throwing a 500 error "
    "every time I click confirm.",
]


async def run_interaction(
    runner: Runner,
    session_id: str,
    user_id: str,
    query: str,
    turn: int,
) -> None:
    print(f"\n{'━' * 62}")
    print(f"  Query {turn}: {query[:80]}{'…' if len(query) > 80 else ''}")
    print(f"{'━' * 62}")

    message = types.Content(role="user", parts=[types.Part(text=query)])

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=message,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            print(event.content.parts[0].text)


async def show_diagnostics(runner: Runner, session_id: str, user_id: str) -> None:
    print(f"\n{'━' * 62}")
    print("  What BanditDB learned")
    print(f"{'━' * 62}")

    prompt = (
        "Call get_campaign_diagnostics and then summarise the results in a "
        "readable table showing each arm's prediction count, reward count, "
        "and theta norm. Note which arm the model currently favours and why "
        "that might make sense for call centre interactions."
    )

    message = types.Content(role="user", parts=[types.Part(text=prompt)])

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=message,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            print(event.content.parts[0].text)


async def main() -> None:
    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    session = session_service.create_session(app_name=APP_NAME, user_id="demo")
    user_id = session.user_id
    session_id = session.id

    print()
    print("  BanditDB · Call Centre Agent Swarm")
    print("  Google ADK — root_agent + context_agent + csr_agent + evaluator_agent")
    print("  Campaign : prompt_strategy  (sandbox.banditdb.com)")
    print("  Arms     : zero_shot | chain_of_thought | few_shot | structured_output")
    print()

    for turn, query in enumerate(DEMO_QUERIES, 1):
        await run_interaction(runner, session_id, user_id, query, turn)

    await show_diagnostics(runner, session_id, user_id)

    print(f"\n{'━' * 62}")
    print("  Done. Rewards recorded to sandbox.banditdb.com/campaign/prompt_strategy")
    print(f"{'━' * 62}\n")


if __name__ == "__main__":
    asyncio.run(main())
