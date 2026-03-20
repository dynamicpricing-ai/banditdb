#!/usr/bin/env python3
"""
Seed the BanditDB sandbox with realistic predict→reward cycles.
Simulates three distinct user populations per campaign so the model
learns meaningful arm preferences.

Usage:
    python scripts/seed_sandbox.py
    BANDITDB_URL=https://sandbox.banditdb.com python scripts/seed_sandbox.py
"""

import os
import random
import time
import urllib.request
import urllib.error
import json

URL = os.environ.get("BANDITDB_URL", "https://sandbox.banditdb.com")
KEY = os.environ.get("BANDITDB_API_KEY", "banditdb-demo")
HEADERS = {"Content-Type": "application/json", "X-Api-Key": KEY}


def post(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{URL}{path}", data=data, headers=HEADERS, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def reward(interaction_id, r):
    body = {"interaction_id": interaction_id, "reward": r}
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{URL}/reward", data=data, headers=HEADERS, method="POST")
    with urllib.request.urlopen(req, timeout=10):
        pass


def run_cycle(campaign_id, context, true_rewards):
    """Predict, then reward based on a noisy ground-truth reward function."""
    resp = post("/predict", {"campaign_id": campaign_id, "context": context})
    arm = resp["arm_id"]
    iid = resp["interaction_id"]
    base = true_rewards.get(arm, 0.3)
    r = max(0.0, min(1.0, base + random.gauss(0, 0.1)))
    time.sleep(0.02)  # simulate delayed reward
    reward(iid, round(r, 3))
    return arm


# ── Sleep improvement ─────────────────────────────────────────────────────────
# Context: [sex, age/100, weight_kg/150, activity_0-1, bedtime_hour/24]
# Ground truth: older/heavier users respond to temperature; young/active to noise reduction

def seed_sleep(n=300):
    print(f"  Seeding sleep ({n} cycles)...")
    profiles = [
        # young active female — light reduction works best
        {"ctx": [1.0, 0.25, 0.45, 0.85, 0.92], "rewards": {"decrease_temperature": 0.35, "decrease_light": 0.78, "decrease_noise": 0.50}},
        # middle-aged male, overweight — temperature reduction works best
        {"ctx": [0.0, 0.45, 0.80, 0.30, 0.96], "rewards": {"decrease_temperature": 0.80, "decrease_light": 0.35, "decrease_noise": 0.40}},
        # elderly sedentary female — noise reduction works best
        {"ctx": [1.0, 0.70, 0.55, 0.15, 0.88], "rewards": {"decrease_temperature": 0.40, "decrease_light": 0.45, "decrease_noise": 0.82}},
        # young male athlete — light reduction works best
        {"ctx": [0.0, 0.28, 0.75, 0.95, 0.94], "rewards": {"decrease_temperature": 0.30, "decrease_light": 0.72, "decrease_noise": 0.48}},
    ]
    counts = {}
    for i in range(n):
        p = profiles[i % len(profiles)]
        ctx = [v + random.gauss(0, 0.02) for v in p["ctx"]]
        ctx = [max(0.0, min(1.0, v)) for v in ctx]
        arm = run_cycle("sleep", ctx, p["rewards"])
        counts[arm] = counts.get(arm, 0) + 1
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{n} — arm distribution: {counts}")
    print(f"  ✓ sleep done. Final distribution: {counts}")


# ── Prompt strategy ───────────────────────────────────────────────────────────
# Context: [task_complexity, domain, input_length_norm, session_turn_norm, user_expertise]
# Ground truth: complex tasks → CoT; simple/expert → zero_shot; structured → structured_output

def seed_prompt_strategy(n=300):
    print(f"  Seeding prompt_strategy ({n} cycles)...")
    profiles = [
        # complex task, low expertise — chain of thought works best
        {"ctx": [0.90, 0.60, 0.70, 0.20, 0.20], "rewards": {"zero_shot": 0.30, "chain_of_thought": 0.85, "few_shot": 0.55, "structured_output": 0.40}},
        # simple task, high expertise — zero shot works best
        {"ctx": [0.15, 0.40, 0.25, 0.80, 0.90], "rewards": {"zero_shot": 0.82, "chain_of_thought": 0.50, "few_shot": 0.55, "structured_output": 0.45}},
        # structured domain (code/data), long input — structured output works best
        {"ctx": [0.60, 0.85, 0.90, 0.50, 0.60], "rewards": {"zero_shot": 0.35, "chain_of_thought": 0.55, "few_shot": 0.50, "structured_output": 0.88}},
        # medium complexity, mid expertise — few shot works best
        {"ctx": [0.50, 0.50, 0.50, 0.50, 0.50], "rewards": {"zero_shot": 0.45, "chain_of_thought": 0.60, "few_shot": 0.78, "structured_output": 0.52}},
    ]
    counts = {}
    for i in range(n):
        p = profiles[i % len(profiles)]
        ctx = [v + random.gauss(0, 0.02) for v in p["ctx"]]
        ctx = [max(0.0, min(1.0, v)) for v in ctx]
        arm = run_cycle("prompt_strategy", ctx, p["rewards"])
        counts[arm] = counts.get(arm, 0) + 1
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{n} — arm distribution: {counts}")
    print(f"  ✓ prompt_strategy done. Final distribution: {counts}")


# ── Client intake ─────────────────────────────────────────────────────────────
# Context: [case_value_norm, matter_complexity, org_size_norm, conflict_risk, capacity_norm]
# Ground truth: high value + complex → consultation; high conflict → decline/refer; small org → intake form

def seed_client_intake(n=300):
    print(f"  Seeding client_intake ({n} cycles)...")
    profiles = [
        # high value, complex, large org, low conflict — schedule consultation
        {"ctx": [0.90, 0.85, 0.80, 0.10, 0.70], "rewards": {"schedule_consultation": 0.88, "send_intake_form": 0.40, "refer_to_partner_firm": 0.35, "decline": 0.05}},
        # low value, simple, small org — send intake form
        {"ctx": [0.20, 0.25, 0.15, 0.20, 0.80], "rewards": {"schedule_consultation": 0.40, "send_intake_form": 0.82, "refer_to_partner_firm": 0.35, "decline": 0.10}},
        # medium value, high conflict risk — refer to partner
        {"ctx": [0.50, 0.60, 0.40, 0.85, 0.50], "rewards": {"schedule_consultation": 0.30, "send_intake_form": 0.25, "refer_to_partner_firm": 0.80, "decline": 0.40}},
        # low value, high conflict, low capacity — decline
        {"ctx": [0.10, 0.70, 0.20, 0.90, 0.05], "rewards": {"schedule_consultation": 0.10, "send_intake_form": 0.15, "refer_to_partner_firm": 0.45, "decline": 0.85}},
    ]
    counts = {}
    for i in range(n):
        p = profiles[i % len(profiles)]
        ctx = [v + random.gauss(0, 0.02) for v in p["ctx"]]
        ctx = [max(0.0, min(1.0, v)) for v in ctx]
        arm = run_cycle("client_intake", ctx, p["rewards"])
        counts[arm] = counts.get(arm, 0) + 1
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{n} — arm distribution: {counts}")
    print(f"  ✓ client_intake done. Final distribution: {counts}")


if __name__ == "__main__":
    random.seed(42)
    print(f"Seeding sandbox at {URL}...\n")
    seed_sleep(750)
    print()
    seed_prompt_strategy(300)
    print()
    seed_client_intake(300)
    print("\nDone. Run a checkpoint to export Parquet:")
    print(f'  curl -s -X POST {URL}/checkpoint -H "X-Api-Key: {KEY}"')
