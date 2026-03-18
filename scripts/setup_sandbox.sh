#!/usr/bin/env sh
# BanditDB Sandbox Setup
# Run once on the Lightsail instance after banditdb.service is running.
# Creates the three demo campaigns used in the README sandbox examples.
#
# Usage:
#   BANDITDB_API_KEY=your-secret-key sh scripts/setup_sandbox.sh
#   BANDITDB_API_KEY=your-secret-key BANDITDB_URL=http://localhost:8080 sh scripts/setup_sandbox.sh

set -e

URL="${BANDITDB_URL:-http://localhost:8080}"
KEY="${BANDITDB_API_KEY:-}"

if [ -z "$KEY" ]; then
  echo "Error: BANDITDB_API_KEY is not set." >&2
  exit 1
fi

H="X-Api-Key: $KEY"

echo "Creating demo campaigns on $URL..."

# ── 1. Sleep improvement ──────────────────────────────────────────────────────
# Context: [sex, age/100, weight_kg/150, activity_0-1, bedtime_hour/24]
curl -sf -X POST "$URL/campaign" \
  -H "Content-Type: application/json" \
  -H "$H" \
  -d '{
    "campaign_id": "sleep",
    "arms": ["decrease_temperature", "decrease_light", "decrease_noise"],
    "feature_dim": 5
  }' > /dev/null
echo "  ✓ sleep"

# ── 2. Prompt strategy (LLM evals) ───────────────────────────────────────────
# Context: [task_complexity, domain, input_length_norm, session_turn_norm, user_expertise]
curl -sf -X POST "$URL/campaign" \
  -H "Content-Type: application/json" \
  -H "$H" \
  -d '{
    "campaign_id": "prompt_strategy",
    "arms": ["zero_shot", "chain_of_thought", "few_shot", "structured_output"],
    "feature_dim": 5
  }' > /dev/null
echo "  ✓ prompt_strategy"

# ── 3. Client intake routing ──────────────────────────────────────────────────
# Context: [case_value_norm, matter_complexity, org_size_norm, conflict_risk, capacity_norm]
curl -sf -X POST "$URL/campaign" \
  -H "Content-Type: application/json" \
  -H "$H" \
  -d '{
    "campaign_id": "client_intake",
    "arms": ["schedule_consultation", "send_intake_form", "refer_to_partner_firm", "decline"],
    "feature_dim": 5
  }' > /dev/null
echo "  ✓ client_intake"

echo ""
echo "All demo campaigns ready. Verify:"
curl -s "$URL/campaigns" -H "$H"
echo ""
