---
description: Operate BanditDB — the Intuition Database. Create campaigns, get intuition, record outcomes, diagnose learning.
---

You are operating BanditDB — the Intuition Database. An in-memory decision database for agents and applications that need to learn which decisions work and get better with every outcome.

## Setup check
First, call `mcp__banditdb__list_campaigns` to verify the connection.
If MCP tools are unavailable, tell the user to:
1. Ensure BanditDB is running: `docker start banditdb` or `banditdb`
2. Ensure banditdb-mcp is installed: `pip install banditdb-mcp`
3. Restart Claude Code so the MCP server loads

## Available tools
- `mcp__banditdb__create_campaign`      — define decision space (arms, feature_dim, algorithm)
- `mcp__banditdb__get_intuition`        — context-aware arm recommendation
- `mcp__banditdb__batch_get_intuition`  — batch predictions (up to 100)
- `mcp__banditdb__record_outcome`       — submit reward signal
- `mcp__banditdb__campaign_report`      — view campaign performance
- `mcp__banditdb__campaign_diagnostics` — inspect model convergence and health
- `mcp__banditdb__list_campaigns`       — list all campaigns
- `mcp__banditdb__archive_campaign`     — soft-delete a campaign
- `mcp__banditdb__restore_campaign`     — restore archived campaign

## Algorithms
- `linucb`      — contextual LinUCB (default; supports propensity export for causal analysis)
- `thompson`    — Thompson Sampling
- `neural`      — NeuralLinUCB (MLP feature layer, retrains at checkpoint)
- `tournament`  — Progressive Tournament (base vs challenger, SNIPS evaluation)

## Reward design
Rewards are floats, typically in [0, 1]. Normalise continuous outcomes before reporting.
Delayed rewards (next-day outcomes, post-checkout conversions) are supported via the TTL cache —
record the outcome whenever it arrives, up to BANDITDB_REWARD_TTL_SECS after prediction.

## Context vectors
Context must be a fixed-length float array matching the campaign's `feature_dim`.
Normalise all features to roughly [0, 1] for best LinUCB performance.
