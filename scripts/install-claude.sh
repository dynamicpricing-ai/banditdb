#!/usr/bin/env bash
# BanditDB × Claude Code — full setup
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/dynamicpricing-ai/banditdb/main/scripts/install-claude.sh | bash
#
# Environment overrides (all optional):
#   BANDITDB_URL=http://localhost:8080
#   BANDITDB_PORT=8080
#   BANDITDB_API_KEY=your-key
#   BANDITDB_INSTALL_DIR=/usr/local/bin
#   SKIP_BANDITDB=1     # skip binary/docker install if already running

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
if [ -t 1 ]; then
  GRN='\033[0;32m' BLU='\033[0;34m' YLW='\033[1;33m' RED='\033[0;31m' RST='\033[0m'
else
  GRN='' BLU='' YLW='' RED='' RST=''
fi
ok()   { printf "${GRN}✓${RST}  %s\n" "$*"; }
info() { printf "${BLU}→${RST}  %s\n" "$*"; }
warn() { printf "${YLW}!${RST}  %s\n" "$*"; }
die()  { printf "${RED}✗${RST}  %s\n" "$*" >&2; exit 1; }
step() { printf "\n${BLU}── %s${RST}\n" "$*"; }

BANDITDB_URL="${BANDITDB_URL:-http://localhost:8080}"
BANDITDB_PORT="${BANDITDB_PORT:-8080}"
BANDITDB_API_KEY="${BANDITDB_API_KEY:-}"

printf "\n"
printf "  BanditDB × Claude Code — Setup\n"
printf "  ════════════════════════════════\n\n"

# ── 1. BanditDB binary / server ───────────────────────────────────────────────
step "BanditDB server"

if [ "${SKIP_BANDITDB:-0}" = "1" ]; then
  warn "Skipping BanditDB install (SKIP_BANDITDB=1)"
else
  if command -v banditdb &>/dev/null; then
    ok "BanditDB binary already installed: $(which banditdb)"
  elif docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^banditdb$"; then
    ok "BanditDB running via Docker"
  else
    # Prefer binary install; fall back to Docker
    if command -v curl &>/dev/null && curl -fsSL \
        "https://api.github.com/repos/dynamicpricing-ai/banditdb/releases/latest" \
        2>/dev/null | grep -q '"tag_name"'; then
      info "Installing BanditDB binary..."
      curl -fsSL \
        "https://raw.githubusercontent.com/dynamicpricing-ai/banditdb/main/scripts/install.sh" \
        | BANDITDB_INSTALL_DIR="${BANDITDB_INSTALL_DIR:-/usr/local/bin}" bash
      ok "BanditDB binary installed"
    elif command -v docker &>/dev/null; then
      info "Starting BanditDB via Docker..."
      docker run -d \
        --name banditdb \
        --restart unless-stopped \
        -p "${BANDITDB_PORT}:8080" \
        simeonlukov/banditdb:latest
      ok "BanditDB container started on port ${BANDITDB_PORT}"
    else
      die "Cannot install BanditDB: need curl+GitHub access or Docker. See https://github.com/dynamicpricing-ai/banditdb/releases"
    fi
  fi
fi

# Start banditdb binary in background if not already listening
if ! curl -sf "${BANDITDB_URL}/health" &>/dev/null; then
  if command -v banditdb &>/dev/null; then
    info "Starting BanditDB server..."
    nohup banditdb > /tmp/banditdb.log 2>&1 &
    sleep 1
  fi
fi

# Wait for server to be ready
info "Waiting for BanditDB at ${BANDITDB_URL}..."
for i in $(seq 1 20); do
  if curl -sf "${BANDITDB_URL}/health" &>/dev/null; then
    ok "BanditDB is ready at ${BANDITDB_URL}"
    break
  fi
  sleep 1
  if [ "$i" = "20" ]; then
    warn "BanditDB not responding after 20s — plugin will still be installed."
    warn "Start BanditDB manually: banditdb   or   docker start banditdb"
  fi
done

# ── 2. MCP server Python package ─────────────────────────────────────────────
step "BanditDB MCP server"

# Resolve pip
PIP=""
for candidate in pip3 pip python3 python; do
  if command -v "$candidate" &>/dev/null; then
    if "$candidate" -m pip --version &>/dev/null 2>&1; then
      PIP="$candidate -m pip"
      break
    fi
  fi
done
[ -z "$PIP" ] && die "Python pip not found. Install Python ≥3.9: https://python.org"

info "Installing banditdb-python..."
$PIP install --quiet --upgrade banditdb-python 2>/dev/null || \
$PIP install --quiet --upgrade --user banditdb-python || \
die "pip install banditdb-python failed. Try manually: pip install banditdb-python"
ok "banditdb-mcp installed"

# ── 3. Claude Code plugin ─────────────────────────────────────────────────────
step "Claude Code plugin"

if command -v claude &>/dev/null; then
  info "Installing banditdb plugin via Claude Code..."
  claude plugin install dynamicpricing-ai/banditdb 2>/dev/null && \
    ok "Plugin installed: dynamicpricing-ai/banditdb" || {
    warn "claude plugin install failed — falling back to manual configuration"
    _MANUAL_INSTALL=1
  }
else
  warn "claude CLI not found in PATH — using manual configuration"
  _MANUAL_INSTALL=1
fi

# Manual fallback: write mcp.json + command file directly
if [ "${_MANUAL_INSTALL:-0}" = "1" ]; then

  CLAUDE_DIR="$HOME/.claude"
  MCP_JSON="$CLAUDE_DIR/mcp.json"
  mkdir -p "$CLAUDE_DIR"

  # Resolve MCP server command
  MCP_CMD=""
  for candidate in banditdb-mcp; do
    if command -v "$candidate" &>/dev/null; then
      MCP_CMD="$(command -v "$candidate")"
      break
    fi
  done
  if [ -z "$MCP_CMD" ]; then
    for path_try in \
      "$HOME/.local/bin/banditdb-mcp" \
      "$HOME/Library/Python/3.12/bin/banditdb-mcp" \
      "$HOME/Library/Python/3.11/bin/banditdb-mcp" \
      "$HOME/Library/Python/3.10/bin/banditdb-mcp"; do
      [ -x "$path_try" ] && MCP_CMD="$path_try" && break
    done
  fi
  if [ -z "$MCP_CMD" ]; then
    python3 -c "import banditdb_mcp" &>/dev/null 2>&1 && \
      MCP_CMD="python3 -m banditdb_mcp" || \
      die "banditdb-mcp not found in PATH. Add $(python3 -m site --user-base)/bin to your PATH."
  fi

  # Merge into mcp.json
  python3 - <<PYEOF
import json, os

cmd_parts = """$MCP_CMD""".split()
entry = {
    "command": cmd_parts[0],
    "env": {"BANDITDB_URL": "$BANDITDB_URL", "BANDITDB_API_KEY": "$BANDITDB_API_KEY"}
}
if len(cmd_parts) > 1:
    entry["args"] = cmd_parts[1:]

path = "$MCP_JSON"
cfg = json.load(open(path)) if os.path.exists(path) else {}
cfg.setdefault("mcpServers", {})["banditdb"] = entry
with open(path, "w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
print(f"Written: {path}")
PYEOF

  # Write command skill
  mkdir -p "$CLAUDE_DIR/commands"
  cat > "$CLAUDE_DIR/commands/banditdb.md" <<'SKILL'
You are operating BanditDB — the Intuition Database. An in-memory decision database for agents and applications that need to learn which decisions work and get better with every outcome.

## Setup check
First, call `mcp__banditdb__list_campaigns` to verify the connection.
If MCP tools are unavailable, tell the user to restart Claude Code.

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
- `linucb`      — contextual LinUCB (default, supports propensity export)
- `thompson`    — Thompson Sampling
- `neural`      — NeuralLinUCB (MLP feature layer)
- `tournament`  — Progressive Tournament (base vs challenger)

## Reward design
Rewards are floats, typically in [0, 1]. Normalise continuous outcomes.
Delayed rewards (next-day, post-checkout) are supported via the TTL cache.

$ARGUMENTS
SKILL

  ok "Manual configuration written to $CLAUDE_DIR"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
printf "\n"
printf "${GRN}  ════════════════════════════════════════════${RST}\n"
printf "${GRN}  BanditDB × Claude Code — setup complete!${RST}\n"
printf "${GRN}  ════════════════════════════════════════════${RST}\n\n"
printf "  BanditDB:  %s\n" "$BANDITDB_URL"
printf "\n"
printf "  Next steps:\n"
printf "  1. Open a new Claude Code session:  claude\n"
printf "  2. Type:  /banditdb:banditdb list all campaigns\n"
printf "\n"
[ -n "$BANDITDB_API_KEY" ] || \
  printf "  ${YLW}Tip:${RST} set BANDITDB_API_KEY before running to wire auth.\n\n"
