# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM rust:1-bookworm AS builder

RUN apt-get update && apt-get install -y \
    pkg-config libssl-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/banditdb
COPY Cargo.toml Cargo.lock* ./
COPY src ./src
COPY docs ./docs

# Release build. Add --features neural for NeuralLinUCB support.
ARG FEATURES=""
RUN if [ -z "$FEATURES" ]; then \
        cargo build --release; \
    else \
        cargo build --release --features "$FEATURES"; \
    fi

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates curl gosu \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -r -s /bin/false -u 1001 banditdb && mkdir -p /data && chown banditdb:banditdb /data

COPY --from=builder /usr/src/banditdb/target/release/banditdb /usr/local/bin/banditdb
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /data

ENV DATA_DIR="/data" \
    PORT="8080" \
    BANDITDB_CHECKPOINT_INTERVAL="" \
    BANDITDB_MAX_WAL_SIZE_MB="" \
    LOG_FORMAT="" \
    BANDITDB_TENANT_MODE="false"

VOLUME ["/data"]
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:8080/health | grep -q '"status":"ok"' || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["banditdb"]
