# ==========================================
# Stage 1: The Builder (Heavy, contains compilers)
# ==========================================
FROM rust:1.93-slim-bookworm as builder

# Install C/C++ build tools required by Polars and ndarray
RUN apt-get update && apt-get install -y pkg-config libssl-dev build-essential

# Create a new empty shell project
WORKDIR /usr/src/banditdb
COPY Cargo.toml Cargo.lock* ./
COPY src ./src

# Build the project in Release mode (Maximum optimization)
RUN cargo build --release

# ==========================================
# Stage 2: The Production Runtime (Tiny, Secure)
# ==========================================
FROM debian:bookworm-slim

# Install CA certificates (useful if DB ever needs to make outbound webhooks later)
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

# Create a directory for our persistent data
RUN mkdir -p /data
ENV DATA_DIR="/data"

# Copy the compiled Rust binary from the builder stage
COPY --from=builder /usr/src/banditdb/target/release/banditdb /usr/local/bin/banditdb

# Expose the Axum Web Server port
EXPOSE 8080

# Run the database!
CMD ["banditdb"]