#!/usr/bin/env sh
# BanditDB installer
# Usage: curl -fsSL https://raw.githubusercontent.com/dynamicpricing-ai/banditdb/main/scripts/install.sh | sh
set -e

REPO="dynamicpricing-ai/banditdb"
BIN_NAME="banditdb"
INSTALL_DIR="${BANDITDB_INSTALL_DIR:-/usr/local/bin}"

# ── Detect OS and architecture ────────────────────────────────────────────────
os="$(uname -s)"
arch="$(uname -m)"

case "$os" in
  Linux)
    case "$arch" in
      x86_64)  target="x86_64-unknown-linux-gnu" ;;
      aarch64|arm64) target="aarch64-unknown-linux-gnu" ;;
      *) echo "Unsupported Linux architecture: $arch" >&2; exit 1 ;;
    esac
    ;;
  Darwin)
    case "$arch" in
      x86_64)  target="x86_64-apple-darwin" ;;
      arm64)   target="aarch64-apple-darwin" ;;
      *) echo "Unsupported macOS architecture: $arch" >&2; exit 1 ;;
    esac
    ;;
  *)
    echo "Unsupported OS: $os" >&2
    echo "For Windows, download the binary from: https://github.com/$REPO/releases/latest" >&2
    exit 1
    ;;
esac

# ── Resolve latest version ────────────────────────────────────────────────────
version="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
  | grep '"tag_name"' | sed 's/.*"tag_name": *"\(.*\)".*/\1/')"

if [ -z "$version" ]; then
  echo "Could not determine latest BanditDB version." >&2
  exit 1
fi

echo "Installing BanditDB $version for $target..."

# ── Download and extract ──────────────────────────────────────────────────────
url="https://github.com/$REPO/releases/download/$version/${BIN_NAME}-${version}-${target}.tar.gz"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

curl -fsSL "$url" -o "$tmp/banditdb.tar.gz"
tar -xzf "$tmp/banditdb.tar.gz" -C "$tmp"

# ── Install ───────────────────────────────────────────────────────────────────
if [ -w "$INSTALL_DIR" ]; then
  mv "$tmp/$BIN_NAME" "$INSTALL_DIR/$BIN_NAME"
  chmod +x "$INSTALL_DIR/$BIN_NAME"
else
  echo "Installing to $INSTALL_DIR (sudo required)..."
  sudo mv "$tmp/$BIN_NAME" "$INSTALL_DIR/$BIN_NAME"
  sudo chmod +x "$INSTALL_DIR/$BIN_NAME"
fi

echo ""
echo "BanditDB $version installed to $INSTALL_DIR/$BIN_NAME"
echo ""
echo "Start the server:"
echo "  banditdb"
echo ""
echo "Health check:"
echo "  curl http://localhost:8080/health"
