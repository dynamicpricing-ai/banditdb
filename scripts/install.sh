#!/usr/bin/env sh
# BanditDB installer
# Usage: curl -fsSL https://raw.githubusercontent.com/dynamicpricing-ai/banditdb/main/scripts/install.sh | sh
set -e

REPO="dynamicpricing-ai/banditdb"
BIN_NAME="banditdb"
INSTALL_DIR="${BANDITDB_INSTALL_DIR:-$HOME/.local/bin}"

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
if command -v jq > /dev/null 2>&1; then
  version="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" | jq -r '.tag_name')"
else
  version="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
    | grep '"tag_name"' | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/')"
fi

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

bin_path="$(find "$tmp" -name "$BIN_NAME" -type f | head -1)"
if [ -z "$bin_path" ]; then
  echo "Could not find '$BIN_NAME' binary after extraction." >&2
  exit 1
fi



# ── Install ───────────────────────────────────────────────────────────────────
mkdir -p "$INSTALL_DIR"
if [ -w "$INSTALL_DIR" ]; then
  mv "$bin_path" "$INSTALL_DIR/$BIN_NAME"
  chmod +x "$INSTALL_DIR/$BIN_NAME"
else
  echo "Installing to $INSTALL_DIR (sudo required)..."
  echo "Tip: set BANDITDB_INSTALL_DIR to a writable path to skip sudo."
  sudo mv "$bin_path" "$INSTALL_DIR/$BIN_NAME"
  sudo chmod +x "$INSTALL_DIR/$BIN_NAME"
fi

if ! echo ":$PATH:" | grep -qF ":$INSTALL_DIR:"; then
  echo ""
  echo "  $INSTALL_DIR is not on your PATH."
  echo "  Add this to your shell profile (~/.bashrc, ~/.zshrc, etc.):"
  echo "    export PATH=\"$INSTALL_DIR:\$PATH\""
fi

cat <<EOF

BanditDB $version installed to $INSTALL_DIR/$BIN_NAME

Start the server (run this first!):
  banditdb

Then verify it's up:
  curl http://localhost:8080/health

EOF