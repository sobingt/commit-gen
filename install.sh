#!/bin/bash
set -e

echo "=== Git Commit Message Generator — Installer ==="
echo ""

# --- Check Python ---
if ! command -v python3 &>/dev/null; then
    echo "❌ Python3 is required but not found. Please install it first."
    exit 1
fi

PYTHON=$(command -v python3)
echo "✅ Python: $($PYTHON --version)"

# --- Check pip ---
if ! $PYTHON -m pip --version &>/dev/null; then
    echo "❌ pip not found. Please install pip first."
    exit 1
fi

# --- Install llama-cpp-python ---
echo ""
echo "📦 Installing llama-cpp-python (CPU)..."
$PYTHON -m pip install llama-cpp-python --upgrade --quiet
echo "✅ llama-cpp-python installed"

# --- Install huggingface_hub ---
echo ""
echo "📦 Installing huggingface_hub..."
$PYTHON -m pip install huggingface_hub --upgrade --quiet
echo "✅ huggingface_hub installed"

# --- Download model ---
MODEL_DIR="$HOME/.cache/git-commit-gen"
mkdir -p "$MODEL_DIR"

EXISTING=$(find "$MODEL_DIR" -name "*.gguf" | head -1)
if [ -n "$EXISTING" ]; then
    echo ""
    echo "✅ Model already exists at: $EXISTING"
else
    echo ""
    echo "⬇️  Downloading TinyLlama 1.1B (~668 MB, one-time download)..."
    $PYTHON -c "
from huggingface_hub import hf_hub_download
import os
path = hf_hub_download(
    repo_id='TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF',
    filename='tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
    local_dir=os.path.expanduser('~/.cache/git-commit-gen'),
)
print('Saved to: ' + path)
"
    echo "✅ Model downloaded to: $MODEL_DIR"
fi

# --- Copy run script to PATH ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_TARGET="/usr/local/bin/git-commit-gen"

echo ""
echo "🔧 Installing git-commit-gen to $INSTALL_TARGET..."

if [ -w "/usr/local/bin" ]; then
    cp "$SCRIPT_DIR/run.sh" "$INSTALL_TARGET"
    chmod +x "$INSTALL_TARGET"
else
    sudo cp "$SCRIPT_DIR/run.sh" "$INSTALL_TARGET"
    sudo chmod +x "$INSTALL_TARGET"
fi

echo "✅ Installed to $INSTALL_TARGET"

# --- Done ---
echo ""
echo "================================================"
echo "✅ Setup complete!"
echo ""
echo "Usage:"
echo "  git diff | git-commit-gen"
echo "  git diff --cached | git-commit-gen"
echo ""
echo "Or add a git alias:"
echo "  git config --global alias.gen-commit '!git diff --cached | git-commit-gen'"
echo "  git gen-commit"
echo "================================================"