#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Installing MedVerse inference engine ==="

# 1. Core Python dependencies (PyTorch + Transformers)
echo "Installing PyTorch and Transformers..."
pip install --upgrade pip
pip install -r requirement.txt

# 2. SGLang (editable, from bundled fork)
echo "Installing SGLang fork (editable)..."
cd sglang
pip install -e "python[all]"
cd ..

echo ""
echo "=== Installation complete ==="
echo ""
echo "Start the server with:"
echo "  python -m sglang.srt.entrypoints.medverse_server \\"
echo "      --model-path /path/to/MedVerse-14B \\"
echo "      --tp-size 1 --port 30000 --trust-remote-code"
