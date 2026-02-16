#!/bin/bash
set -e

echo "Setting up Copilot development environment..."

# Install uv (version matching GitHub Actions)
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/0.10.2/install.sh | sh

# Source uv environment to make it available in the current shell
if [ -f "$HOME/.local/bin/env" ]; then
    . "$HOME/.local/bin/env"
fi
export PATH="$HOME/.local/bin:$PATH"

# Verify uv installation
echo "Verifying uv installation..."
if ! uv --version; then
    echo "ERROR: uv installation failed or uv is not in PATH"
    exit 1
fi

# Install Python and dependencies
echo "Installing dependencies..."
uv sync --all-packages

echo "Development environment setup complete!"
