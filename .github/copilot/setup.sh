#!/bin/bash
set -e

echo "Setting up Copilot development environment..."

# Install uv (version matching GitHub Actions)
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/0.10.2/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Verify uv installation
uv --version

# Install Python and dependencies
echo "Installing dependencies..."
uv sync --all-packages

echo "Development environment setup complete!"
