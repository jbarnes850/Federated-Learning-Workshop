#!/bin/bash

# Exit on error
set -e

echo "Starting installation..."

# Check Python version
python3 --version || {
    echo "❌ Python 3 not found"
    exit 1
}

# Create and activate virtual environment
python3 -m venv .venv || {
    echo "❌ Failed to create virtual environment"
    exit 1
}
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt || {
    echo "❌ Failed to install requirements"
    exit 1
}
pip install "nillion-aivm[examples]" || {
    echo "❌ Failed to install AIVM"
    exit 1
}

# Set up AIVM configuration
if [ -z "$AIVM_API_KEY" ]; then
    echo "⚠️ AIVM_API_KEY not set. Please set it manually."
    echo "export AIVM_API_KEY='your_api_key_here'" >> .env
else
    echo "✓ AIVM_API_KEY found"
fi

# Set up Nillion Network configuration
if [ -z "$NILLION_NODE_KEY" ]; then
    echo "⚠️ NILLION_NODE_KEY not set. Please set it manually."
    echo "export NILLION_NODE_KEY='your_node_key_here'" >> .env
fi

if [ -z "$NILLION_NETWORK_KEY" ]; then
    echo "⚠️ NILLION_NETWORK_KEY not set. Please set it manually."
    echo "export NILLION_NETWORK_KEY='your_network_key_here'" >> .env
fi

# Start AIVM devnet
echo "Starting AIVM devnet..."
aivm-devnet || {
    echo "❌ Failed to start AIVM devnet"
    exit 1
}

echo "✓ Installation complete"
