#!/bin/bash

# Exit on error
set -e

echo "Starting installation..."

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Copy environment example if .env doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✓ Created .env file from example"
fi

# Start AIVM devnet
echo "Starting AIVM devnet..."
aivm-devnet || {
    echo "❌ Failed to start AIVM devnet"
    exit 1
}

echo "✓ Installation complete - AIVM devnet running"
