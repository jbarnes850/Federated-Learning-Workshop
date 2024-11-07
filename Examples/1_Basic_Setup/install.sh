#!/bin/bash
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

echo "✓ Installation complete"
echo "To start the workshop:"
echo "1. Open a new terminal and run: aivm-devnet"
echo "2. In this terminal, run: python Examples/1_Basic_Setup/test_setup.py"
