#!/bin/bash

case "$1" in
    start)
        echo "Starting AIVM devnet..."
        echo "Please ensure you have:"
        echo "1. Activated your virtual environment with 'source .venv/bin/activate'"
        echo "2. Installed dependencies with 'pip install \"nillion-aivm[examples]\"'"
        echo ""
        echo "Starting devnet in this terminal. Keep this terminal open and running."
        echo "Open a new terminal to continue with the workshop steps."
        aivm-devnet
        ;;
    status)
        if pgrep -f "aivm-devnet" > /dev/null; then
            echo "✓ AIVM devnet is running"
            echo "You can proceed with the workshop steps in a different terminal"
        else
            echo "❌ AIVM devnet is not running"
            echo "Please start the devnet first with: $0 start"
            echo "Keep the devnet terminal open and use a new terminal for other commands"
        fi
        ;;
    *)
        echo "Usage: $0 {start|status}"
        echo ""
        echo "start  - Start the AIVM development network (keep terminal open)"
        echo "status - Check if AIVM devnet is running"
        exit 1
esac