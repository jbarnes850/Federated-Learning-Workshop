#!/bin/bash

case "$1" in
    start)
        echo "Starting AIVM devnet..."
        aivm-devnet
        ;;
    status)
        if pgrep -f "aivm-devnet" > /dev/null; then
            echo "✓ AIVM devnet is running"
        else
            echo "❌ AIVM devnet is not running"
        fi
        ;;
    *)
        echo "Usage: $0 {start|status}"
        exit 1
esac 