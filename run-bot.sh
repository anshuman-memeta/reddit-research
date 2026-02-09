#!/bin/bash
# Run the Reddit Research Bot
# Usage: ./run-bot.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load environment variables
ENV_FILE="${SCRIPT_DIR}/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "Error: .env file not found at $ENV_FILE"
    echo "Copy .env.example to .env and fill in your credentials."
    exit 1
fi

cd "$SCRIPT_DIR"
python3 bot.py
