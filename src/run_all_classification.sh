#!/bin/bash
set -e
LIMIT=200

echo "--- Starting Global Classification Fix Run ---"

# API Models
echo ">> Re-running Classification for gpt-4o-mini..."
python3 src/api_eval.py --model gpt-4o-mini --task classification --limit $LIMIT

# Local Models
echo ">> Re-running Classification for Local Models..."
python3 src/local_eval.py --model llama3.2:latest --task classification --limit $LIMIT
python3 src/local_eval.py --model llama3.2:3b --task classification --limit $LIMIT
python3 src/local_eval.py --model mistral:latest --task classification --limit $LIMIT
python3 src/local_eval.py --model phi4:latest --task classification --limit $LIMIT
python3 src/local_eval.py --model qwen2.5:7b --task classification --limit $LIMIT

echo "--- Global Classification Fix Complete ---"
