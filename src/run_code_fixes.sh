#!/bin/bash
set -e
LIMIT=164

echo "--- Starting Code Fixes Run (Timeout Added) ---"

# 3. Fix Code Inflation (Local Models)
# Re-run Code with execution metric

echo ">> Re-running Local Code with execution metric (Timeout 5s)..."
# We'll run them sequentially
python3 src/local_eval.py --model llama3.2:latest --task code --limit $LIMIT
python3 src/local_eval.py --model mistral:latest --task code --limit $LIMIT
python3 src/local_eval.py --model phi4:latest --task code --limit $LIMIT
python3 src/local_eval.py --model qwen2.5:7b --task code --limit $LIMIT

echo "--- Code Fixes Complete ---"
