#!/bin/bash
set -e
LIMIT=200

echo "--- Starting Classification Fix Run ---"
# 2. Fix Classification Collapse (GPT-4o)
echo ">> Re-running GPT-4o Classification with new signature and fixed loader..."
python3 src/api_eval.py --model gpt-4o --task classification --limit $LIMIT

echo "--- Classification Fix Complete ---"
