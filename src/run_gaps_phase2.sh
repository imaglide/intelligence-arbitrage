#!/bin/bash
set -e
LIMIT=200

echo "--- Starting Phase 2 Gap Fill Runs ---"

# 1. GPT-4o-mini Analysis (Fast)
echo ">> Running gpt-4o-mini analysis..."
python3 src/api_eval.py --model gpt-4o-mini --task analysis --limit $LIMIT

# 2. Mistral Extraction (Local)
echo ">> Running mistral extraction..."
python3 src/local_eval.py --model "mistral:latest" --task extraction --limit $LIMIT

# 3. Phi4 Code (Local - might be slow, let's try)
echo ">> Running phi4 code..."
python3 src/local_eval.py --model "phi4:latest" --task code --limit $LIMIT

echo "--- Phase 2 Complete ---"
