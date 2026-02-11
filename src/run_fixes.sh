#!/bin/bash
set -e
LIMIT=200

echo "--- Starting Fixes Run ---"

# 1. Fix llama3.2_3b Zeroes
# We need to run the tasks that were essentially skipped or errored out
MODEL_3B="llama3.2:3b"
echo ">> Running $MODEL_3B missing tasks..."
python3 src/local_eval.py --model "$MODEL_3B" --task classification --limit $LIMIT
python3 src/local_eval.py --model "$MODEL_3B" --task math --limit $LIMIT
python3 src/local_eval.py --model "$MODEL_3B" --task qa --limit $LIMIT
python3 src/local_eval.py --model "$MODEL_3B" --task extraction --limit $LIMIT
python3 src/local_eval.py --model "$MODEL_3B" --task analysis --limit $LIMIT
python3 src/local_eval.py --model "$MODEL_3B" --task rag --limit $LIMIT


# 2. Fix Classification Collapse (GPT-4o)
echo ">> Re-running GPT-4o Classification with new signature..."
python3 src/api_eval.py --model gpt-4o --task classification --limit $LIMIT


# 3. Fix Code Inflation (Local Models)
# Re-run Code with execution metric
echo ">> Re-running Local Code with execution metric..."
python3 src/local_eval.py --model llama3.2:latest --task code --limit 164
python3 src/local_eval.py --model mistral:latest --task code --limit 164
python3 src/local_eval.py --model phi4:latest --task code --limit 164
python3 src/local_eval.py --model qwen2.5:7b --task code --limit 164

echo "--- Fixes Complete ---"
