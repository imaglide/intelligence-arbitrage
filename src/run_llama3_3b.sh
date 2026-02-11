#!/bin/bash
set -e
LIMIT=200

echo "--- Starting llama3.2_3b Benchmark Runs ---"

# Model name note: User referred to "llama3.2_3b" which usually maps to "llama3.2:3b" or just "llama3.2" (which is 3B by default for 3.2 unless specs say 1B). 
# However, "llama3.2" row exists and is full. "llama3.2_3b" row exists and is empty.
# In Archive, files were likely named "local_baseline_{task}_llama3.2_3b.csv".
# We will use explicit model name "llama3.2:3b" for Ollama to be safe and consistent.

MODEL="llama3.2:3b"

echo ">> Running $MODEL agentic..."
python3 src/local_eval.py --model "$MODEL" --task agentic --limit $LIMIT

echo ">> Running $MODEL synthesis..."
python3 src/local_eval.py --model "$MODEL" --task synthesis --limit $LIMIT

echo ">> Running $MODEL code..."
python3 src/local_eval.py --model "$MODEL" --task code --limit $LIMIT

echo "--- llama3.2_3b Complete ---"
