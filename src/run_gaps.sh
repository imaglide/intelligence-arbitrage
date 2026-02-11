#!/bin/bash
set -e

# LIMIT=200 as per requirement to fill holes
LIMIT=200

echo "--- Starting Gap Fill Runs ---"

# 1. GPT-4o
echo ">> Running gpt-4o synthesis..."
python3 src/api_eval.py --model gpt-4o --task synthesis --limit $LIMIT

# 2. GPT-4o-mini
echo ">> Running gpt-4o-mini agentic..."
python3 src/api_eval.py --model gpt-4o-mini --task agentic --limit $LIMIT
echo ">> Running gpt-4o-mini synthesis..."
python3 src/api_eval.py --model gpt-4o-mini --task synthesis --limit $LIMIT
echo ">> Running gpt-4o-mini code..."
python3 src/api_eval.py --model gpt-4o-mini --task code --limit $LIMIT
echo ">> Running gpt-4o-mini analysis..."
python3 src/api_eval.py --model gpt-4o-mini --task analysis --limit $LIMIT

# 3. Mistral (Local)
echo ">> Running mistral extraction..."
# Note: config.py map "mistral" -> "mistral:latest"
python3 src/local_eval.py --model "mistral:latest" --task extraction --limit $LIMIT

# 4. Phi4 (Local) -> "phi4" not in main map?
# Check Config.MODELS. Wait, user said "phi4" but config has:
# "mistral": "mistral:latest", "llama3.2": "llama3.2", "qwen2.5": "qwen2.5:7b"
# The user might have older config or manual mapping.
# In src/config.py, I see:
#     MODELS = {
#        "llama3.2": "llama3.2",
#        "qwen2.5": "qwen2.5:7b",
#        "mistral": "mistral:latest"
#    }
# Phi4 is NOT in the current config.
# However, local_eval just takes --model and passes it to ollama/openai client.
# Attempting "phi4:latest" if that's what was used before.
# Previous logs named it "local_baseline_agentic_phi4_latest.csv".
# So model name likely "phi4:latest".
echo ">> Running phi4 code..."
python3 src/local_eval.py --model "phi4:latest" --task code --limit $LIMIT

echo "--- Gap Fill Complete ---"
