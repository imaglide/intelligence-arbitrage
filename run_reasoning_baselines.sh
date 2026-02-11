#!/bin/bash
# Run baseline evaluations for reasoning models (qwen3:4b, deepseek-r1:7b)
# Matches Phase 1 methodology: 50 examples, temperature=0, ChainOfThought/Predict

set -e

TASKS="classification math qa extraction analysis synthesis agentic rag code"
MODELS="qwen3:4b deepseek-r1:7b"
LIMIT=50

for MODEL in $MODELS; do
    echo ""
    echo "=========================================="
    echo "  MODEL: $MODEL"
    echo "=========================================="
    for TASK in $TASKS; do
        SAFE_MODEL=$(echo $MODEL | tr ':' '_')
        OUTPUT="results/local_baseline_${TASK}_${SAFE_MODEL}.csv"

        if [ -f "$OUTPUT" ]; then
            echo "SKIP: $OUTPUT already exists"
            continue
        fi

        echo ""
        echo ">>> Running: $MODEL on $TASK (limit=$LIMIT)"
        PYTHONUNBUFFERED=1 python3 -u -m src.local_eval \
            --model "$MODEL" \
            --task "$TASK" \
            --limit $LIMIT \
            --temperature 0.0

        echo ">>> Done: $MODEL on $TASK"
    done
done

echo ""
echo "All baseline evaluations complete!"
