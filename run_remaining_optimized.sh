#!/bin/bash
# Run remaining optimized evaluations for Tasks 4, 5, 6
set -e
export PYTHONUNBUFFERED=1

TASKS="agentic analysis classification code extraction math qa rag synthesis"

echo "========================================"
echo "TASK 6: Mistral synthesis (1 eval)"
echo "========================================"
python3 src/local_eval.py --model mistral:latest --task synthesis \
    --program results/optimized_programs/mistral_synthesis_mipro_v2.json \
    --limit 50 --run-id optimized_mistral
echo ">> Task 6 DONE"

echo ""
echo "========================================"
echo "TASK 4: Phi4 all 9 tasks"
echo "========================================"
for TASK in $TASKS; do
    echo ">> phi4 / $TASK"
    python3 src/local_eval.py --model phi4 --task "$TASK" \
        --program "results/optimized_programs/phi4_${TASK}_mipro_v2.json" \
        --limit 50 --run-id optimized_phi4
done
echo ">> Task 4 DONE"

echo ""
echo "========================================"
echo "TASK 5: Qwen2.5:7b all 9 tasks"
echo "========================================"
for TASK in $TASKS; do
    echo ">> qwen2.5:7b / $TASK"
    python3 src/local_eval.py --model qwen2.5:7b --task "$TASK" \
        --program "results/optimized_programs/qwen2.5_7b_${TASK}_mipro_v2.json" \
        --limit 50 --run-id optimized_qwen2.5_7b
done
echo ">> Task 5 DONE"

echo ""
echo "========================================"
echo "ALL COMPLETE"
echo "========================================"
