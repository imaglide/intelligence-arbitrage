#!/bin/bash
# run_experiment.sh
# Orchestrates the full Golden Run for DSPy Delta Experiment

set -e # Exit on error

# Configuration
MODELS=("mistral" "llama3.2" "qwen2.5") 
# API models can be added here if we are optimizing them too, but usually we use them as teachers or baselines.
# For this script, we assume we are optimizing LOCAL Student models using a generic Teacher (e.g., gpt-4o or self).

TASKS=("synthesis" "info_extraction" "classification") # Mapped to internal names below if needed
# Actually, let's use the internal names from config
TASKS=("synthesis" "extraction" "classification" "math" "qa")

TEACHER="gpt-4o" # Gold standard teacher
OPTIMIZER="mipro_v2"

DRY_RUN=false
CACHE_SETTING="True"

# Parse Flags
for arg in "$@"
do
    case $arg in
        --dry-run)
        DRY_RUN=true
        echo ">> DRY RUN MODE ENABLED"
        ;;
    esac
done

if [ "$DRY_RUN" = true ]; then
    # Keep cache for dry run to be fast, or clear it if we want to test that too.
    # Let's keep it to verify logic.
    CACHE_SETTING="True"
else
    # Golden Run: CLEAR CACHE
    echo ">> CLEARING CACHE for Golden Run..."
    rm -rf .dspy_cache/
    export DSP_CACHEBOOL="False"
fi

# Ensure directories exist
mkdir -p results/optimized_programs

# Loop Structure Refactored for "Baselines First" Strategy

# PHASE 1: BASELINES
echo "================================================================"
echo "PHASE 1: RUNNING ALL BASELINES"
echo "================================================================"
for MODEL in "${MODELS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        echo ">> Baseline: $MODEL | $TASK"
        python3 src/local_eval.py --model "$MODEL" --task "$TASK" $DRY_RUN_FLAG
    done
done

# PHASE 2: OPTIMIZATION
echo "================================================================"
echo "PHASE 2: RUNNING ALL OPTIMIZATIONS"
echo "================================================================"
for MODEL in "${MODELS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        echo ">> Optimization: $MODEL | $TASK"
        
        python3 src/optimize.py \
            --model "$MODEL" \
            --task "$TASK" \
            --optimizer "$OPTIMIZER" \
            --teacher "$TEACHER" \
            $DRY_RUN_FLAG
    done
done

# PHASE 3: EVALUATION OF OPTIMIZED PROGRAMS
echo "================================================================"
echo "PHASE 3: EVALUATING OPTIMIZED PROGRAMS"
echo "================================================================"
for MODEL in "${MODELS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        echo ">> Eval Optimized: $MODEL | $TASK"

        # Construct expected program path
        SAFE_MODEL=${MODEL//:/_}
        SAFE_MODEL=${SAFE_MODEL//\//_}
        PROGRAM_PATH="results/optimized_programs/${SAFE_MODEL}_${TASK}_${OPTIMIZER}.json"
        
        if [ -f "$PROGRAM_PATH" ]; then
            python3 src/local_eval.py \
                --model "$MODEL" \
                --task "$TASK" \
                --program "$PROGRAM_PATH" \
                $DRY_RUN_FLAG
        else
            echo "!! WARNING: Optimization failed to produce program at $PROGRAM_PATH. Skipping eval."
        fi
    done
done

echo ">> Golden Run Complete!"
