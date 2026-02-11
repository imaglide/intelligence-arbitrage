import subprocess
import argparse
import sys
import time
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
# Defined in PRD
TASKS = Config.TASKS
LIMIT = 200

# Models to run
# Models to run
API_MODELS = list(Config.MODEL_PRICING.keys())
LOCAL_MODELS = list(Config.OLLAMA_MODELS.values())

import datetime

# Ensure logs dir exists
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Generate log filename for this run
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"benchmark_run_{TIMESTAMP}.log")

def run_command(cmd):
    # Tee output to both stdout and log file
    full_cmd = f"{cmd} 2>&1 | tee -a {LOG_FILE}"
    print(f"Running: {cmd} (Logging to {LOG_FILE})")
    try:
        # Open log file to write header
        with open(LOG_FILE, "a") as f:
            f.write(f"\n\n--- Executing: {cmd} ---\n")
            
        subprocess.check_call(full_cmd, shell=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        with open(LOG_FILE, "a") as f:
            f.write(f"\n[ERROR] Command failed with exit code {e.returncode}\n")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run with limit=1 for all")
    parser.add_argument("--skip-api", action="store_true", help="Skip API models")
    parser.add_argument("--skip-local", action="store_true", help="Skip Local models")
    parser.add_argument("--filter-model", type=str, help="Only run models containing this string")
    parser.add_argument("--filter-task", type=str, help="Only run tasks containing this string")
    args = parser.parse_args()

    current_limit = 1 if args.dry_run else LIMIT
    failed_runs = []

    print(f"--- Starting Full Benchmark Run (Limit per task: {current_limit}) ---")
    start_time = time.time()

    # 1. API Baselines
    if not args.skip_api:
        for model in API_MODELS:
            if args.filter_model and args.filter_model not in model:
                continue
            for task in TASKS:
                if args.filter_task and args.filter_task not in task:
                    continue
                print(f"\n>>> Benchmarking API: {model} on {task}")
                success = run_command(f"python3 src/api_eval.py --model {model} --task {task} --limit {current_limit}")
                if not success:
                    failed_runs.append(f"API: {model}-{task}")
                time.sleep(1) # Brief pause

    # 2. Local Baselines
    if not args.skip_local:
        for model in LOCAL_MODELS:
            if args.filter_model and args.filter_model not in model:
                continue
            for task in TASKS:
                if args.filter_task and args.filter_task not in task:
                    continue
                print(f"\n>>> Benchmarking Local: {model} on {task}")
                # Note: local_eval expects model tag exactly as in ollama list
                success = run_command(f"python3 src/local_eval.py --model {model} --task {task} --limit {current_limit}")
                if not success:
                    failed_runs.append(f"Local: {model}-{task}")
                time.sleep(2) # Allow cooling/cleanup

    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n\n--- Benchmark Run Complete in {duration:.2f}s ---")
    if failed_runs:
        print("FAILED RUNS:")
        for failures in failed_runs:
            print(f"- {failures}")
    else:
        print("All runs completed successfully.")

if __name__ == "__main__":
    main()
