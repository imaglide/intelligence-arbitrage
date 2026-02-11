import os
import sys
import subprocess
import glob
import json
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config

import argparse

# 1. Configuration
EXPERIMENT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "baseline_v2")
os.makedirs(EXPERIMENT_DIR, exist_ok=True)
RAW_SCORES_PATH = os.path.join(EXPERIMENT_DIR, "raw_scores.json")
SUMMARY_PATH = os.path.join(EXPERIMENT_DIR, "summary_table.md")

LIMIT = 50 # Heuristic limit for baseline
TEMPERATURE = 0.0

# Define Model Mapping (Experiment Name -> API/Ollama Name)
MODELS = {
    # API
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-5.2": "gpt-5.2", 
    "gpt-5.2-chat-latest": "gpt-5.2-chat-latest",
    
    # Local (Ollama Tags)
    "llama3.2": "llama3.2",
    "llama3.2_3b": "llama3.2", # assuming standard is 3b
    "mistral": "mistral",
    "phi4": "phi4",
    "qwen2.5_7b": "qwen2.5:7b"
}

TASKS = ["agentic", "analysis", "classification", "code", "extraction", "math", "qa", "rag", "synthesis"]

def get_script_for_model(model_name):
    if "gpt" in model_name:
        return "src/api_eval.py"
    else:
        return "src/local_eval.py"

def run_baselines(dry_run=False, filter_model=None, filter_task=None):
    print(f"--- Starting Baseline V2 (Temp={TEMPERATURE}, Limit={LIMIT}, DryRun={dry_run}) ---")
    
    # Strictly enforce Cache=False via Env Var for this process and children
    os.environ["DSP_CACHEBOOL"] = "False"
    
    for model_key, model_name in MODELS.items():
        if filter_model and filter_model not in model_key:
            continue
            
        script = get_script_for_model(model_key)
        
        # Special handling for gpt-5.2-chat-latest which rejects temp=0
        current_temp = TEMPERATURE
        if model_key == "gpt-5.2-chat-latest":
            current_temp = 1.0
        
        for task in TASKS:
            if filter_task and filter_task not in task:
                continue

            print(f"\n[Running] Model: {model_key} ({model_name}) | Task: {task}")
            
            cmd = [
                "python3", script,
                "--model", model_name,
                "--task", task,
                "--temperature", str(current_temp),
                "--run-id", model_key # Use unique experiment key as run-id
            ]
            
            if dry_run:
                cmd.append("--dry-run")
            elif LIMIT:
                cmd.extend(["--limit", str(LIMIT)])
                
            try:
                subprocess.run(cmd, check=True, env=os.environ)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Failed {model_key} on {task}: {e}")

def aggregate_results():
    print("\n--- Aggregating Results ---")
    
    scores = {} # {model: {task: score}}
    
    # Initialize structure
    for m in MODELS.keys():
        scores[m] = {}
        for t in TASKS:
            scores[m][t] = "N/A"
            
    results_dir = Config.RESULTS_DIR
    
    for model_key in MODELS.keys():
        # Determine prefix based on model type
        prefix = "api" if "gpt" in model_key else "local"
        
        for task in TASKS:
            # Expected filename format from api_eval/local_eval with run_id:
            # {prefix}_baseline_{task}_{run_id}.csv
            filename = f"{prefix}_baseline_{task}_{model_key}.csv"
            filepath = os.path.join(results_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"[Results] Missing file: {filename}")
                continue
                
            try:
                df = pd.read_csv(filepath)
                if df.empty or "is_correct" not in df.columns:
                    print(f"[Results] Empty or invalid file: {filename}")
                    continue
                
                score = df["is_correct"].mean()
                scores[model_key][task] = round(score, 4)
                
            except Exception as e:
                print(f"[Results] Error processing {filename}: {e}")

    # Save JSON
    with open(RAW_SCORES_PATH, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"Raw scores saved to {RAW_SCORES_PATH}")
    
    return scores

def generate_markdown(scores):
    # Create Table
    # | Model | Task1 | Task2 | ... |
    # |-------|-------|-------| ... |
    
    headers = ["Model"] + TASKS
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    
    for model in MODELS.keys():
        row = [model]
        for task in TASKS:
            val = scores.get(model, {}).get(task, "N/A")
            if isinstance(val, float):
                row.append(f"{val:.2%}")
            else:
                row.append(str(val))
        lines.append("| " + " | ".join(row) + " |")
    
    table_str = "\n".join(lines)
    
    with open(SUMMARY_PATH, "w") as f:
        f.write("# Baseline V2 Results\n\n")
        f.write(table_str)
        
    print(f"\nMarkdown summary saved to {SUMMARY_PATH}")
    print("\n" + table_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run with limit 5 for verification")
    parser.add_argument("--filter-model", type=str, help="Only run specific model")
    parser.add_argument("--filter-task", type=str, help="Only run specific task")
    args = parser.parse_args()
    
    run_baselines(dry_run=args.dry_run, filter_model=args.filter_model, filter_task=args.filter_task)
    scores = aggregate_results()
    generate_markdown(scores)
