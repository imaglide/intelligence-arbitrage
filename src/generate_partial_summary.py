import os
import sys
import glob
import pandas as pd
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config

MODELS = {
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-5.2": "gpt-5.2", 
    "gpt-5.2-chat-latest": "gpt-5.2-chat-latest",
    "llama3.2": "llama3.2",
    "llama3.2_3b": "llama3.2", 
    "mistral": "mistral",
    "phi4": "phi4",
    "qwen2.5_7b": "qwen2.5:7b"
}

TASKS = ["agentic", "analysis", "classification", "code", "extraction", "math", "qa", "rag", "synthesis"]

def aggregate_results():
    print("\n--- Aggregating Partial Results ---")
    
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
                continue
                
            try:
                df = pd.read_csv(filepath)
                if df.empty or "is_correct" not in df.columns:
                    continue
                
                score = df["is_correct"].mean()
                scores[model_key][task] = round(score, 4)
                
            except Exception as e:
                pass

    return scores

def generate_markdown(scores):
    headers = ["Model"] + TASKS
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    
    for model in MODELS.keys():
        row = [model]
        for task in TASKS:
            val = scores.get(model, {}).get(task, "N/A")
            if isinstance(val, (float, int)):
                row.append(f"{val:.2%}")
            else:
                row.append(str(val))
        lines.append("| " + " | ".join(row) + " |")
    
    table_str = "\n".join(lines)
    print("\n" + table_str)

if __name__ == "__main__":
    scores = aggregate_results()
    generate_markdown(scores)
