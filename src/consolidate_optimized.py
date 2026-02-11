import os
import glob
import pandas as pd
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config

BASELINE_CSV = "results/golden/summary_scores.csv"
RESULTS_DIR = "results"
OUTPUT_FILE = "experiments/optimization_phase1/summary_table_optimized.md"

def main():
    # 1. Load Baseline Scores
    if not os.path.exists(BASELINE_CSV):
        print(f"Error: Baseline scores not found at {BASELINE_CSV}")
        return
    
    df_baseline = pd.read_csv(BASELINE_CSV, index_col=0)
    print("Loaded Baseline Scores.")

    # 2. Scan Optimized Results
    # Pattern: results/local_optimized_{task}_{run_id}.csv
    # But run_id is usually "optimized_{model}"
    files = glob.glob(os.path.join(RESULTS_DIR, "local_optimized_*.csv"))
    
    optimized_data = []
    
    for f in files:
        try:
            # Filename parse
            basename = os.path.basename(f)
            # Filename parse
            basename = os.path.basename(f)
            # Example: local_optimized_agentic_optimized_gpt-4o.csv
            
            # Remove prefix and suffix
            inner = basename.replace("local_optimized_", "").replace(".csv", "")
            # inner is now: agentic_optimized_gpt-4o
            
            # Split by the separator "_optimized_"
            # We expect {task}_optimized_{model}
            if "_optimized_" in inner:
                parts = inner.split("_optimized_")
                task_part = parts[0]
                model_name = parts[1]
            else:
                print(f"Skipping malformed filename: {basename}")
                continue
            
            # Read CSV
            df = pd.read_csv(f)
            if len(df) == 0: continue
            
            # Calculate Score
            score = 0
            if "is_correct" in df.columns:
                score = df["is_correct"].mean()
            
            optimized_data.append({
                "Model": model_name,
                "Task": task_part,
                "Optimized_Score": score,
                "Count": len(df)
            })
        except Exception as e:
            print(f"Error parsing {f}: {e}")

    df_opt = pd.DataFrame(optimized_data)
    
    if df_opt.empty:
        print("No optimized results found.")
        return

    # Deduplicate: Keep entry with highest count (completeness) for each Model/Task
    # Sort by Count descending, then drop duplicates
    df_opt = df_opt.sort_values(by="Count", ascending=False)
    df_opt = df_opt.drop_duplicates(subset=["Model", "Task"], keep="first")

    # Pivot Optimized Data
    pivot_opt = df_opt.pivot(index="Model", columns="Task", values="Optimized_Score")
    
    # 3. Generate Comparison Table
    # We want a table where cells are "Opt% (Diff%)" e.g. "85.0% (+5.0%)"
    
    # Align columns and rows
    all_models = sorted(list(set(df_baseline.index) | set(pivot_opt.index)))
    all_tasks = sorted(list(set(df_baseline.columns) | set(pivot_opt.columns)))
    
    markdown_lines = []
    markdown_lines.append("# Optimization Results Summary\n")
    markdown_lines.append(f"**Generated:** {pd.Timestamp.now()}\n")
    
    # Header
    header = "| Model | " + " | ".join(all_tasks) + " |"
    markdown_lines.append(header)
    markdown_lines.append("|---" * (len(all_tasks) + 1) + "|")
    
    for model in all_models:
        row = [f"**{model}**"]
        for task in all_tasks:
            base_score = float("nan")
            opt_score = float("nan")
            
            if model in df_baseline.index and task in df_baseline.columns:
                base_score = df_baseline.loc[model, task]
                
            if model in pivot_opt.index and task in pivot_opt.columns:
                opt_score = pivot_opt.loc[model, task]
            
            # Format Cell
            cell_text = "-"
            
            if pd.notnull(opt_score):
                opt_str = f"{opt_score:.1%}"
                
                if pd.notnull(base_score):
                    delta = opt_score - base_score
                    icon = "🔺" if delta > 0 else "🔻" if delta < 0 else "▫️"
                    delta_str = f"{icon} {delta:+.1%}"
                    cell_text = f"**{opt_str}** <br> {delta_str}"
                else:
                    cell_text = f"**{opt_str}** <br> (New)"
            elif pd.notnull(base_score):
                 cell_text = f"{base_score:.1%} <br> (Pending)"
            
            row.append(cell_text)
        markdown_lines.append("| " + " | ".join(row) + " |")

    # Write to file
    output_dir = os.path.dirname(OUTPUT_FILE)
    os.makedirs(output_dir, exist_ok=True)
    
    content = "\n".join(markdown_lines)
    with open(OUTPUT_FILE, "w") as f:
        f.write(content)
        
    print(f"Comparison summary generated at {OUTPUT_FILE}")
    print(pivot_opt)

if __name__ == "__main__":
    main()
