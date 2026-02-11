import os
import pandas as pd
import glob
import shutil

RESULTS_DIR = "results"
GOLDEN_DIR = os.path.join(RESULTS_DIR, "golden")
DETAILS_DIR = os.path.join(GOLDEN_DIR, "details")

def main():
    os.makedirs(DETAILS_DIR, exist_ok=True)
    
    # 1. Scan all CSV files
    files = glob.glob(os.path.join(RESULTS_DIR, "*_baseline_*.csv"))
    
    data = []
    
    print(f"Found {len(files)} result files.")
    
    for f in files:
        if "golden" in f: continue
        
        try:
            df = pd.read_csv(f)
            if len(df) == 0: continue
            
            # Extract metadata from filename or content
            # Filename format: {type}_baseline_{task}_{model}.csv
            basename = os.path.basename(f)
            parts = basename.replace(".csv", "").split("_baseline_")
            run_type = parts[0]
            rest = parts[1]
            
            # Task is usually the first part of 'rest', but checking known tasks is safer
            known_tasks = ["classification", "math", "qa", "extraction", "analysis", "rag", "synthesis", "agentic", "code"]
            task = "unknown"
            model_part = rest
            
            for t in known_tasks:
                if rest.startswith(t):
                    task = t
                    model_part = rest[len(t)+1:] # +1 for underscore
                    break
            
            # Normalize model name
            # 1. Standardize separators
            model = model_part.replace(":", "_").replace("/", "_")
            # 2. Strip common suffixes to merge rows
            if model.endswith("_latest"):
                model = model[:-7] # Remove _latest
            # 3. Handle specific version merges if safe (e.g. 3b -> base)
            # For now, just merging _latest is the biggest win.

            
            # Calculate metrics
            count = len(df)
            score = 0
            if "is_correct" in df.columns:
                score = df["is_correct"].mean()
            
            # Timestamp (from file modified time if not in df, but prefer df)
            timestamp = os.path.getmtime(f)
            if "timestamp" in df.columns:
                 # Try to parse last timestamp
                 try:
                     timestamp = pd.to_datetime(df["timestamp"].iloc[-1]).timestamp()
                 except: pass

            data.append({
                "file": f,
                "model": model,
                "task": task,
                "type": run_type,
                "count": count,
                "score": score,
                "timestamp": timestamp,
                "df": df
            })
            
        except Exception as e:
            print(f"Error skipping {f}: {e}")

    # 2. Group and Select Best
    # Strategy: Group by (Model, Task). score = max(score), prefer higher count.
    # Actually, user said: "most complete and robust".
    # Primary Key: Count (Sample Size). Secondary: Score (assuming robustness).
    
    df_meta = pd.DataFrame(data)
    if len(df_meta) == 0:
        print("No valid data found.")
        return

    best_runs = []
    
    groups = df_meta.groupby(["model", "task"])
    
    for (model, task), group in groups:
        # Sort by count (desc), then timestamp (desc) -> Prefer largest, then newest
        sorted_group = group.sort_values(by=["count", "timestamp"], ascending=[False, False])
        best = sorted_group.iloc[0]
        best_runs.append(best)
        
        # specific file logic
        src_file = best["file"]
        # Normalize model name for filename
        safe_model = model.replace(":", "_").replace("/", "_")
        dest_filename = f"{safe_model}_{task}.csv"
        dest_path = os.path.join(DETAILS_DIR, dest_filename)
        
        # Copy to golden/details
        best["df"].to_csv(dest_path, index=False)
        print(f"Selected {task} for {model}: {best['count']} rows, {best['score']:.2%} (Source: {os.path.basename(src_file)})")

    # 3. Generate Summary Matrix
    summary_data = []
    for run in best_runs:
        summary_data.append({
            "Model": run["model"],
            "Task": run["task"],
            "Score": run["score"],
            "Samples": run["count"]
        })
        
    df_summary = pd.DataFrame(summary_data)
    
    # Pivot?
    if not df_summary.empty:
        pivot = df_summary.pivot(index="Model", columns="Task", values="Score")
        pivot_count = df_summary.pivot(index="Model", columns="Task", values="Samples")
        
        # Save CSVs
        pivot.to_csv(os.path.join(GOLDEN_DIR, "summary_scores.csv"))
        pivot_count.to_csv(os.path.join(GOLDEN_DIR, "summary_counts.csv"))
        
        # Save Markdown with clean formatting
        with open(os.path.join(GOLDEN_DIR, "SUMMARY.md"), "w") as f:
            f.write("# Benchmark Golden Summary\n\n")
            
            f.write("## Accuracy Scores\n")
            # Format as percentage strings, fill NaN with "-"
            pivot_formatted = pivot.apply(lambda x: x.map(lambda v: f"{v:.1%}" if pd.notnull(v) else "-"))
            f.write(pivot_formatted.to_markdown())
            
            f.write("\n\n## Sample Counts\n")
            # Fill NaN with 0 and convert to int for clean display
            pivot_count_formatted = pivot_count.fillna(0).astype(int)
            # Replace 0 with "-" for cleaner look if preferred, or keep 0
            f.write(pivot_count_formatted.to_markdown())
            
        print(f"\nSummary generated in {GOLDEN_DIR}")
        print(pivot)

if __name__ == "__main__":
    main()
