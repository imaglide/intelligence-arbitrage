import os
import pandas as pd
import glob
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config

BASELINE_CSV = "results/golden/summary_scores.csv"
RESULTS_DIR = "results"
OUTPUT_FILE = "experiments/optimization_phase1/api_vs_local_optimized.md"

API_MODELS = ["gpt-4o", "gpt-5.2", "gpt-4o-mini", "gpt-5.2-chat-latest"]
LOCAL_MODELS = ["mistral", "llama3.2", "phi4", "qwen2.5_7b"]

def main():
    # 1. Load Baseline Scores
    if not os.path.exists(BASELINE_CSV):
        print(f"Error: Baseline scores not found at {BASELINE_CSV}")
        return
    
    df_baseline = pd.read_csv(BASELINE_CSV, index_col=0)
    print("Loaded Baseline Scores.")

    # 2. Load Optimized Local Scores
    files = glob.glob(os.path.join(RESULTS_DIR, "local_optimized_*.csv"))
    
    optimized_data = []
    
    for f in files:
        try:
            basename = os.path.basename(f)
            # Remove prefix/suffix
            inner = basename.replace("local_optimized_", "").replace(".csv", "")
            
            if "_optimized_" in inner:
                parts = inner.split("_optimized_")
                task = parts[0]
                model = parts[1]
                
                if model in LOCAL_MODELS:
                    df = pd.read_csv(f)
                    if len(df) > 0:
                        score = 0
                        if "is_correct" in df.columns:
                            score = df["is_correct"].mean()
                        
                        optimized_data.append({
                            "Model": model,
                            "Task": task,
                            "Score": score
                        })
        except Exception as e:
             pass

    df_opt = pd.DataFrame(optimized_data)
    
    # 3. Aggregate by Task
    all_tasks = sorted(list(set(df_baseline.columns) | set(df_opt["Task"] if not df_opt.empty else [])))
    
    markdown_lines = []
    markdown_lines.append("# API Baseline vs. Local Optimized Comparison\n")
    markdown_lines.append("> Comparing the best **unoptimized** API model performance against the best **optimized** local model performance.\n")
    
    # Collect data for all local models
    detailed_rows = []
    arbitrage_wins = []
    arbitrage_closers = []
    
    headers_detail = ["Task", "Model", "Baseline", "Optimized", "Internal Delta", "vs API Best"]
    
    for task in all_tasks:
        # 1. Identify Best API Baseline for this task
        api_best_score = -1
        api_best_model = "N/A"
        if task in df_baseline.columns:
            for model in API_MODELS:
                if model in df_baseline.index:
                    score = df_baseline.loc[model, task]
                    if pd.notnull(score) and score > api_best_score:
                        api_best_score = score
                        api_best_model = model
                        
        # 2. Iterate through ALL Local Models
        for model in LOCAL_MODELS:
            # Baseline Score
            base_score = float("nan")
            if model in df_baseline.index and task in df_baseline.columns:
                base_score = df_baseline.loc[model, task]
            
            # Optimized Score
            opt_score = float("nan")
            # Look up in df_opt
            if not df_opt.empty:
                matches = df_opt[(df_opt["Model"] == model) & (df_opt["Task"] == task)]
                if not matches.empty:
                    opt_score = matches["Score"].iloc[0]
            
            # Skip if we have NO data for this model/task pair
            if pd.isna(base_score) and pd.isna(opt_score):
                continue
                
            # Formatting scores
            base_str = f"{base_score:.1%}" if pd.notnull(base_score) else "-"
            opt_str = f"**{opt_score:.1%}**" if pd.notnull(opt_score) else "-"
            
            # Internal Delta
            delta_internal_str = "-"
            if pd.notnull(base_score) and pd.notnull(opt_score):
                diff = opt_score - base_score
                icon = "📈" if diff > 0 else "📉"
                delta_internal_str = f"{icon} {diff:+.1%}"
            elif pd.notnull(opt_score):
                delta_internal_str = "(New)"
            
            # Comparison vs API
            vs_api_str = "-"
            is_win = False
            is_close = False
            gap_closed_msg = ""
            
            if api_best_score != -1 and pd.notnull(opt_score):
                diff_api = opt_score - api_best_score
                
                if diff_api >= -0.01: # Win or Tie (within 1%)
                     icon = "🏆"
                     vs_api_str = f"{icon} {diff_api:+.1%}"
                     is_win = True
                elif diff_api >= -0.05: # Within 5%
                     icon = "✅" 
                     vs_api_str = f"{icon} {diff_api:+.1%}"
                     is_close = True
                else:
                     icon = "❌"
                     vs_api_str = f"{diff_api:+.1%}"
                     
                # Check for "Greatly Improved Gap" (Arbitrage Closer)
                # If baseline gap was huge, and opt gap is small
                if pd.notnull(base_score):
                    base_gap = base_score - api_best_score
                    opt_gap = diff_api
                    # If we closed the gap by > 50%
                    if base_gap < -0.10 and opt_gap > base_gap: # Was behind by 10%+ and improved
                         improvement = opt_gap - base_gap
                         if improvement > 0.10: # Improved by 10% absolute
                             gap_closed_msg = f"Closed gap by {improvement:.1%}"
                             is_close = True # count as interesting
            
            detailed_rows.append([task, model, base_str, opt_str, delta_internal_str, vs_api_str])
            
            # Arbitrage Logic
            if is_win:
                arbitrage_wins.append({
                    "Task": task,
                    "Model": model,
                    "Score": opt_score,
                    "API Model": api_best_model,
                    "API Score": api_best_score,
                    "Delta": opt_score - api_best_score
                })
            elif is_close or gap_closed_msg:
                 arbitrage_closers.append({
                    "Task": task,
                    "Model": model,
                    "Score": opt_score,
                    "Baseline": base_score,
                    "API Model": api_best_model,
                    "API Score": api_best_score,
                    "Note": gap_closed_msg if gap_closed_msg else "Within 5% of API"
                })

    # Sort detailed rows by Task then Model
    detailed_rows.sort(key=lambda x: (x[0], x[1]))

    # --- Generate Markdown ---
    lines = []
    lines.append("# Intelligence Arbitrage Report")
    lines.append(f"> Analysis generated: {pd.Timestamp.now()}\n")
    
    # Section 1: Arbitrage Opportunities
    lines.append("## 1. Intelligence Arbitrage (Wins & Catch-ups)")
    lines.append("Where local models either beat the API or significantly closed the gap.")
    
    if arbitrage_wins:
        lines.append("\n### 🏆 Wins (Local > API)")
        lines.append("| Task | Winner (Local) | Score | vs API Best | Delta |")
        lines.append("|---|---|---|---|---|")
        for w in arbitrage_wins:
            lines.append(f"| {w['Task']} | **{w['Model']}** | {w['Score']:.1%} | {w['API Model']} ({w['API Score']:.1%}) | +{w['Delta']:.1%} |")
    else:
        lines.append("\n*No direct wins recorded yet.*")
        
    if arbitrage_closers:
        lines.append("\n### 🚀 Gap Closers (High Improvement)")
        lines.append("| Task | Contender (Local) | Opt Score | Previous Baseline | vs API Best | Note |")
        lines.append("|---|---|---|---|---|---|")
        for c in arbitrage_closers:
             base_fmt = f"{c['Baseline']:.1%}" if pd.notnull(c['Baseline']) else "N/A"
             api_gap = c['Score'] - c['API Score']
             lines.append(f"| {c['Task']} | {c['Model']} | **{c['Score']:.1%}** | {base_fmt} | {c['API Model']} ({api_gap:+.1%}) | {c['Note']} |")
    else:
         lines.append("\n*No significant gap closers recorded yet.*")

    # Section 2: Detailed Matrix
    lines.append("\n## 2. Detailed Internal Deltas (Optimized vs Unoptimized)")
    lines.append("Performance improvement for every model/task pair.")
    
    # Pretty print detailed table
    col_widths = [0] * len(headers_detail)
    # Pre-calculate widths
    all_table_data = [headers_detail] + detailed_rows
    for row in all_table_data:
        for i, cell in enumerate(row):
             col_widths[i] = max(col_widths[i], len(str(cell)))
    
    lines.append(f"| {' | '.join([h.ljust(w) for h, w in zip(headers_detail, col_widths)])} |")
    lines.append(f"| {' | '.join(['-' * w for w in col_widths])} |")
    
    for row in detailed_rows:
        lines.append(f"| {' | '.join([str(c).ljust(w) for c, w in zip(row, col_widths)])} |")

    # Section 3: Regression Analysis
    lines.append("\n## 3. Analysis of Regressions (Why did some scores drop?)")
    lines.append("Investigating the negative deltas in **Code** (-17% to -21%) and **Agentic** tasks.")
    lines.append("- **Cause:** The Optimized Programs for these tasks (`llama3.2_code_mipro_v2.json`, `mistral_code_mipro_v2.json`) were found to be **empty** (containing 0 few-shot examples).")
    lines.append("- **Reason:** The small local models likely failed to generate *any* passing code during the strict training phase (MBPP dataset). Because the optimizer couldn't find any high-scoring examples to use as demos, it defaulted to a 0-shot prompt.")
    lines.append("- **Conclusion:** Evaluation-based optimization (Student-as-Teacher) fails for hard tasks where the student cannot solve the problem 0-shot. These tasks require a **Teacher (GPT-4)** to generate the initial training traces.")

    content = "\n".join(lines)
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write(content)
        
    print(f"Report generated at {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
