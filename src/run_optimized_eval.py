
import os
import sys
import subprocess
import argparse
import glob

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config

def get_model_tag(filename_model_part):
    """
    Map filename model part back to Ollama tag or API name.
    """
    # Direct Key Match in Config.MODELS
    if filename_model_part in Config.OLLAMA_MODELS:
        return Config.OLLAMA_MODELS[filename_model_part]
    
    # Try reversing aliases (value matching)
    for alias, tag in Config.OLLAMA_MODELS.items():
        if alias == filename_model_part:
            return tag
            
    # Fallback: assume it's the tag itself (e.g., phi4, gpt-4o)
    return filename_model_part

def run_optimized_evaluations(results_dir, dry_run=False, limit=None):
    programs_dir = os.path.join(results_dir, "optimized_programs")
    if not os.path.exists(programs_dir):
        print(f"No optimized programs found at {programs_dir}")
        return

    json_files = glob.glob(os.path.join(programs_dir, "*.json"))
    print(f"Found {len(json_files)} optimized programs.")
    
    # Store dry run results separately
    output_dir = os.path.join(results_dir, "dry_run") if dry_run else results_dir
    os.makedirs(output_dir, exist_ok=True)

    for json_file in sorted(json_files):
        filename = os.path.basename(json_file)
        # Expected format: {model}_{task}_mipro_v2.json
        
        # Heuristic parsing:
        found_task = None
        for task in Config.TASKS:
            if f"_{task}_" in filename:
                found_task = task
                break
        
        if not found_task:
            print(f"Skipping {filename}: Could not identify task.")
            continue
            
        # Extract model name: everything before _{task}_
        model_part = filename.split(f"_{found_task}_")[0]
        
        # Get actual model tag
        model_tag = get_model_tag(model_part)
        
        # Check if output already exists to avoid re-running
        # Filename logic mirrors local_eval.py
        output_filename = f"local_optimized_{found_task}_optimized_{model_part}.csv"
        output_path = os.path.join(output_dir, output_filename)
        
        if os.path.exists(output_path) and not dry_run:
            print(f"Skipping {filename}: Output already exists at {output_filename}")
            continue
        
        print(f"\nEvaluating Optimized: Model={model_tag} | Task={found_task} | File={filename}")
        
        cmd = [
            "python3", "src/local_eval.py",
            "--model", model_tag,
            "--task", found_task,
            "--program", json_file,
            "--run-id", f"optimized_{model_part}" 
        ]
        
        if limit is not None:
             cmd.extend(["--limit", str(limit)])
        
        if dry_run:
            cmd.append("--dry-run")
            
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Evaluation failed for {filename}: {e}")
        except KeyboardInterrupt:
            print("Interrupted.")
            return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of evaluation examples")
    args = parser.parse_args()
    
    run_optimized_evaluations(Config.RESULTS_DIR, dry_run=args.dry_run, limit=args.limit)
