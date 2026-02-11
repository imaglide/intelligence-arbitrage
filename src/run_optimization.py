
import argparse
import subprocess
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config

def get_config_tasks():
    # Return tasks from config, ensuring we get the latest list
    return Config.TASKS

def run_optimization_loop(models, tasks, teacher=None, dry_run=False):
    print(f"--- Starting Optimization Matrix ---")
    print(f"Models: {models}")
    print(f"Tasks: {tasks}")
    print(f"Teacher: {teacher if teacher else 'Self (Student)'}")
    print(f"Dry Run: {dry_run}")

    for model in models:
        # Resolve model alias if present in Config
        model_tag = Config.OLLAMA_MODELS.get(model, model)
        
        for task in tasks:
            print(f"\n[Optimization] Model: {model} ({model_tag}) | Task: {task}")
            
            cmd = [
                "python3", "src/optimize.py",
                "--model", model_tag,
                "--task", task,
                "--optimizer", "mipro_v2" 
            ]
            
            if teacher:
                cmd.extend(["--teacher", teacher])
                
            if dry_run:
                cmd.append("--dry-run")
                
            try:
                # We use subprocess to run each optimization as a separate process to ensure clean state
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Failed to optimize {model} on {task}: {e}")
            except KeyboardInterrupt:
                print("\nOptimization interrupted by user.")
                return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrate DSPy Optimization Runs")
    parser.add_argument("--models", nargs="+", required=True, help="List of models to optimize")
    parser.add_argument("--tasks", nargs="+", default=["all"], help="List of tasks or 'all'")
    parser.add_argument("--teacher", type=str, help="Teacher model (optional)")
    parser.add_argument("--dry-run", action="store_true", help="Run fast dry-run for verification")
    
    args = parser.parse_args()
    
    # Resolve tasks
    if "all" in args.tasks:
        tasks = get_config_tasks()
    else:
        tasks = args.tasks
        
    run_optimization_loop(
        models=args.models,
        tasks=tasks,
        teacher=args.teacher,
        dry_run=args.dry_run
    )
