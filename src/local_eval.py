import argparse
import dspy
import pandas as pd
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.config import Config
import warnings
# Suppress noisy Pydantic warnings appearing in dspy/litellm interaction
warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")
# Removed legacy imports

def setup_local_lm(model_name, temperature=0.0):
    """Configure the DSPy LM for local Ollama."""
    return dspy.OpenAI(
        model=model_name, 
        api_base=f"{Config.OLLAMA_BASE_URL}/v1",
        api_key="ollama",
        temperature=temperature,
        model_type="chat"
    )

def run_local_evaluation(model_name, task_type, limit=None, dry_run=False, program_path=None, temperature=0.0, run_id=None):
    print(f"\n--- Starting Local Evaluation: {model_name} on {task_type} ---")
    
    # Strictly respect cache setting
    os.environ["DSP_CACHEBOOL"] = str(Config.DSP_CACHEBOOL)

    if dry_run:
        print(">> DRY RUN MODE: Limiting to 5 examples.")
        limit = 5

    # 1. Setup Task
    from src.tasks.loaders import get_loader
    
    try:
        loader = get_loader(task_type)
        Signature = loader.get_signature()
        metric_fn = loader.get_metric()
    except ValueError as e:
        print(f"Task setup error: {e}")
        return

    # Load Data
    try:
        dataset = loader.get_eval_data(limit=limit) # Using correct eval split (test/val)
    except Exception as e:
        print(f"Error loading data for {task_type}: {e}")
        return

    print(f"Loaded {len(dataset)} examples.")

    # 2. Setup Model
    try:
        if "gpt" in model_name or "claude" in model_name:
            from src.api_eval import setup_lm
            lm = setup_lm(model_name, temperature=temperature)
        else:
            lm = setup_local_lm(model_name, temperature=temperature)
        dspy.configure(lm=lm)
    except Exception as e:
        print(f"Failed to configure Local LM: {e}")
        return

    # 3. Setup Program
    def get_module_for_task(task_name, signature):
        """
        Selects the baseline architecture.
        Complex reasoning tasks get CoT to prevent strawman baselines.
        Strict output tasks get Predict to prevent hallucination.
        """
        if task_name in ["classification", "extraction"]:
            return dspy.Predict(signature)
        else:
            # Includes: math, qa, agentic, code, rag, synthesis, analysis
            return dspy.ChainOfThought(signature)

    try:
        if program_path and os.path.exists(program_path):
            print(f"Loading optimized program from {program_path}...")
            # For dspy.Module.load to work, we need an instance of the class first usually,
            # OR we depend on how it was saved.
            # optimize.py saves using .save(), which expects .load() on an instance.
            # We assume it's a ChainOfThought or similar.
            
            # Reconstruct the structure used in optimize.py
            # In optimize.py: program = dspy.ChainOfThought(signature)
            # So we should be able to load into that.
            program = dspy.ChainOfThought(Signature)
            program.load(program_path)
        else:
            if program_path:
                print(f"Warning: Program path {program_path} does not exist. Using baseline.")
            # Dynamically create module with correct signature
            program = get_module_for_task(task_type, Signature)
    except Exception as e:
        print(f"Failed to initialize program for {task_type}: {e}")
        return
    
    # 4. Run Evaluation Loop
    results = []
    
    for i, example in enumerate(dataset):
        print(f"Processing example {i+1}/{len(dataset)}...")
        start_time = datetime.now()
        
        try:
            # Predict
            kwargs = example.inputs().toDict()
            pred = program(**kwargs)
            
            # Ground truth handling - depends on task structure
            # DSPy keys might vary, but example usually has the answer field
            
            # Use modular metric
            is_correct = False
            try:
                is_correct = metric_fn(example, pred)
            except Exception as metric_err:
                print(f"Metric error: {metric_err}")
                is_correct = False
            
            # Extract basic prediction text for logging
            # Try common output fields
            prediction_text = "N/A"
            if hasattr(pred, "answer"): prediction_text = pred.answer
            elif hasattr(pred, "label"): prediction_text = pred.label
            elif hasattr(pred, "entities"): prediction_text = pred.entities
            elif hasattr(pred, "summary"): prediction_text = pred.summary
            elif hasattr(pred, "completion"): prediction_text = pred.completion
            elif hasattr(pred, "reasoning"): prediction_text = f"{pred.reasoning} -> {getattr(pred, 'answer', '')}"
            
            ground_truth_text = "N/A"
            if hasattr(example, "answer"): ground_truth_text = example.answer
            elif hasattr(example, "label"): ground_truth_text = example.label
            elif hasattr(example, "entities"): ground_truth_text = example.entities
            elif hasattr(example, "summary"): ground_truth_text = example.summary
            elif hasattr(example, "canonical_solution"): ground_truth_text = example.canonical_solution

            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            result = {
                "experiment_id": f"{model_name}_local_baseline_{task_type}",
                "model": model_name,
                "task": task_type,
                "example_id": i,
                "prediction": str(prediction_text),
                "ground_truth": str(ground_truth_text),
                "is_correct": is_correct,
                "latency_ms": latency,
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
            
            if dry_run:
                print(f"Dry Run Result: {result}")

        except Exception as e:
            print(f"Error on example {i}: {e}")
            results.append({
                "experiment_id": f"{model_name}_local_baseline_{task_type}",
                "model": model_name,
                "task": task_type,
                "example_id": i,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

    # 5. Save Results
    df = pd.DataFrame(results)
    
    # Redirect dry-run to separate folder to prevent overwriting real results
    if dry_run:
        output_dir = os.path.join(Config.RESULTS_DIR, "dry_run")
    else:
        output_dir = Config.RESULTS_DIR
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Sanitize model name for filename
    safe_model_name = model_name.replace(":", "_").replace("/", "_")
    run_type = "optimized" if program_path else "baseline"
    
    if run_id:
        filename = f"local_{run_type}_{task_type}_{run_id}.csv"
    else:
        filename = f"local_{run_type}_{task_type}_{safe_model_name}.csv"
        
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    if "is_correct" in df.columns:
        accuracy = df["is_correct"].mean()
        print(f"Approx Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Ollama model tag (e.g., llama3.2:3b)")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., classification, qa, extraction)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--dry-run", action="store_true", help="Run 5 examples to test")
    parser.add_argument("--program", type=str, default=None, help="Path to optimized program JSON")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--run-id", type=str, default=None, help="Unique run identifier for output filename")
    
    args = parser.parse_args()
    
    run_local_evaluation(args.model, args.task, args.limit, args.dry_run, args.program, args.temperature, args.run_id)
