import argparse
import dspy
import pandas as pd
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.config import Config

def get_completion_params(model: str, temperature: float = 0.0) -> dict:
    """Handle parameter differences between model families."""
    base_params = {"temperature": temperature}
    
    # GPT-5.2 and o-series use max_completion_tokens
    if any(x in model for x in ["gpt-5.2", "gpt-5", "o1", "o3", "o4"]):
        base_params["max_completion_tokens"] = 1024
        base_params["max_tokens"] = None
    else:
        base_params["max_tokens"] = 1024
    
    return base_params

def setup_lm(model_name, temperature=0.0):
    """Configure the DSPy LM based on the model name."""
    params = get_completion_params(model_name, temperature)
    if "gpt" in model_name:
        return dspy.OpenAI(model=model_name, api_key=Config.OPENAI_API_KEY, **params)
    elif "claude" in model_name:
        # Fallback for Claude if specific client missing, or use OpenAI client for compatible endpoints
        # Note: If dspy.Anthropic exists, use it, but keeping simple for now.
        return dspy.OpenAI(model=model_name, api_key=Config.ANTHROPIC_API_KEY, **params) 
    else:
        # Fallback
        return dspy.OpenAI(model=model_name, api_key=Config.OPENAI_API_KEY, **params)

def run_evaluation(model_name, task_type, limit=None, dry_run=False, program_path=None, temperature=0.0, run_id=None):
    print(f"\n--- Starting Evaluation: {model_name} on {task_type} ---")
    
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
        dataset = loader.get_eval_data(limit=limit) # Using correct eval split for API baselines
    except Exception as e:
        print(f"Error loading data for {task_type}: {e}")
        return

    print(f"Loaded {len(dataset)} examples.")

    # 2. Setup Model
    # 2. Setup Model
    try:
        lm = setup_lm(model_name, temperature=temperature)
        dspy.configure(lm=lm)
    except Exception as e:
        print(f"Failed to configure LM: {e}")
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
            # optimize.py saves the module. We try to load it into a COT with same signature.
            program = dspy.ChainOfThought(Signature)
            program.load(program_path)
        else:
            if program_path:
                print(f"Warning: Program path {program_path} does not exist. Using baseline.")
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
            
            # Ground truth text extraction
            ground_truth = getattr(example, 'label', None) or getattr(example, 'answer', None) or getattr(example, "entities", None) or getattr(example, "summary", None) or getattr(example, "canonical_solution", None)
            prediction = getattr(pred, 'label', None) or getattr(pred, 'answer', None) or getattr(pred, "entities", None) or getattr(pred, "summary", None) or getattr(pred, "completion", None)

            # Use modular metric
            is_correct = False
            if ground_truth is not None and prediction is not None:
                try:
                    is_correct = metric_fn(example, pred)
                except Exception as metric_err:
                    print(f"Metric error: {metric_err}")
                    is_correct = False

            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            result = {
                "experiment_id": f"{model_name}_baseline_{task_type}",
                "model": model_name,
                "task": task_type,
                "example_id": i,
                "prediction": prediction,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "latency_ms": latency,
                "input_tokens": 0,
                "output_tokens": 0,
                "api_cost_usd": 0.0,
                "timestamp": datetime.now().isoformat()
            }
            
            # Attempt to capture token usage and cost
            try:
                if hasattr(dspy.settings.lm, "history") and dspy.settings.lm.history:
                    last_interaction = dspy.settings.lm.history[-1]
                    # Check for 'usage' in various dspy history formats
                    usage = last_interaction.get("usage") or last_interaction.get("response", {}).get("usage")
                    
                    if usage:
                        input_tokens = usage.get("prompt_tokens", 0)
                        output_tokens = usage.get("completion_tokens", 0)
                        result["input_tokens"] = input_tokens
                        result["output_tokens"] = output_tokens
                        
                        # Calculate cost
                        if model_name in Config.MODEL_PRICING:
                            price_in, price_out = Config.MODEL_PRICING[model_name]
                            cost = (input_tokens / 1_000_000 * price_in) + (output_tokens / 1_000_000 * price_out)
                            result["api_cost_usd"] = cost
            except Exception as cost_err:
                 print(f"Warning: Could not calculate cost: {cost_err}")

            results.append(result)
            
            if dry_run:
                print(f"Dry Run Result: {result}")
                
        except Exception as e:
            print(f"Error on example {i}: {e}")
            results.append({
                "experiment_id": f"{model_name}_baseline_{task_type}",
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
    run_type = "optimized" if program_path else "baseline"
    
    # Use run_id if provided, otherwise default formatting
    if run_id:
        filename = f"api_{run_type}_{task_type}_{run_id}.csv"
    else:
        filename = f"api_{run_type}_{task_type}_{model_name}.csv"
        
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # Summary
    if "is_correct" in df.columns:
        accuracy = df["is_correct"].mean()
        print(f"Approx Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., gpt-4o)")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., classification, qa, extraction)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--dry-run", action="store_true", help="Run 5 examples to test")
    parser.add_argument("--program", type=str, default=None, help="Path to optimized program JSON")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--run-id", type=str, default=None, help="Unique run identifier for output filename")
    
    args = parser.parse_args()
    
    run_evaluation(args.model, args.task, args.limit, args.dry_run, args.program, args.temperature, args.run_id)
