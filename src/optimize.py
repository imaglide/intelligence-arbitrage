import argparse
import dspy
import os
import sys
import json
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, MIPROv2

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.config import Config

from src.local_eval import setup_local_lm

# Ensure directory for optimized programs exists
OPTIMIZED_DIR = os.path.join(Config.RESULTS_DIR, "optimized_programs")
os.makedirs(OPTIMIZED_DIR, exist_ok=True)

def validate_extraction(example, pred, trace=None):
    """Metric for Extraction tasks (F1-like approximate)."""
    # Logic extracted from loaders.py for standalone optimization usage if needed, 
    # but for simplicity we will try to reuse the loader's metric if possible.
    # However, since this file defines custom validated_ functions, we implement a simple version here.
    try:
        pred_text = getattr(pred, "entities", "")
        gold_text = getattr(example, "entities", "")
        # Very rough check for optimization signal: non-empty intersection of keys
        return len(set(pred_text.split()) & set(gold_text.split())) > 0
    except:
        return False

def validate_analysis(example, pred, trace=None):
    """Metric for Analysis tasks (Boolean Exact Match)."""
    try:
        return str(example.answer).strip().lower() == str(pred.answer).strip().lower()
    except:
        return False

def get_metric(task_type):
    from src.tasks.loaders import get_loader
    # Prefer using the official metric from loader to ensure consistency
    try:
        return get_loader(task_type).get_metric()
    except:
        # Fallback if needed
        if task_type == "math": return validate_math
        elif task_type == "classification": return validate_classification
        elif task_type == "qa": return validate_qa
        elif task_type == "extraction": return validate_extraction
        elif task_type == "analysis": return validate_analysis
        else:
            raise ValueError(f"No metric defined for {task_type}")

def train_optimization(model_name, task_type, optimizer_type="mipro_v2", teacher_name=None, train_size=50, dry_run=False):
    print(f"\n--- Starting Optimization: {model_name} | {task_type} | {optimizer_type} ---")
    
    # Configure global settings first
    # Strictly respect cache setting
    os.environ["DSP_CACHEBOOL"] = str(Config.DSP_CACHEBOOL)
    
    if dry_run:
        print(">> DRY RUN MODE: Reduced train set size.")
        train_size = 5 # Standardized dry-run size

    # 1. Setup Data
    from src.tasks.loaders import get_loader
    loader = get_loader(task_type)
    
    try:
        # Load training data directly from the loader
        # We assume loader.load_data gives us dspy.Example objects
        dataset = loader.get_train_data(limit=train_size)
    except Exception as e:
        print(f"Error loading data for {task_type}: {e}")
        return

    print(f"Loaded {len(dataset)} training examples.")

    # 2. Setup Model (Student)
    try:
        if "gpt" in model_name or "claude" in model_name:
            from src.api_eval import setup_lm
            student = setup_lm(model_name)
        else:
            student = setup_local_lm(model_name)
    except Exception as e:
        print(f"Failed to configure Student LM: {e}")
        return

    # 3. Setup Teacher (Optional)
    teacher = student # Default to self-optimization
    if teacher_name:
         if "gpt" in teacher_name or "claude" in teacher_name:
             # Use API setup
             from src.api_eval import setup_lm
             teacher = setup_lm(teacher_name)
         else:
             teacher = setup_local_lm(teacher_name)
    else:
        print("\n\n [WARNING] No teacher model provided. Defaulting to Student as Teacher.")
        print(" -> This may lead to poor 'Delta' for small models as they cannot generate high-quality bootstraps.")
        print(" -> Recommended: Use --teacher gpt-4o or similiar for rigorous comparison.\n\n")

    # Configure DSPy with cache setting
    # experiment=True (or implied default) allows caching, so we must be explicit if we want to disable it.
    # However, dspy.settings.configure() is often used.
    # We'll set the LM and let the env var handle the cache, but explicitly passing it is safer if supported.
    dspy.configure(lm=student)
    
    # 4. Setup Program
    signature = loader.get_signature()
    # Program is defined inside the loop or just cloned. 
    # dspy.ChainOfThought(signature) creates a module. Copied for each seed iteration below.
    
    # Classification needs options (Banking77 specific handling if not in loader)
    if task_type == "classification":
        pass

    # 5. Configure Optimizer & Loop
    metric = get_metric(task_type)
    
    best_program = None
    best_score = -1.0
    
    # Validation split for selection (Heuristic: 20% of loaded train data, min 5)
    # We loaded 'train_size' examples. Let's split them.
    if len(dataset) > 10:
        split_idx = int(0.8 * len(dataset))
        train_set = dataset[:split_idx]
        dev_set = dataset[split_idx:]
    else:
        # Too small to split robustly, use same set (risk of overfitting but better than crashing)
        train_set = dataset
        dev_set = dataset
    
    # Run 3 candidates to smooth out stochasticity (unless dry run)
    seeds = [10, 20, 30] if not dry_run else [10]
    
    for seed in seeds:
        print(f"\n>> Optimization Run (Seed {seed})")
        dspy.settings.configure(random_seed=seed)
        
        # Reset program state
        program = dspy.ChainOfThought(signature)

        if optimizer_type == "mipro_v2":
            # MIPROv2 requires a teacher to generate instructions/examples
            # Note: prompt_model is the teacher
            optimizer = MIPROv2(
                metric=metric,
                prompt_model=teacher,
                task_model=student,
                num_candidates=7, # User requested 7
                init_temperature=1.0
            )
            compile_kwargs = {
                "num_trials": 10 if not dry_run else 2, # User requested 10
                "max_bootstrapped_demos": 3,
                "max_labeled_demos": 5,
                "requires_permission_to_run": False,
                "minibatch_size": 25 if not dry_run else 1,
            }
        elif optimizer_type == "bootstrap_rs":
            # BootstrapFewShotWithRandomSearch
            # Also beneficial to clear teacher history if possible, but objects are fresh.
            optimizer = BootstrapFewShotWithRandomSearch(
                metric=metric,
                teacher_settings=dict(lm=teacher), # Explicitly pass teacher (if supported by version, else it uses dspy.teacher)
                max_bootstrapped_demos=4,
                max_labeled_demos=4,
                num_candidate_programs=10 if not dry_run else 2,
                num_threads=1
            )
            compile_kwargs = {}
        else:
            print(f"Unknown optimizer: {optimizer_type}")
            return

        print(f"Compiling with {optimizer_type} (Seed {seed})...")
        
        try:
            candidate_program = optimizer.compile(
                program,
                trainset=train_set,
                **compile_kwargs
            )
            
            # Evaluate on Dev Set
            # We can use dspy.Evaluate
            from dspy.evaluate import Evaluate
            evaluator = Evaluate(devset=dev_set, metric=metric, num_threads=1, display_progress=False)
            score = evaluator(candidate_program)
            
            print(f"  Result (Seed {seed}): Dev Score = {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_program = candidate_program
                print(f"  -> New Best Program found!")
                
        except Exception as e:
            print(f"Optimization run failed for seed {seed}: {e}")
            import traceback
            traceback.print_exc()

    if best_program:
        optimized_program = best_program
    else:
        print("All optimization runs failed.")
        return
        
    # 6. Save Program
    safe_model_name = model_name.replace(":", "_").replace("/", "_")
    filename = f"{safe_model_name}_{task_type}_{optimizer_type}.json"
    save_path = os.path.join(OPTIMIZED_DIR, filename)
    # Ensure it's a module
    if isinstance(optimized_program, dspy.Module):
        optimized_program.save(save_path)
        print(f"Optimized program saved to {save_path}")
    else:
        print(f"Error: Optimized program is not a dspy.Module, cannot save. Type: {type(optimized_program)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Student model name (Ollama tag)")
    parser.add_argument("--teacher", type=str, default=None, help="Teacher model name (optional)")
    parser.add_argument("--task", type=str, required=True, choices=["classification", "qa", "math", "extraction", "analysis", "synthesis", "agentic", "code", "rag"])
    parser.add_argument("--optimizer", type=str, default="mipro_v2", choices=["mipro_v2", "bootstrap_rs"])
    parser.add_argument("--train-size", type=int, default=50, help="Number of training examples")
    parser.add_argument("--dry-run", action="store_true", help="Run with minimal parameters")
    
    args = parser.parse_args()
    
    train_optimization(
        model_name=args.model,
        task_type=args.task,
        optimizer_type=args.optimizer,
        teacher_name=args.teacher,
        train_size=args.train_size,
        dry_run=args.dry_run
    )
