import argparse
import sys
import os
import json
import glob
from typing import List, Any, Dict, Optional
from dataclasses import dataclass

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import Config
from src.tasks.gdpval_loader import load_gdpval_dataset, get_occupation_splits, GDPvalSplit, GDPvalTask, GDPvalStudentSignature
from src.tasks.gdpval_judge import GDPvalJudge
from src.tipping_point.regime_classifier import run_regime_classification, RegimeReport, OptimizationRegime, RegimeClassification
from src.tipping_point.optimizer import TippingPointOptimizer, OptimizationConfig, TippingPointCurve
from src.tipping_point.analyzer import export_results
from src.tasks.loaders import get_loader

@dataclass
class StandardizedTaskSplit:
    name: str # task_name or occupation
    train_data: List[Any]
    test_data: List[Any]
    signature_class: Any
    metric_fn: Any
    source: str # "core" or "gdpval"
    complexity: str = "standard"

def load_standard_tasks(task_names: List[str]) -> List[StandardizedTaskSplit]:
    results = []
    for t_name in task_names:
        try:
            loader = get_loader(t_name)
            # Limit data for feasibility
            train = loader.get_train_data(limit=50) # Enough for n=8 + candidates
            test = loader.get_eval_data(limit=50)   # Enough for curves
            
            results.append(StandardizedTaskSplit(
                name=t_name,
                train_data=train,
                test_data=test,
                signature_class=loader.get_signature(),
                metric_fn=loader.get_metric(),
                source="core"
            ))
        except Exception as e:
            print(f"Failed to load core task {t_name}: {e}")
    return results

def load_gdpval_splits(limit_occupations: Optional[int] = None) -> List[StandardizedTaskSplit]:
    try:
        tasks = load_gdpval_dataset()
        splits = get_occupation_splits(tasks)
        
        if limit_occupations:
            keys = list(splits.keys())[:limit_occupations]
            splits = {k: splits[k] for k in keys}
            
        return [
            StandardizedTaskSplit(
                name=s.occupation,
                train_data=s.train_tasks,
                test_data=s.test_tasks,
                signature_class=GDPvalStudentSignature,
                metric_fn=None, # Optimizer creates generic metric wrapping judge
                source="gdpval",
                complexity=s.complexity.value if hasattr(s.complexity, 'value') else str(s.complexity)
            )
            for s in splits.values()
        ]
    except Exception as e:
        print(f"Failed to load GDPval: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="DSPy Tipping Point Experiment Runner")
    # Using OLLAMA_MODELS by default if TIPPING_POINT keys not found
    parser.add_argument("--models", nargs="+", default=Config.TIPPING_POINT_CONFIG["models"], help="Models to run")
    parser.add_argument("--example-counts", nargs="+", type=int, default=Config.TIPPING_POINT_CONFIG["example_counts_per_occupation"], help="Example counts to test")
    parser.add_argument("--output-dir", default="results/tipping_point", help="Output directory")
    
    # Selection
    parser.add_argument("--core-tasks", action="store_true", help="Include core tasks")
    parser.add_argument("--gdpval", action="store_true", default=False, help="Include GDPval tasks")
    parser.add_argument("--tasks", nargs="+", help="Specific core tasks to run")
    parser.add_argument("--limit-groups", type=int, default=None, help="Limit number of task groups (occupations/tasks)")
    
    args = parser.parse_args()
    
    # Default to GDPval if nothing specified, for backward compat?
    # Or default to Config tasks?
    # Spec implies we run Tipping Point.
    if not args.core_tasks and not args.gdpval:
        print("Defaulting to GDPval (enable --core-tasks to run standard benchmark)")
        args.gdpval = True

    print(f"--- Starting Unified Tipping Point Experiment ---")
    print(f"Models: {args.models}")
    print(f"Counts: {args.example_counts}")
    print(f"Output: {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # 1. Load Data
    all_splits = []
    
    if args.core_tasks:
        target_tasks = args.tasks if args.tasks else Config.TASKS
        print(f"Loading Core Tasks: {target_tasks}")
        all_splits.extend(load_standard_tasks(target_tasks))
        
    if args.gdpval:
        print(f"Loading GDPval Tasks...")
        all_splits.extend(load_gdpval_splits(args.limit_groups))
        
    if not all_splits:
        print("No tasks loaded.")
        return
        
    print(f"Total Task Groups: {len(all_splits)}")
    
    # 2. Initialize Shared Components
    # Judge is only needed for GDPval metric creation
    judge = GDPvalJudge(model="gpt-4o") # Cost incurred only if used
    optimizer_module = TippingPointOptimizer(judge=judge, checkpoint_dir=checkpoints_dir)
    
    curves = []
    regime_results = []
    
    # 3. Main Execution Loop
    for model in args.models:
        print(f"\n=== Model: {model} ===")
        
        for split in all_splits:
            print(f"\n>> Group: {split.name} ({split.source})")
            
            # A. Baseline / Regime Check
            # We run the 0-shot baseline to determine regime
            # Optimizer can do this, but we want to log it specifically.
            # Using optimizer to run n=0 config first?
            # Or simplified baseline run.
            
            # Let's use the optimizer's run_optimization for n=0 to get baseline.
            # Note: run_optimization returns OptimizationResult check baseline_scores.
            
            print("  Measuring Baseline...")
            try:
                base_cfg = OptimizationConfig(
                    model_name=model,
                    task_ids=[],
                    num_examples=0,
                    num_candidates=1, seeds=[10], # 1 seed for baseline check
                    task_source=split.source
                )
                
                # We need to construct metric_fn correctly
                metric = split.metric_fn # None for gdpval (created by opt)
                
                res = optimizer_module.run_optimization(
                    config=base_cfg,
                    train_data=split.train_data,
                    test_data=split.test_data,
                    signature_class=split.signature_class,
                    metric_fn=metric
                )
                
                if res:
                    baseline_score = res.mean_baseline # 0-shot score
                    
                    # Determine Regime
                    # Thresholds: < 0.10 (Hard), > 0.70 (Solved) ?
                    # For standard classification (Banking77), 0.70 is likely achievable.
                    # Should we skip 'Solved' tasks?
                    # "Tipping point" implies we look for improvement.
                    # If score is 0.95, optimizing might not show much.
                    # Retaining < 0.10 check.
                    
                    regime = OptimizationRegime.REGIME_2_OPTIMIZABLE
                    if baseline_score < 0.10: regime = OptimizationRegime.REGIME_1_TOO_HARD
                    elif baseline_score > 0.85: regime = OptimizationRegime.REGIME_3_ALREADY_SOLVED # Higher threshold for core tasks?
                    
                    print(f"  Baseline: {baseline_score:.4f} -> {regime.value}")
                    
                    # Store regime result mock
                    regime_results.append(RegimeClassification(
                        model_name=model,
                        task_id=split.name,
                        occupation=split.name,
                        complexity=split.complexity,
                        baseline_score=baseline_score,
                        regime=regime.value,
                        rationale="Unified Runner Baseline",
                        evaluated_at="now"
                    ))
                    
                    if regime != OptimizationRegime.REGIME_2_OPTIMIZABLE:
                        print(f"  Skipping Optimization ({regime.value})")
                        continue
                        
                    # B. Run Optimization Curve
                    print(f"  Running Curves: {args.example_counts}")
                    curve = optimizer_module.run_tipping_point_experiment(
                        model_name=model,
                        train_data=split.train_data,
                        test_data=split.test_data,
                        signature_class=split.signature_class,
                        metric_fn=metric,
                        example_counts=args.example_counts,
                        group_id=split.name,
                        complexity_label=split.complexity,
                        task_source=split.source
                    )
                    curves.append(curve)
                    
            except Exception as e:
                print(f"  Error processing {split.name}: {e}")
                # import traceback
                # traceback.print_exc()

    # 4. Generate Report
    print("\ngenerating Report...")
    # Wrap regime results
    dummy_report = RegimeReport(
        total_pairs=len(regime_results),
        classifications=regime_results
    )
    # Count stats
    for r in regime_results:
        if r.regime == OptimizationRegime.REGIME_1_TOO_HARD.value: dummy_report.regime_1_count += 1
        elif r.regime == OptimizationRegime.REGIME_2_OPTIMIZABLE.value: dummy_report.regime_2_count += 1
        elif r.regime == OptimizationRegime.REGIME_3_ALREADY_SOLVED.value: dummy_report.regime_3_count += 1
        
    export_results(curves, dummy_report, args.output_dir)
    print("Done!")

if __name__ == "__main__":
    main()
