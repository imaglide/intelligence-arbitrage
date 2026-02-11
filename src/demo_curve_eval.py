"""
Demo Curve Evaluation: Tests whether MIPROv2 chose the wrong demo count.

For each (model, task) program with 5 demos, evaluates at n=0,1,2,3,4,5
demos while keeping optimized instructions fixed. Also runs raw baseline
(no optimization) to decompose: was it instructions or demos that hurt?

Usage:
    python3 -m src.demo_curve_eval [--limit N] [--dry-run]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional

import dspy
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.local_eval import setup_local_lm
from src.tasks.loaders import get_loader
from src.results_writer import safe_write_csv, get_run_metadata

import warnings
warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")


@dataclass
class EvalCase:
    model_ollama: str       # ollama tag e.g. "phi4:latest"
    model_label: str        # program prefix e.g. "phi4"
    task: str
    phase1_delta: float     # known delta from Phase 1


# Testable cases: negative/low delta + 5 demos (local models)
EVAL_CASES = [
    # Negative deltas (the core question)
    EvalCase("phi4:latest",     "phi4",     "analysis",  -4.0),
    EvalCase("llama3.2",        "llama3.2", "synthesis", -0.3),
    # Near-zero (did demos barely help or slightly hurt?)
    EvalCase("phi4:latest",     "phi4",     "synthesis", +1.4),
    EvalCase("phi4:latest",     "phi4",     "rag",       +2.0),
    # Positive controls (large delta — were all 5 demos needed?)
    EvalCase("phi4:latest",     "phi4",     "math",      +36.0),
    EvalCase("llama3.2",        "llama3.2", "math",      +26.0),
]


def load_program_with_n_demos(signature_class, program_path: str, n_demos: int):
    """Load an optimized program but only use the first n demonstrations."""
    with open(program_path) as f:
        saved_state = json.load(f)

    inner = saved_state.get("self", saved_state)
    ext_sig = inner.get("extended_signature", inner.get("signature", {}))
    all_demos = inner.get("demos", [])

    program = dspy.ChainOfThought(signature_class)

    state_3x = {
        "predict": {
            "traces": [],
            "train": [],
            "demos": all_demos[:n_demos],
            "signature": ext_sig,
            "lm": None
        }
    }
    program.load_state(state_3x)
    return program


def get_baseline_program(task_name, signature_class):
    """Create a raw baseline program (no optimization)."""
    if task_name in ["classification", "extraction"]:
        return dspy.Predict(signature_class)
    return dspy.ChainOfThought(signature_class)


def evaluate_program(program, test_data, metric_fn, max_examples=50):
    """Run program on test data and return accuracy."""
    correct = 0
    total = 0
    errors = 0

    for example in test_data[:max_examples]:
        try:
            kwargs = example.inputs().toDict()
            pred = program(**kwargs)
            score = metric_fn(example, pred)
            if score:
                correct += 1
            total += 1
        except Exception:
            errors += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, total, errors


def run_demo_curve(case: EvalCase, limit: int = 50, dry_run: bool = False):
    """Run full demo curve for one (model, task) case."""
    program_path = f"results/optimized_programs/{case.model_label}_{case.task}_mipro_v2.json"
    if not os.path.exists(program_path):
        print(f"  SKIP: {program_path} not found")
        return None

    # Load task
    loader = get_loader(case.task)
    signature_class = loader.get_signature()
    metric_fn = loader.get_metric()
    test_data = loader.get_eval_data(limit=limit)

    if dry_run:
        test_data = test_data[:5]

    # Setup model
    lm = setup_local_lm(case.model_ollama, temperature=0.0)
    dspy.configure(lm=lm)

    results = []

    # 1. Raw baseline (no optimization at all)
    print(f"    baseline (raw)...", end=" ", flush=True)
    t0 = time.time()
    baseline_prog = get_baseline_program(case.task, signature_class)
    acc, total, errs = evaluate_program(baseline_prog, test_data, metric_fn, limit)
    elapsed = time.time() - t0
    print(f"{acc:.0%} ({elapsed:.0f}s)")
    results.append({
        "model": case.model_label,
        "task": case.task,
        "condition": "baseline",
        "n_demos": -1,
        "accuracy": acc,
        "n_examples": total,
        "n_errors": errs,
        "elapsed_s": elapsed,
        "phase1_delta": case.phase1_delta,
    })

    # 2. Optimized with 0..5 demos
    with open(program_path) as f:
        max_demos = len(json.load(f).get("self", {}).get("demos", []))

    for n in range(0, max_demos + 1):
        print(f"    optimized n={n}...", end=" ", flush=True)
        t0 = time.time()
        prog = load_program_with_n_demos(signature_class, program_path, n)
        acc, total, errs = evaluate_program(prog, test_data, metric_fn, limit)
        elapsed = time.time() - t0
        delta_vs_baseline = acc - results[0]["accuracy"]
        print(f"{acc:.0%} (delta={delta_vs_baseline:+.0%}, {elapsed:.0f}s)")
        results.append({
            "model": case.model_label,
            "task": case.task,
            "condition": f"optimized_n{n}",
            "n_demos": n,
            "accuracy": acc,
            "n_examples": total,
            "n_errors": errs,
            "elapsed_s": elapsed,
            "phase1_delta": case.phase1_delta,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Demo Curve Evaluation")
    parser.add_argument("--limit", type=int, default=50, help="Examples per evaluation")
    parser.add_argument("--dry-run", action="store_true", help="5 examples only")
    parser.add_argument("--cases", nargs="+", help="Subset: 'phi4:analysis' etc.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing result file")
    args = parser.parse_args()

    cases = EVAL_CASES
    if args.cases:
        filtered = []
        for spec in args.cases:
            model, task = spec.split(":")
            for c in EVAL_CASES:
                if c.model_label == model and c.task == task:
                    filtered.append(c)
        cases = filtered

    limit = 5 if args.dry_run else args.limit

    print(f"Demo Curve Evaluation")
    print(f"  Cases: {len(cases)}")
    print(f"  Examples per eval: {limit}")
    print(f"  Demo counts: baseline + [0, 1, 2, 3, 4, 5]")
    print()

    all_results = []

    # Group by model to minimize Ollama model swaps
    from itertools import groupby
    sorted_cases = sorted(cases, key=lambda c: c.model_ollama)
    for model_tag, group in groupby(sorted_cases, key=lambda c: c.model_ollama):
        group_list = list(group)
        print(f"=== {model_tag} ({len(group_list)} tasks) ===")

        for case in group_list:
            print(f"  [{case.model_label} {case.task}] Phase 1 delta={case.phase1_delta:+.1f}")
            results = run_demo_curve(case, limit=limit, dry_run=args.dry_run)
            if results:
                all_results.extend(results)
            print()

    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        output_dir = "results/demo_curves"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "demo_curve_results.csv")
        metadata = get_run_metadata(
            script="src/demo_curve_eval.py",
            limit=args.limit, dry_run=args.dry_run,
            cases=[f"{c.model_label}:{c.task}" for c in cases],
        )
        safe_write_csv(df, output_path, metadata, force=args.force or args.dry_run)
        print(f"\nResults saved to {output_path}")

        # Summary table
        print(f"\n{'='*80}")
        print("SUMMARY: Accuracy by demo count")
        print(f"{'='*80}")
        print(f"{'Model':<10} {'Task':<12} {'P1Δ':>5} {'base':>6} {'n=0':>6} {'n=1':>6} {'n=2':>6} {'n=3':>6} {'n=4':>6} {'n=5':>6} {'Best n':>7}")
        print("-" * 80)

        for (model, task), group_df in df.groupby(["model", "task"]):
            baseline_acc = group_df[group_df["condition"] == "baseline"]["accuracy"].values[0]
            p1_delta = group_df["phase1_delta"].values[0]

            row = f"{model:<10} {task:<12} {p1_delta:>+5.1f} {baseline_acc:>5.0%}"
            best_n = -1
            best_acc = baseline_acc

            for n in range(6):
                opt_rows = group_df[group_df["n_demos"] == n]
                if not opt_rows.empty:
                    acc = opt_rows["accuracy"].values[0]
                    delta = acc - baseline_acc
                    row += f" {acc:>5.0%}"
                    if acc > best_acc:
                        best_acc = acc
                        best_n = n
                else:
                    row += f"  {'?':>4}"

            best_label = f"n={best_n}" if best_n >= 0 else "base"
            row += f"  {best_label:>5}"
            print(row)

        # Answer the question
        print(f"\n{'='*80}")
        print("ANALYSIS: Did wrong demo count cause negative deltas?")
        print(f"{'='*80}")
        for (model, task), group_df in df.groupby(["model", "task"]):
            p1_delta = group_df["phase1_delta"].values[0]

            baseline_acc = group_df[group_df["condition"] == "baseline"]["accuracy"].values[0]
            opt_n0 = group_df[group_df["n_demos"] == 0]
            opt_n5 = group_df[group_df["n_demos"] == 5]

            n0_acc = opt_n0["accuracy"].values[0] if not opt_n0.empty else baseline_acc
            n5_acc = opt_n5["accuracy"].values[0] if not opt_n5.empty else baseline_acc
            instr_effect = n0_acc - baseline_acc
            demo_effect = n5_acc - n0_acc

            # Find best n (including n=0 which means "instructions only")
            best_n = -1
            best_acc = baseline_acc
            for n in range(6):
                opt = group_df[group_df["n_demos"] == n]
                if not opt.empty and opt["accuracy"].values[0] > best_acc:
                    best_acc = opt["accuracy"].values[0]
                    best_n = n

            # Find worst demo count
            worst_n = -1
            worst_acc = 1.0
            for n in range(6):
                opt = group_df[group_df["n_demos"] == n]
                if not opt.empty and opt["accuracy"].values[0] < worst_acc:
                    worst_acc = opt["accuracy"].values[0]
                    worst_n = n

            flag = "NEGATIVE" if p1_delta < 0 else ("NEAR-ZERO" if p1_delta <= 2 else "POSITIVE")
            print(f"\n  {model} {task} (Phase 1: {p1_delta:+.1f}pp) [{flag}]")
            print(f"    Baseline (raw):           {baseline_acc:.0%}")
            print(f"    Instructions only (n=0):  {n0_acc:.0%}  (instruction effect: {instr_effect:+.0%})")
            print(f"    Full optimized (n=5):     {n5_acc:.0%}  (demo effect: {demo_effect:+.0%})")

            if instr_effect < -0.02 and demo_effect >= 0:
                print(f"    CAUSE: Instructions hurt ({instr_effect:+.0%}), demos were neutral/helpful")
            elif instr_effect >= -0.02 and demo_effect < -0.02:
                print(f"    CAUSE: Demos hurt ({demo_effect:+.0%}) — instructions were fine!")
                if best_n >= 0 and best_n < 5:
                    print(f"    FIX: n={best_n} would have scored {best_acc:.0%} (vs {n5_acc:.0%} at n=5)")
                elif best_n == -1:
                    print(f"    FIX: n=0 (instructions only) would have been best")
            elif instr_effect < -0.02 and demo_effect < -0.02:
                print(f"    CAUSE: Both instructions AND demos hurt — double penalty")
            elif n5_acc > baseline_acc + 0.02:
                if best_n >= 0 and best_n < 5:
                    print(f"    NOTE: Helped, but n={best_n} ({best_acc:.0%}) beats n=5 ({n5_acc:.0%})")
                else:
                    print(f"    All 5 demos contributed — n=5 was correct")
            else:
                print(f"    Optimization was neutral (±2pp)")


if __name__ == "__main__":
    main()
