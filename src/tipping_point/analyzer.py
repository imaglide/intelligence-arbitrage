import pandas as pd
import json
import os
import sys
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.results_writer import safe_write_csv, get_run_metadata

from src.tipping_point.optimizer import TippingPointCurve
from src.tipping_point.regime_classifier import RegimeReport, OptimizationRegime

def aggregate_results(
    curves: List[TippingPointCurve],
    regime_report: RegimeReport
) -> pd.DataFrame:
    """
    Aggregate all results into DataFrame.
    """
    rows = []
    
    # 1. Add Curve Data (Regime 2)
    for c in curves:
        for i, n in enumerate(c.example_counts):
            rows.append({
                "model": c.model_name,
                "task_group": c.task_group,
                "complexity": c.complexity,
                "num_examples": n,
                "score": c.scores[i],
                "delta": c.deltas[i],
                "is_tipping_point": n == c.tipping_point,
                "regime": OptimizationRegime.REGIME_2_OPTIMIZABLE.value
            })
            
    # 2. Add Regime 1/3 Data (Mocked as N/A points for heatmap context if needed)
    # We might want to see them in the main table
    for clf in regime_report.classifications:
        if clf.regime != OptimizationRegime.REGIME_2_OPTIMIZABLE.value:
            # Add a single row for n=0 just to show baseline?
            rows.append({
                "model": clf.model_name,
                "task_group": clf.occupation, # occupation matches task_group in curves
                "complexity": clf.complexity,
                "num_examples": 0,
                "score": clf.baseline_score,
                "delta": 0.0,
                "is_tipping_point": False,
                "regime": clf.regime
            })
            
    df = pd.DataFrame(rows)
    return df

def generate_heatmap(
    df: pd.DataFrame,
    output_path: str
):
    """
    Generate heatmap: Model vs Complexity.
    Value: Tipping Point (Int) or Regime.
    """
    # Filter for Regime 2 where we have tipping points
    # We want average tipping point per (Model, Complexity)
    
    # Pivot
    # We need one value per (Model, Complexity)
    # 1. Filter optimizable
    opt_df = df[df["regime"] == OptimizationRegime.REGIME_2_OPTIMIZABLE.value]
    
    if opt_df.empty:
        print("No optimizable data for heatmap.")
        return

    # Extract just the tipping point rows? No, the TippingPoint is a property of the Curve (Group)
    # We need to aggregate by Task Group first.
    
    # Let's simplify: Group by [model, complexity] -> Mean Tipping Point
    # But tipping point is stored on the curve object, here flattened in df.
    # We used "is_tipping_point" flag.
    # We can reconstruct: for each group, find N where is_tipping_point=True.
    
    tp_data = []
    groups = df.groupby(["model", "task_group", "complexity", "regime"])
    
    for name, group in groups:
        model, task_group, complexity, regime = name
        
        if regime != OptimizationRegime.REGIME_2_OPTIMIZABLE.value:
            val = -1 # Code for N/A
        else:
            # Find the N marked as is_tipping_point
            matches = group[group["is_tipping_point"] == True]
            if not matches.empty:
                val = matches.iloc[0]["num_examples"]
            else:
                # Max examples? Or 0?
                # If no TP detected, maybe it didn't plateau?
                val = group["num_examples"].max()
        
        tp_data.append({
            "model": model,
            "complexity": complexity,
            "tipping_point": val
        })
        
    tp_df = pd.DataFrame(tp_data)
    
    # Pivot: Index=Model, Col=Complexity
    pivot = tp_df.pivot_table(
        index="model", 
        columns="complexity", 
        values="tipping_point", 
        aggfunc="mean" # Average TP across occupations in that complexity
    )
    
    # Reorder columns manually if possible [low, medium, high, standard]
    cols = [c for c in ["low", "medium", "high", "standard"] if c in pivot.columns]
    if not cols:
        print("No columns available for heatmap.")
        return
    pivot = pivot[cols]

    if pivot.empty:
        print("No data available for heatmap.")
        return

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".1f")
    plt.title("Optimization Tipping Point (Mean Examples)")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Heatmap saved to {output_path}")

def export_results(
    curves: List[TippingPointCurve],
    regime_report: RegimeReport,
    output_dir: str,
    force: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)

    df = aggregate_results(curves, regime_report)
    metadata = get_run_metadata(
        script="src/tipping_point/analyzer.py",
        total_curves=len(curves),
    )
    safe_write_csv(df, os.path.join(output_dir, "results.csv"), metadata, force=force)
    
    generate_heatmap(df, os.path.join(output_dir, "heatmap.png"))
    
    # Summary JSON
    summary = {
        "total_curves": len(curves),
        "mean_tipping_point": float(df[df["is_tipping_point"]==True]["num_examples"].mean()) if not df.empty else 0.0
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
