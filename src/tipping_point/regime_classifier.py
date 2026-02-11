from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any, Optional
import os
import sys
import json
import dspy
from datetime import datetime
import pandas as pd

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import Config
from src.tasks.gdpval_loader import GDPvalTask, ComplexityBucket, GDPvalStudentSignature
from src.tasks.gdpval_judge import GDPvalJudge
from src.local_eval import setup_local_lm
from src.api_eval import setup_lm

class OptimizationRegime(Enum):
    REGIME_1_TOO_HARD = "too_hard"         # < 10%
    REGIME_2_OPTIMIZABLE = "optimizable"    # 10%-70%
    REGIME_3_ALREADY_SOLVED = "already_solved" # > 70%

@dataclass
class RegimeClassification:
    model_name: str
    task_id: str
    occupation: str
    complexity: str # String of Enum
    baseline_score: float
    regime: str # String of Enum.value
    rationale: str
    evaluated_at: str

@dataclass
class RegimeReport:
    total_pairs: int = 0
    regime_1_count: int = 0
    regime_2_count: int = 0
    regime_3_count: int = 0
    classifications: List[RegimeClassification] = field(default_factory=list)
    
    def get_optimizable_pairs(self) -> List[Tuple[str, str]]:
        """Return (model, task_id) pairs in Regime 2."""
        return [
            (c.model_name, c.task_id) 
            for c in self.classifications 
            if c.regime == OptimizationRegime.REGIME_2_OPTIMIZABLE.value
        ]
        
    def to_json(self):
        return {
            "total_pairs": self.total_pairs,
            "regime_1_count": self.regime_1_count,
            "regime_2_count": self.regime_2_count,
            "regime_3_count": self.regime_3_count,
            "classifications": [asdict(c) for c in self.classifications]
        }

    @classmethod
    def from_json(cls, data):
        report = cls()
        report.total_pairs = data.get("total_pairs", 0)
        report.regime_1_count = data.get("regime_1_count", 0)
        report.regime_2_count = data.get("regime_2_count", 0)
        report.regime_3_count = data.get("regime_3_count", 0)
        report.classifications = [RegimeClassification(**c) for c in data.get("classifications", [])]
        return report

REGIME_1_THRESHOLD: float = 0.10
REGIME_3_THRESHOLD: float = 0.70

def classify_regime(baseline_score: float) -> Tuple[OptimizationRegime, str]:
    """Classify score into regime with rationale."""
    if baseline_score < REGIME_1_THRESHOLD:
        return OptimizationRegime.REGIME_1_TOO_HARD, f"Score {baseline_score:.2f} < {REGIME_1_THRESHOLD:.2f}"
    elif baseline_score > REGIME_3_THRESHOLD:
        return OptimizationRegime.REGIME_3_ALREADY_SOLVED, f"Score {baseline_score:.2f} > {REGIME_3_THRESHOLD:.2f}"
    else:
        return OptimizationRegime.REGIME_2_OPTIMIZABLE, f"{REGIME_1_THRESHOLD:.2f} <= Score {baseline_score:.2f} <= {REGIME_3_THRESHOLD:.2f}"

def run_regime_classification(
    models: List[str],
    tasks: List[GDPvalTask],
    judge: GDPvalJudge,
    checkpoint_path: str = None
) -> RegimeReport:
    """
    Classify all (model, task) pairs.
    
    - Loads checkpoint if exists (resume capability)
    - Runs 0-shot inference for each pair
    - Evaluates with judge
    - Checkpoints after each task
    """
    
    report = RegimeReport()
    
    # Load checkpoint if exists AND valid
    processed = set()
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r") as f:
                data = json.load(f)
                report = RegimeReport.from_json(data)
                processed = set((c.model_name, c.task_id) for c in report.classifications)
            print(f"Loaded checkpoint with {len(processed)} pairs processed.")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint {checkpoint_path}: {e}")

    # For each model
    for model_name in models:
        print(f"\n>>> Starting Regime Classification for {model_name}")
        
        # Setup Student Logic
        # We need to setup the LM for the student.
        try:
            if "gpt" in model_name or "claude" in model_name:
                student_lm = setup_lm(model_name)
            else:
                student_lm = setup_local_lm(model_name)
        except Exception as e:
            print(f"Failed to setup student {model_name}: {e}")
            continue
            
        # Define Zero-Shot Program
        # Because we need to invoke it per task
        with dspy.settings.context(lm=student_lm):
            student_prog = dspy.ChainOfThought(GDPvalStudentSignature)
            
            for i, task in enumerate(tasks):
                if (model_name, task.task_id) in processed:
                    continue
                
                print(f" [{i+1}/{len(tasks)}] Classifying {model_name} on {task.task_id} ({task.occupation})...")
                
                try:
                    # 1. Run Baseline (0-shot)
                    # Truncate refs if needed (heuristic 6000 chars)
                    refs_str = "\n".join(task.reference_materials)[:6000]
                    
                    pred = student_prog(
                        task_description=task.description,
                        reference_materials=refs_str
                    )
                    
                    completion = pred.answer
                    
                    # 2. Grade with Judge (Using external judge instance provided)
                    # We assume judge handles its own LM context
                    result = judge.forward(task, completion)
                    score = result.score
                    
                    # 3. Classify
                    regime, rationale = classify_regime(score)
                    
                    classification = RegimeClassification(
                        model_name=model_name,
                        task_id=task.task_id,
                        occupation=task.occupation,
                        complexity=task.complexity.value if hasattr(task.complexity, "value") else str(task.complexity),
                        baseline_score=score,
                        regime=regime.value,
                        rationale=f"{rationale} | Judge: {result.rationale[:100]}...",
                        evaluated_at=datetime.now().isoformat()
                    )
                    
                    report.classifications.append(classification)
                    
                    # Update counts
                    if regime == OptimizationRegime.REGIME_1_TOO_HARD: report.regime_1_count += 1
                    elif regime == OptimizationRegime.REGIME_2_OPTIMIZABLE: report.regime_2_count += 1
                    elif regime == OptimizationRegime.REGIME_3_ALREADY_SOLVED: report.regime_3_count += 1
                    report.total_pairs += 1
                    
                    # 4. Checkpoint
                    if checkpoint_path:
                        with open(checkpoint_path, "w") as f:
                            json.dump(report.to_json(), f, indent=2)
                            
                except Exception as e:
                    print(f"Error processing {model_name}-{task.task_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    
    return report
