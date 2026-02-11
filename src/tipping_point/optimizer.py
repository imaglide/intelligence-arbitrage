import dspy
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any, Callable, Type
import os
import sys
import json
import time
from datetime import datetime

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import Config
from src.local_eval import setup_local_lm
from src.api_eval import setup_lm
from src.tasks.gdpval_loader import GDPvalTask, GDPvalStudentSignature
from src.tasks.gdpval_judge import GDPvalJudge

@dataclass
class OptimizationConfig:
    model_name: str
    task_ids: List[str] # Or simply identifiers
    num_examples: int
    num_candidates: int = 3
    seeds: List[int] = field(default_factory=lambda: [10, 20, 30])
    teacher_model: str = "gpt-4o"
    # New: allow passing task source metadata
    task_source: str = "gdpval"

@dataclass
class OptimizationResult:
    config: OptimizationConfig
    baseline_scores: Dict[str, float]
    optimized_scores: Dict[str, float]
    mean_baseline: float
    mean_optimized: float
    delta: float
    best_seed: int
    program_path: str
    optimization_time_seconds: float
    judge_cost_usd: float
    
    def to_json(self):
        return asdict(self)

@dataclass
class TippingPointCurve:
    model_name: str
    task_group: str
    complexity: str
    example_counts: List[int]
    scores: List[float]
    deltas: List[float]
    tipping_point: Optional[int]
    tipping_point_method: str
    
    def to_json(self):
        return asdict(self)

class TippingPointOptimizer:
    def __init__(self, judge: Optional[Any], checkpoint_dir: str):
        # Judge is optional now (only needed for GDPval)
        self.judge = judge
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def _create_gdpval_metric(self) -> Callable:
        """Create metric wrapper for GDPval logic."""
        def gdpval_metric(example, prediction, trace=None):
            task_data = getattr(example, "task_data", None)
            if not task_data: return 0.0
            temp_task = GDPvalTask(**task_data)
            try:
                pred_str = getattr(prediction, "answer", "")
                res = self.judge.forward(temp_task, pred_str)
                return res.score
            except Exception as e:
                print(f"Metric Error: {e}")
                return 0.0
        return gdpval_metric

    def prepare_data(self, data_points: List[Any], task_source: str = "gdpval"):
        """Convert raw items to dspy.Examples based on source."""
        examples = []
        
        if task_source == "gdpval":
            from src.tasks.gdpval_loader import GDPvalTask
            for t in data_points:
                if not isinstance(t, GDPvalTask): continue
                # GDPval Logic
                refs_str = "\n".join(t.reference_materials)[:6000]
                ex = dspy.Example(
                    task_description=t.description,
                    reference_materials=refs_str,
                    task_data=t.__dict__,
                    gold_answer=t.gold_reference 
                ).with_inputs("task_description", "reference_materials")
                examples.append(ex)
        
        elif task_source == "core":
            # For core tasks, we assume data_points are already dspy.Examples from loader.get_train_data()
            # We just pass them through or ensure inputs are set.
            # Assuming they come from loader.get_train_data() they are ready.
            for ex in data_points:
                if isinstance(ex, dspy.Example):
                    examples.append(ex)
                    
        return examples

    def run_optimization(
        self,
        config: OptimizationConfig,
        train_data: List[Any],
        test_data: List[Any],
        signature_class: Type[dspy.Signature],
        metric_fn: Optional[Callable] = None
    ) -> OptimizationResult:
        
        start_time = time.time()
        print(f"\n>>> Running Optimization: {config.model_name} | n={config.num_examples} | Source={config.task_source}")
        
        # 1. Setup Student
        try:
            if "gpt" in config.model_name:
                student = setup_lm(config.model_name)
            else:
                student = setup_local_lm(config.model_name)
        except Exception as e:
            print(f"Failed to setup student: {e}")
            return None

        # 2. Setup Teacher
        try:
            teacher = setup_lm(config.teacher_model)
        except Exception as e:
            print(f"Failed to setup teacher: {e}")
            teacher = student

        dspy.configure(lm=student)
        
        # 3. Prepare Data
        # If core, we assume incoming data is already format we want or dspy Examples
        train_examples = self.prepare_data(train_data, config.task_source)
        test_examples = self.prepare_data(test_data, config.task_source)
        
        # Metric Logic — wrap to handle None predictions from reasoning models
        raw_metric = self._create_gdpval_metric() if config.task_source == "gdpval" else metric_fn

        def safe_metric(example, prediction, trace=None):
            try:
                return raw_metric(example, prediction, trace)
            except (AttributeError, TypeError):
                return 0.0

        optimizer_metric = safe_metric
        eval_metric = safe_metric

        # 4. Baseline Evaluation
        print(" Evaluating Baseline...")
        baseline_prog = dspy.ChainOfThought(signature_class)
        
        baseline_scores = {}
        
        # Evaluate Test Set
        for i, ex in enumerate(test_examples):
            key = ex.task_data.get('task_id', f"item_{i}") if config.task_source == "gdpval" else f"item_{i}"
            try:
                kwargs = ex.inputs().toDict()
                pred = baseline_prog(**kwargs)
                score = eval_metric(ex, pred)
                baseline_scores[key] = float(score) if isinstance(score, (int, float, bool)) else 0.0
            except Exception as e:
                baseline_scores[key] = 0.0
                
        mean_baseline = sum(baseline_scores.values()) / len(baseline_scores) if baseline_scores else 0.0
        print(f" Baseline Mean Score: {mean_baseline:.4f}")

        # 5. Optimization Loop
        best_score = -1
        best_program = None
        best_seed = -1
        
        for seed in config.seeds:
            print(f" Optimizing Seed {seed}...")
            dspy.settings.configure(random_seed=seed)
            
            program = dspy.ChainOfThought(signature_class)
            n_shots = config.num_examples
            
            teleprompter = BootstrapFewShotWithRandomSearch(
                metric=optimizer_metric,
                teacher_settings={"lm": teacher},
                max_bootstrapped_demos=n_shots,
                max_labeled_demos=n_shots, 
                num_candidate_programs=config.num_candidates,
                num_threads=1 
            )
            
            try:
                optimized_prog = teleprompter.compile(
                    program,
                    trainset=train_examples,
                )
                
                # Verify on Test
                current_scores = []
                for ex in test_examples:
                    try:
                        kwargs = ex.inputs().toDict()
                        pred = optimized_prog(**kwargs)
                        s = eval_metric(ex, pred)
                        current_scores.append(float(s))
                    except Exception:
                        current_scores.append(0.0)
                
                avg_test = sum(current_scores) / len(current_scores) if current_scores else 0.0
                
                if avg_test > best_score:
                    best_score = avg_test
                    best_program = optimized_prog
                    best_seed = seed
                    
            except Exception as e:
                print(f"Optimization failed for seed {seed}: {e}")
                # traceback.print_exc()

        # 6. Final Evaluation & Save
        optimized_scores = {}
        if best_program:
            for i, ex in enumerate(test_examples):
                key = ex.task_data.get('task_id', f"item_{i}") if config.task_source == "gdpval" else f"item_{i}"
                try:
                    kwargs = ex.inputs().toDict()
                    pred = best_program(**kwargs)
                    s = eval_metric(ex, pred)
                    optimized_scores[key] = float(s)
                except Exception:
                    optimized_scores[key] = 0.0
                
            mean_optimized = sum(optimized_scores.values()) / len(optimized_scores)
            
            # Save Program
            safe_name = config.model_name.replace(":", "_")
            # Include task in filename if possible
            # But OptimizationConfig usually focuses on one group.
            prog_filename = f"{config.task_source}_{safe_name}_n{config.num_examples}_seed{best_seed}.json"
            prog_path = os.path.join(self.checkpoint_dir, "../optimized_programs", prog_filename)
            os.makedirs(os.path.dirname(prog_path), exist_ok=True)
            best_program.save(prog_path)
        else:
            mean_optimized = mean_baseline
            optimized_scores = baseline_scores.copy()
            prog_path = "failed"
            
        judge_cost = 0.0
        if self.judge and config.task_source == "gdpval":
             judge_cost = self.judge.get_cost_estimate()
             
        result = OptimizationResult(
            config=config,
            baseline_scores=baseline_scores,
            optimized_scores=optimized_scores,
            mean_baseline=mean_baseline,
            mean_optimized=mean_optimized,
            delta=mean_optimized - mean_baseline,
            best_seed=best_seed,
            program_path=prog_path,
            optimization_time_seconds=time.time() - start_time,
            judge_cost_usd=judge_cost
        )
        
        return result

    def detect_tipping_point(
        self,
        example_counts: List[int],
        deltas: List[float],
        threshold: float = 0.02
    ) -> Optional[int]:
        if len(deltas) < 2: return None
        for i in range(1, len(deltas)):
            improvement = deltas[i] - deltas[i-1]
            if improvement < threshold:
                return example_counts[i-1]
        return None

    def run_tipping_point_experiment(
        self,
        model_name: str,
        # Generic arguments now
        train_data: List[Any],
        test_data: List[Any],
        signature_class: Type[dspy.Signature],
        metric_fn: Optional[Callable],
        example_counts: List[int],
        group_id: str, # e.g. "software_dev" or "math"
        complexity_label: str = "standard",
        task_source: str = "gdpval"
    ) -> TippingPointCurve:
        
        scores = []
        deltas = []
        
        for n in example_counts:
            # Config
            cfg = OptimizationConfig(
                model_name=model_name,
                task_ids=[f"{group_id}_{i}" for i in range(len(train_data))],
                num_examples=n,
                num_candidates=3, 
                seeds=[10],
                task_source=task_source
            )
            
            res = self.run_optimization(cfg, train_data, test_data, signature_class, metric_fn)
            
            if res:
                scores.append(res.mean_optimized)
                deltas.append(res.delta)
                
                # Checkpoint result
                cp_file = f"result_{model_name.replace(':','_')}_{group_id}_{n}.json"
                with open(os.path.join(self.checkpoint_dir, cp_file), "w") as f:
                    json.dump(res.to_json(), f, indent=2)
            else:
                scores.append(0.0)
                deltas.append(0.0)
                
        tp = self.detect_tipping_point(example_counts, deltas)
        
        curve = TippingPointCurve(
            model_name=model_name,
            task_group=group_id,
            complexity=complexity_label,
            example_counts=example_counts,
            scores=scores,
            deltas=deltas,
            tipping_point=tp,
            tipping_point_method="delta_threshold_0.02"
        )
        return curve
