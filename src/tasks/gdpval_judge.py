import dspy
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict
import os
import sys

# Add project root to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import Config
from src.api_eval import setup_lm
from src.tasks.gdpval_loader import GDPvalTask

@dataclass
class JudgeResult:
    score: float              # 0.0 to 1.0
    rationale: str            # Explanation
    task_id: str
    prediction: str
    token_usage: int
    latency_ms: int
    timestamp: str            # ISO format

class GDPvalJudgeSignature(dspy.Signature):
    """
    Evaluate a professional task completion based on the following rubric:
    
    SCORING RUBRIC (0.0 to 1.0):
    - 0.0-0.2 (Fail): Completely wrong, off-topic, or harmful
    - 0.2-0.4 (Poor): Major factual errors or gaps
    - 0.4-0.6 (Fair): Significant quality issues
    - 0.6-0.8 (Good): Minor issues, room for improvement
    - 0.8-1.0 (Excellent): Professional-quality, meets all requirements

    EVALUATION CRITERIA:
    1. Accuracy: Facts, calculations correct?
    2. Completeness: All parts addressed?
    3. Format: Matches expected output?
    4. Professionalism: Workplace acceptable?
    5. Reasoning: Logic sound?
    """
    
    task_description = dspy.InputField(desc="The professional task instructions")
    reference_materials = dspy.InputField(desc="Supporting documents or context")
    expected_format = dspy.InputField(desc="Format requirements")
    gold_reference = dspy.InputField(desc="The expert/gold answer")
    prediction = dspy.InputField(desc="The model's completion to evaluate")
    
    rationale = dspy.OutputField(desc="Detailed explanation of the score based on criteria")
    score = dspy.OutputField(desc="Final score as a float between 0.0 and 1.0")

class GDPvalJudge(dspy.Module):
    def __init__(self, model: str = "gpt-4o", max_retries: int = 3):
        super().__init__()
        self.model_name = model
        self.max_retries = max_retries
        
        # Initialize the Judge LM (Teacher/Grader)
        # We assume setup_lm handles API keys and configuration
        try:
            self.lm = setup_lm(model)
        except Exception as e:
            print(f"Error initializing Judge LM ({model}): {e}")
            self.lm = None
            
        self.prog = dspy.Retry(dspy.ChainOfThought(GDPvalJudgeSignature))
        
        # Metrics tracking
        self.total_cost = 0.0
        self.total_tokens_in = 0
        self.total_tokens_out = 0

    def forward(self, task: GDPvalTask, prediction: str) -> JudgeResult:
        if not self.lm:
            raise RuntimeError("Judge LM not initialized. Check API keys.")

        start_time = datetime.now()
        
        # Handle empty prediction
        if not prediction or not prediction.strip():
            return JudgeResult(
                score=0.0,
                rationale="Empty prediction provided.",
                task_id=task.task_id,
                prediction="",
                token_usage=0,
                latency_ms=0,
                timestamp=datetime.now().isoformat()
            )

        # Truncate reference materials if too long to fit in context?
        # GPT-4o has 128k context, so likely fine, but good to join safely.
        # We limit to reasonable length if needed, but for now just join.
        context_str = "\n\n".join(task.reference_materials[:5]) # Limit to 5 docs just in case
        
        # Execute with the judge LM context
        try:
            with dspy.settings.context(lm=self.lm):
                pred = self.prog(
                    task_description=task.description,
                    reference_materials=context_str,
                    expected_format=task.expected_output_format,
                    gold_reference=task.gold_reference or "N/A",
                    prediction=prediction
                )
        except Exception as e:
             # Retry logic is handled by dspy.Retry, but if it ultimately fails:
             print(f"Judge failed for task {task.task_id}: {e}")
             return JudgeResult(
                score=0.0,
                rationale=f"Judgement failed: {str(e)}",
                task_id=task.task_id,
                prediction=prediction,
                token_usage=0,
                latency_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                timestamp=datetime.now().isoformat()
            )

        # Parse score safely
        raw_score = pred.score
        try:
            # Handle cases where model outputs "0.8 (Good)" or similar
            if isinstance(raw_score, str):
                 # Extract first float found
                import re
                match = re.search(r"(\d+(\.\d+)?)", raw_score)
                if match:
                    score = float(match.group(1))
                else:
                    score = 0.0
            else:
                score = float(raw_score)
                
            # Clamp
            score = max(0.0, min(1.0, score))
            
        except (ValueError, TypeError):
            score = 0.0
            print(f"Warning: Could not parse score '{raw_score}' for task {task.task_id}")

        end_time = datetime.now()
        latency = int((end_time - start_time).total_seconds() * 1000)
        
        # Track usage
        tokens_in, tokens_out = 0, 0
        if hasattr(self.lm, "history") and self.lm.history:
            last = self.lm.history[-1]
            usage = last.get("usage") or last.get("response", {}).get("usage")
            if usage:
                tokens_in = usage.get("prompt_tokens", 0)
                tokens_out = usage.get("completion_tokens", 0)
                
        self._update_cost(tokens_in, tokens_out)
        
        return JudgeResult(
            score=score,
            rationale=pred.rationale,
            task_id=task.task_id,
            prediction=prediction,
            token_usage=tokens_in + tokens_out,
            latency_ms=latency,
            timestamp=end_time.isoformat()
        )

    def _update_cost(self, input_tokens: int, output_tokens: int):
        self.total_tokens_in += input_tokens
        self.total_tokens_out += output_tokens
        
        # Use config pricing
        price_in, price_out = Config.MODEL_PRICING.get(self.model_name, (2.50, 10.00))
        cost = (input_tokens / 1_000_000 * price_in) + (output_tokens / 1_000_000 * price_out)
        self.total_cost += cost
    
    def get_cost_estimate(self) -> float:
        return self.total_cost
    
    def reset_metrics(self) -> None:
        self.total_cost = 0.0
        self.total_tokens_in = 0
        self.total_tokens_out = 0
