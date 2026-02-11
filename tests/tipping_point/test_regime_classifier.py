import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.tipping_point.regime_classifier import (
    classify_regime, 
    OptimizationRegime, 
    RegimeReport, 
    run_regime_classification
)
from src.tasks.gdpval_loader import GDPvalTask, ComplexityBucket
from src.tasks.gdpval_judge import JudgeResult

class TestRegimeClassifier(unittest.TestCase):
    
    def test_classify_thresholds(self):
        """Test the basic arithmetic of classification."""
        
        # Test Regime 1 (< 0.10)
        r, _ = classify_regime(0.00)
        self.assertEqual(r, OptimizationRegime.REGIME_1_TOO_HARD)
        r, _ = classify_regime(0.09)
        self.assertEqual(r, OptimizationRegime.REGIME_1_TOO_HARD)
        
        # Test Regime 2 (0.10 <= x <= 0.70)
        r, _ = classify_regime(0.10)
        self.assertEqual(r, OptimizationRegime.REGIME_2_OPTIMIZABLE)
        r, _ = classify_regime(0.40)
        self.assertEqual(r, OptimizationRegime.REGIME_2_OPTIMIZABLE)
        r, _ = classify_regime(0.70)
        self.assertEqual(r, OptimizationRegime.REGIME_2_OPTIMIZABLE)
        
        # Test Regime 3 (> 0.70)
        r, _ = classify_regime(0.71)
        self.assertEqual(r, OptimizationRegime.REGIME_3_ALREADY_SOLVED)
        r, _ = classify_regime(1.00)
        self.assertEqual(r, OptimizationRegime.REGIME_3_ALREADY_SOLVED)

    @patch("src.tipping_point.regime_classifier.setup_local_lm")
    def test_run_classification_flow(self, mock_setup_lm):
        """Test the full loop with mocks."""
        
        # Mock LM
        mock_lm = MagicMock()
        mock_setup_lm.return_value = mock_lm
        
        # Mock Tasks
        tasks = [
            GDPvalTask(
                task_id="t1", occupation="coder", sector="tech", 
                complexity=ComplexityBucket.HIGH, description="Code something",
                reference_materials=[], expected_output_format="code", 
                gold_reference="print('hi')", estimated_hours=1, source_index=0
            )
        ]
        
        # Mock Judge
        mock_judge = MagicMock()
        mock_judge.forward.return_value = JudgeResult(
            score=0.45, rationale="Decent", task_id="t1", prediction="print('hello')",
            token_usage=100, latency_ms=500, timestamp="now"
        )
        
        # Run
        # Note: We need to mock dspy.ChainOfThought behavior because it tries to compile/call via the LM
        # But since we use dspy.settings.context(lm=mock_lm), the mock_lm should be called?
        # dspy is complex to mock fully. We will trust that if it gets to judge, logic holds.
        # However, dspy.ChainOfThought call raises error if LM doesn't respond correctly in real environment.
        # For unit test, simple mocking of dspy.ChainOfThought is safer.
        
        with patch("dspy.ChainOfThought") as mock_cot_cls:
            mock_prog = MagicMock()
            mock_cot_cls.return_value = mock_prog
            mock_prog.return_value.answer = "Mocked Prediction"
            
            report = run_regime_classification(
                models=["test_model"],
                tasks=tasks,
                judge=mock_judge,
                checkpoint_path="test_checkpoint.json"
            )
            
            # Verify results
            self.assertEqual(report.total_pairs, 1)
            self.assertEqual(report.regime_2_count, 1)
            self.assertEqual(report.classifications[0].regime, "optimizable")
            self.assertEqual(report.classifications[0].baseline_score, 0.45)
            
            # Clean up
            if os.path.exists("test_checkpoint.json"):
                os.remove("test_checkpoint.json")

if __name__ == "__main__":
    unittest.main()
