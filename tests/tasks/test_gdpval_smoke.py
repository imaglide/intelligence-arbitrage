import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.tasks.gdpval_loader import load_gdpval_dataset, GDPvalTask, GDPvalSector, ComplexityBucket, get_occupation_splits
from src.tasks.gdpval_judge import GDPvalJudge, JudgeResult
from src.config import Config

class TestGDPvalSmoke(unittest.TestCase):
    def test_01_loader_imports(self):
        """Verify we can import and seeing basic classes."""
        self.assertTrue(issubclass(GDPvalSector, object))
        self.assertTrue(issubclass(ComplexityBucket, object))
        print("\n[Pass] Loader imports successful.")

    @patch("src.tasks.gdpval_loader.load_dataset")
    def test_02_loader_structure(self, mock_load_dataset):
        """Verify parsing logic with mocked HF dataset (avoiding network dep for smoke test)."""
        # Mock HF dataset iterator
        mock_data = [
            {
                "occupation": "software_developers",
                "sector": "information",
                "question": "Write a Python script.",
                "context": ["Ref 1", "Ref 2"],
                "answer": "print('hello')",
                "time_estimate": 0.5
            },
            {
                "occupation": "accountants",
                "sector": "finance_and_insurance",
                "question": "Balance this.",
                "context": "Spreadsheet", # String, not list
                "answer": "Done",
                "time_estimate": 1.0
            }
        ]
        mock_load_dataset.return_value = mock_data
        
        tasks = load_gdpval_dataset()
        self.assertEqual(len(tasks), 2)
        
        t1 = tasks[0]
        self.assertEqual(t1.occupation, "software_developers")
        self.assertEqual(t1.complexity, ComplexityBucket.HIGH) # software dev = high
        self.assertEqual(len(t1.reference_materials), 2)
        
        t2 = tasks[1]
        self.assertEqual(t2.occupation, "accountants")
        self.assertEqual(t2.complexity, ComplexityBucket.MEDIUM) # accountant = medium
        self.assertIsInstance(t2.reference_materials, list)
        self.assertEqual(t2.reference_materials[0], "Spreadsheet")
        
        print("\n[Pass] Loader structure parsing successful.")

    def test_03_judge_instantiation(self):
        """Verify Judge can be instantiated."""
        try:
            # We don't want to actually connect to OpenAI if no key, but setup_lm might complain.
            # We will rely on checking if logic loads.
            if Config.OPENAI_API_KEY:
                judge = GDPvalJudge(model="gpt-4o")
                self.assertIsNotNone(judge.lm)
                print("\n[Pass] Judge instantiation successful (w/ API Key).")
            else:
                print("\n[Skip] No OPENAI_API_KEY, skipping Judge instantiation test.")
        except Exception as e:
            self.fail(f"Judge instantiation failed: {e}")

if __name__ == "__main__":
    unittest.main()
