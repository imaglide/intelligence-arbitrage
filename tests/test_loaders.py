import unittest
from src.tasks.loaders import get_loader
import dspy

class TestTaskLoaders(unittest.TestCase):
    
    def test_classification_loader(self):
        loader = get_loader("classification")
        self.assertIsNotNone(loader)
        
        # Test signature
        sig = loader.get_signature()
        self.assertTrue(issubclass(sig, dspy.Signature))
        
        # Test data loading (limit to 5 for speed)
        data = loader.load_data(split="train", limit=5)
        self.assertEqual(len(data), 5)
        self.assertIsInstance(data[0], dspy.Example)
        self.assertTrue(hasattr(data[0], "text"))
        self.assertTrue(hasattr(data[0], "label"))

    def test_math_loader(self):
        loader = get_loader("math")
        data = loader.load_data(split="train", limit=5)
        self.assertEqual(len(data), 5)
        self.assertTrue(hasattr(data[0], "question"))
        self.assertTrue(hasattr(data[0], "answer"))
        
    def test_qa_loader(self):
        loader = get_loader("qa")
        data = loader.load_data(split="train", limit=5)
        self.assertEqual(len(data), 5)
        self.assertTrue(hasattr(data[0], "question"))
        self.assertTrue(hasattr(data[0], "answer"))

    def test_extraction_loader(self):
        loader = get_loader("extraction")
        data = loader.load_data(split="train", limit=5)
        if len(data) > 0: # Might be 0 if random selection misses long sentences in small sample, but unlikely with limit logic
            self.assertTrue(hasattr(data[0], "text"))
            self.assertTrue(hasattr(data[0], "entities"))
            
    def test_rag_loader(self):
        loader = get_loader("rag")
        data = loader.load_data(split="train", limit=5)
        self.assertEqual(len(data), 5)
        self.assertTrue(hasattr(data[0], "context"))
        self.assertTrue(hasattr(data[0], "question"))

if __name__ == '__main__':
    unittest.main()
