import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

    # Ollama Settings
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Cache Settings
    # Default to True if not set, but allow env var to override (e.g. "False", "0", "false")
    DSP_CACHEBOOL = os.getenv("DSP_CACHEBOOL", "True").lower() in ("true", "1", "t", "yes")
    
    # Model Names (Ollama Tags)
    OLLAMA_MODELS = {
        "llama3.2": "llama3.2",
        "qwen2.5": "qwen2.5:7b",
        "qwen2.5_7b": "qwen2.5:7b",
        "mistral": "mistral:latest",
        # Reasoning models for tipping point experiment
        "qwen3:4b": "qwen3:4b",
        "deepseek-r1:7b": "deepseek-r1:7b"
    }
    
    # Detailed Model Configs (New in Tipping Point Spec)
    MODEL_METADATA = {
        "qwen3:4b": {
            "name": "qwen3:4b",
            "context_length": 32768,
            "memory_gb": 2.5,
            "supports_thinking": True,
        },
        "deepseek-r1:7b": {
            "name": "deepseek-r1:7b",
            "context_length": 65536,
            "memory_gb": 4.5,
            "supports_thinking": True,
        },
    }

    # Load Custom Models
    try:
        from src.registry import Registry
        custom_models, custom_meta = Registry.load_custom_models()
        OLLAMA_MODELS.update(custom_models)
        MODEL_METADATA.update(custom_meta)
    except ImportError:
        pass # Registry might not exist during early bootstrap
    except Exception as e:
        print(f"Config Warning: {e}")

    TIPPING_POINT_CONFIG = {
        "models": ["qwen3:4b", "deepseek-r1:7b"],
        "example_counts_per_occupation": [0, 1, 2],
        "example_counts_pooled": [0, 1, 2, 3, 5, 8],
        "regime_thresholds": {
            "too_hard": 0.10,
            "already_solved": 0.70,
        },
        "mipro_config": {
            "num_candidates": 3,
            "seeds": [10, 20, 30],
            "teacher_model": "gpt-4o",
        },
    }

    # Pricing per 1M tokens (Input, Output)
    MODEL_PRICING = {
        # New Frontier Baselines
        "gpt-5.2": (10.0, 30.0),
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60)
    }

    # Task Settings
    TASKS = [
        "classification",   # Banking77
        "math",             # GSM8K
        "qa",               # HotPotQA
        "extraction",       # CoNLL-2003
        "analysis",         # BoolQ
        "synthesis",        # XSum
        "agentic",          # StrategyQA
        "rag",              # RAG (QA w/ Context)
        "code"              # HumanEval/MBPP
    ]
    
    # Paths
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

    @classmethod
    def validate(cls):
        """Ensure critical configuration is present."""
        if not cls.OPENAI_API_KEY:
            print("WARNING: OPENAI_API_KEY not found in environment. Baseline comparisons will fail.")
        if not cls.ANTHROPIC_API_KEY:
            print("WARNING: ANTHROPIC_API_KEY not found in environment. Baseline comparisons will fail.")
