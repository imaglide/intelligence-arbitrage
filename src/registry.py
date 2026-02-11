import json
import os
import sys
from typing import Dict, Any, List, Tuple

class Registry:
    @staticmethod
    def load_custom_models(path: str = "models.json") -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        Load custom model definitions from JSON.
        Returns tuple of (models_map, metadata_map).
        """
        if not os.path.exists(path):
            return {}, {}
            
        try:
            with open(path, "r") as f:
                data = json.load(f)
                
            models_map = {}
            metadata_map = {}
            
            # Expected format: {"model_tag": { "ollama_tag": "...", "context_length": ... }}
            # Or simplified: {"model_tag": "ollama_tag"}
            
            for key, value in data.items():
                if isinstance(value, str):
                    models_map[key] = value
                elif isinstance(value, dict):
                    # Complex definition
                    ollama_tag = value.get("ollama_tag", key) # Default to key if missing
                    models_map[key] = ollama_tag
                    metadata_map[key] = value
                    
            print(f"Loaded {len(models_map)} custom models from {path}")
            return models_map, metadata_map
            
        except Exception as e:
            print(f"Error loading custom models from {path}: {e}")
            return {}, {}

    @staticmethod
    def load_custom_tasks(path: str = "tasks.json") -> Dict[str, Any]:
        """
        Load custom task definitions.
        Format:
        {
            "task_name": {
                "file_path": "data/my_task.csv",
                "format": "csv",
                "input_columns": ["question", "context"],
                "target_column": "answer",
                "metric": "exact_match" # or "llm_judge"
            }
        }
        """
        if not os.path.exists(path):
            return {}
            
        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            print(f"Loaded {len(data)} custom tasks from {path}")
            return data
            
        except Exception as e:
            print(f"Error loading custom tasks from {path}: {e}")
            return {}
