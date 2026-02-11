import pandas as pd
import dspy
import os
import json
from typing import List, Dict, Any, Optional

class GenericSignature:
    """Dynamically created signature base."""
    pass

class GenericTaskLoader:
    def __init__(self, task_name: str, config: Dict[str, Any]):
        self.task_name = task_name
        self.config = config
        self.file_path = config.get("file_path")
        self.format = config.get("format", "csv")
        self.input_cols = config.get("input_columns", ["question"])
        self.target_col = config.get("target_column", "answer")
        self.metric_type = config.get("metric", "exact_match")
        
        # Load Data
        self.df = self._load_data()
        
        # Create Signature Class dynamically
        self.Signature = self._create_signature()
        
    def _load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
            
        if self.format == "csv":
            return pd.read_csv(self.file_path)
        elif self.format == "json":
            return pd.read_json(self.file_path)
        else:
            raise ValueError(f"Unsupported format: {self.format}")

    def _create_signature(self):
        # Dynamically create fields
        fields = {}
        for col in self.input_cols:
            fields[col] = dspy.InputField()
        fields[self.target_col] = dspy.OutputField()
        
        # Create Type
        # Note: dspy.Signature meta programming requires creating class
        return type(f"{self.task_name}Signature", (dspy.Signature,), fields)

    def get_signature(self):
        return self.Signature

    def get_metric(self):
        # Return generic metric based on config
        if self.metric_type == "exact_match":
            return dspy.evaluate.metrics.answer_exact_match
        elif self.metric_type == "passage_match":
             return dspy.evaluate.metrics.answer_passage_match
        else:
            # Default exact
            return dspy.evaluate.metrics.answer_exact_match

    def _df_to_examples(self, df: pd.DataFrame) -> List[dspy.Example]:
        examples = []
        for _, row in df.iterrows():
            inputs = {col: str(row[col]) for col in self.input_cols if col in row}
            target = {self.target_col: str(row[self.target_col])} if self.target_col in row else {}
            
            # Combine
            ex = dspy.Example(**inputs, **target).with_inputs(*self.input_cols)
            examples.append(ex)
        return examples

    def get_train_data(self, limit: Optional[int] = None) -> List[dspy.Example]:
        # Simple split: 80% train
        split_idx = int(len(self.df) * 0.8)
        train_df = self.df.iloc[:split_idx]
        if limit:
            train_df = train_df.iloc[:limit]
        return self._df_to_examples(train_df)

    def get_eval_data(self, limit: Optional[int] = None) -> List[dspy.Example]:
        # Simple split: 20% eval
        split_idx = int(len(self.df) * 0.8)
        eval_df = self.df.iloc[split_idx:]
        if limit:
            eval_df = eval_df.iloc[:limit]
        return self._df_to_examples(eval_df)
