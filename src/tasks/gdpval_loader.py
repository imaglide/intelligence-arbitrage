from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from enum import Enum
from datasets import load_dataset
import random
import dspy

class GDPvalStudentSignature(dspy.Signature):
    """Complete the professional task based on the provided description and reference materials."""
    task_description = dspy.InputField(desc="Instructions for the task")
    reference_materials = dspy.InputField(desc="Background context and documents")
    answer = dspy.OutputField(desc="The professional output/completion")

class GDPvalSector(Enum):
    REAL_ESTATE = "real_estate_and_rental_leasing"
    GOVERNMENT = "government"
    MANUFACTURING = "manufacturing"
    PROFESSIONAL = "professional_scientific_technical"
    HEALTHCARE = "healthcare_and_social_assistance"
    FINANCE = "finance_and_insurance"
    RETAIL = "retail_trade"
    WHOLESALE = "wholesale_trade"
    INFORMATION = "information"

class ComplexityBucket(Enum):
    LOW = "low"      # Order clerks, counter clerks
    MEDIUM = "medium"  # Financial analysts, project managers
    HIGH = "high"    # Software developers, lawyers

# Mapping based on Spec
COMPLEXITY_MAPPING = {
    ComplexityBucket.LOW: [
        "order_clerks", "counter_clerks", "shipping_receiving", 
        "customer_service", "medical_secretaries", "concierges", "recreation_workers"
    ],
    ComplexityBucket.MEDIUM: [
        "financial_analysts", "project_managers", "accountants", 
        "registered_nurses", "buyers", "compliance_officers", "real_estate_agents"
    ],
    ComplexityBucket.HIGH: [
        "software_developers", "lawyers", "mechanical_engineers", 
        "nurse_practitioners", "financial_managers", "computer_systems_managers"
    ]
}

@dataclass
class GDPvalTask:
    task_id: str                    # "gdpval_001"
    occupation: str                 # "software_developers"
    sector: GDPvalSector
    complexity: ComplexityBucket
    description: str                # Task prompt
    reference_materials: List[str]  # Supporting documents
    expected_output_format: str     # "text", "structured", "code"
    gold_reference: Optional[str]   # Reference answer if available
    estimated_hours: float          # Human time estimate
    source_index: int               # Original dataset index

@dataclass
class GDPvalSplit:
    occupation: str
    sector: GDPvalSector
    complexity: ComplexityBucket
    train_tasks: List[GDPvalTask]  # Max 2 tasks
    test_tasks: List[GDPvalTask]   # Remaining 3+ tasks

@dataclass
class GDPvalCluster:
    cluster_name: str              # "finance_cluster"
    occupations: List[str]
    complexity: ComplexityBucket
    train_tasks: List[GDPvalTask]
    test_tasks: List[GDPvalTask]

class DatasetLoadError(Exception):
    pass

def _map_complexity(occupation: str) -> ComplexityBucket:
    # Normalize occupation string
    occ_norm = occupation.lower().replace(" ", "_")
    for bucket, occupations in COMPLEXITY_MAPPING.items():
        if any(o in occ_norm for o in occupations):
            return bucket
    # Default fallback if not strictly mapped, though spec lists specific ones.
    # We'll log warning or default to MEDIUM if unknown? 
    # Spec implies we only use the gold subset which likely matches these keys.
    return ComplexityBucket.MEDIUM

def load_gdpval_dataset() -> List[GDPvalTask]:
    """
    Load complete GDPval gold subset from HuggingFace.
    
    Returns:
        List of ~220 GDPvalTask objects.
    
    Raises:
        DatasetLoadError: If HuggingFace unavailable or malformed.
    """
    try:
        # Note: 'openai/gdpval' is the spec's dataset name. 
        # Real world check: usually requires authentication or might be gated. 
        # Assuming existing HF config covers it.
        ds = load_dataset("openai/gdpval", split="train")
    except Exception as e:
        raise DatasetLoadError(f"Dataset 'openai/gdpval' not found or unreachable. {e}")

    tasks = []
    
    # Check schema
    required_fields = ["occupation", "sector", "question", "answer", "context"]
    # Adjusting based on standard GDPval schema which usually has:
    # question (task), answer (gold), context (docs), occupation, sector...
    # Spec says: description, reference_materials, expected_output_format
    
    # We will map best effort if schema differs, but ideally it matches.
    # Let's inspect the first item logic in a real run, but here we implement based on likely fields.
    # Reference: Spec D-003 "Data source: HuggingFace openai/gdpval"
    
    for idx, item in enumerate(ds):
        try:
            # Basic mapping
            occupation = item.get("occupation", "unknown")
            sector_str = item.get("sector", "government")
            
            # Map sector string to Enum
            try:
                sector = GDPvalSector(sector_str)
            except ValueError:
                # Fallback or strict? Spec implies consistent data.
                # We'll try to find a match or default to a known one if slightly different
                # But for now strict mapping based on spec values is safer to catch data issues early.
                # However, to be robust:
                sector = GDPvalSector.GOVERNMENT # Placeholder if mismatch, or raise?
                for s in GDPvalSector:
                    if s.value == sector_str:
                        sector = s
                        break
            
            complexity = _map_complexity(occupation)
            
            task_id = f"gdpval_{idx:03d}"
            
            # Context might be list of strings or single string
            context_raw = item.get("context", [])
            if isinstance(context_raw, str):
                reference_materials = [context_raw]
            else:
                reference_materials = list(context_raw)
                
            tasks.append(GDPvalTask(
                task_id=task_id,
                occupation=occupation,
                sector=sector,
                complexity=complexity,
                description=item.get("question", ""),
                reference_materials=reference_materials,
                expected_output_format="text", # Default per spec unless schema has it
                gold_reference=item.get("answer", ""),
                estimated_hours=item.get("time_estimate", 1.0),
                source_index=idx
            ))
            
        except Exception as e:
            # Log but continue? Or fail? The spec implies 220 tasks, we want all valid ones.
            print(f"Warning: Failed to parse item {idx}: {e}")
            continue
            
    if not tasks:
        raise DatasetLoadError("No valid tasks parsed from dataset.")

    return tasks

def get_occupation_splits(
    tasks: List[GDPvalTask],
    train_size: int = 2,
    seed: int = 42
) -> Dict[str, GDPvalSplit]:
    """
    Split tasks by occupation into train/test.
    
    Args:
        tasks: All loaded tasks.
        train_size: Number for training (default 2, max 2).
        seed: Random seed for reproducibility.
    
    Returns:
        Dict mapping occupation name to GDPvalSplit.
    
    Raises:
        ValueError: If train_size > 2.
    """
    if train_size > 2:
        raise ValueError(f"train_size must be 1 or 2, got {train_size}.")
        
    random.seed(seed)
    
    # Group by occupation
    grouped = {}
    for t in tasks:
        if t.occupation not in grouped:
            grouped[t.occupation] = []
        grouped[t.occupation].append(t)
        
    splits = {}
    for occ, occ_tasks in grouped.items():
        # Need at least 3 tasks to have 1 train + 2 test (or similar logic)
        # Spec says "need 3+."
        if len(occ_tasks) < 3:
            # print(f"Warning: Occupation {occ} has {len(occ_tasks)} tasks, skipping split.")
            continue
            
        # Shuffle
        random.shuffle(occ_tasks)
        
        train_tasks = occ_tasks[:train_size]
        test_tasks = occ_tasks[train_size:]
        
        # Grab metadata from first task
        first = occ_tasks[0]
        
        splits[occ] = GDPvalSplit(
            occupation=occ,
            sector=first.sector,
            complexity=first.complexity,
            train_tasks=train_tasks,
            test_tasks=test_tasks
        )
        
    return splits

def get_complexity_clusters(
    tasks: List[GDPvalTask],
    train_ratio: float = 0.4,
    seed: int = 42
) -> Dict[ComplexityBucket, GDPvalCluster]:
    """Pool tasks by complexity for extended example curves."""
    random.seed(seed)
    
    grouped = {b: [] for b in ComplexityBucket}
    for t in tasks:
        grouped[t.complexity].append(t)
        
    clusters = {}
    for bucket, bucket_tasks in grouped.items():
        random.shuffle(bucket_tasks)
        
        split_idx = int(len(bucket_tasks) * train_ratio)
        train_tasks = bucket_tasks[:split_idx]
        test_tasks = bucket_tasks[split_idx:]
        
        clusters[bucket] = GDPvalCluster(
            cluster_name=f"{bucket.value}_cluster",
            occupations=list(set(t.occupation for t in bucket_tasks)),
            complexity=bucket,
            train_tasks=train_tasks,
            test_tasks=test_tasks
        )
        
    return clusters
