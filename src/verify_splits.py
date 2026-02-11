import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tasks.loaders import LOADERS, get_loader

def get_example_id(task_name, example):
    """Extract a unique signature from an example."""
    if task_name == "code":
        # Check if it looks like MBPP or HumanEval
        # HumanEval has 'entry_point', MBPP usually just 'text'/'code'
        if hasattr(example, "entry_point"):
             return f"HE_{example.prompt[:30]}"
        else:
             return f"MBPP_{example.prompt[:30]}"
    elif task_name == "classification":
        return example.text[:50]
    elif task_name == "math":
        return example.question[:50]
    elif task_name == "qa":
        return example.question[:50]
    elif task_name == "extraction":
        return example.text[:50]
    elif task_name == "analysis":
        return example.passage[:20] + "_" + example.question[:20]
    elif task_name == "synthesis":
        # XSum has boilerplate "Media playback..." prefixes. Use hash of full content.
        return str(hash(example.document + example.summary))
    elif task_name == "agentic":
        return example.question[:50]
    elif task_name == "rag":
        return example.question[:50]
    return str(example)

def verify_task(task_name):
    print(f"--- Verifying {task_name} ---")
    try:
        loader = get_loader(task_name)
        
        # Load small samples
        train_data = loader.get_train_data(limit=50)
        eval_data = loader.get_eval_data(limit=50)
        
        print(f"  Train size: {len(train_data)}")
        print(f"  Eval size: {len(eval_data)}")
        
        train_ids = set(get_example_id(task_name, ex) for ex in train_data)
        eval_ids = set(get_example_id(task_name, ex) for ex in eval_data)
        
        intersection = train_ids.intersection(eval_ids)
        
        if intersection:
            print(f"  [FAIL] Overlap detected: {len(intersection)} items")
            print(f"  Sample overlap: {list(intersection)[:3]}")
            return False
        else:
            print(f"  [PASS] No overlap.")
            
        # Additional checks for Code
        if task_name == "code":
            if any("HE_" in x for x in train_ids):
                print("  [FAIL] HumanEval detected in Train!")
                return False
            if any("MBPP_" in x for x in eval_ids):
                print("  [FAIL] MBPP detected in Eval!")
                return False
            print("  [PASS] Code sources correct (Train=MBPP, Eval=HE).")
            
        return True
    except Exception as e:
        print(f"  [ERROR] Verification crashed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    all_passed = True
    for task in LOADERS.keys():
        if not verify_task(task):
            all_passed = False
    
    if all_passed:
        print("\n\n>>> ALL TASKS PASSED SPLIT VERIFICATION <<<")
    else:
        print("\n\n>>> FAILURES DETECTED <<<")
        sys.exit(1)

if __name__ == "__main__":
    main()
