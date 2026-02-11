import dspy
from datasets import load_dataset
import random
import json

class TaskLoader:
    def __init__(self, task_name):
        self.task_name = task_name
    
    def load_data(self, split="train", limit=None):
        raise NotImplementedError
    
    def get_signature(self):
        raise NotImplementedError

    def get_metric(self):
        raise NotImplementedError

    def get_train_data(self, limit=None):
        return self.load_data(split="train", limit=limit)

    def get_eval_data(self, limit=None):
        # Default eval split logic using global registry if simple, 
        # or rely on the loader's load_data to handle the specific split name passed here.
        # We will use the TASK_SPLITS mapping to decide the split name.
        split_name = TASK_SPLITS.get(self.task_name, {}).get("eval", "test")
        return self.load_data(split=split_name, limit=limit)

TASK_SPLITS = {
    "classification": {"train": "train", "eval": "test"},
    "math":           {"train": "train", "eval": "test"},
    "qa":             {"train": "train", "eval": "validation"},
    "extraction":     {"train": "train", "eval": "test"},
    "analysis":       {"train": "train", "eval": "validation"},
    "synthesis":      {"train": "train", "eval": "test"},
    "agentic":        {"train": "train", "eval": "test"}, # Manual split handling inside loader
    "rag":            {"train": "train", "eval": "validation"},
    "code":           {"train": "train", "eval": "test"}  # MBPP vs HumanEval handling inside loader
}

# ==================================================================================
# 1. Classification (Banking77)
# ==================================================================================

class Banking77Signature(dspy.Signature):
    """Classify the user query into one of the 77 banking intent categories.
    Valid labels:
    actual_feature_request, beneficiary_not_allowed, cancel_transfer, card_about_to_expire, card_acceptance, card_arrival, card_delivery_estimate, card_linking, card_not_working, card_payment_fee_charged, card_payment_not_recognised, card_payment_wrong_exchange_rate, card_swallowed, cash_withdrawal_charge, cash_withdrawal_not_recognised, change_pin, compromised_card, contactless_not_working, country_support, declined_card_payment, declined_cash_withdrawal, declined_transfer, direct_debit_payment_not_recognised, disposable_card_limits, edit_personal_details, exchange_charge, exchange_rate, exchange_via_app, extra_charge_on_statement, failed_transfer, fiat_currency_support, get_disposable_virtual_card, get_physical_card, getting_spare_card, getting_virtual_card, lost_or_stolen_card, lost_or_stolen_phone, order_physical_card, passcode_forgotten, pending_card_payment, pending_cash_withdrawal, pending_top_up, pending_transfer, pin_blocked, receiving_money, refund_not_showing_up, request_refund, reverted_card_payment?, supported_cards_and_currencies, terminate_account, top_up_by_bank_transfer_charge, top_up_by_card_charge, top_up_by_cash_or_cheque, top_up_failed, top_up_limits, top_up_reverted, top_up_server_error, transaction_charged_twice, transfer_fee_charged, transfer_into_account, transfer_not_received_by_recipient, transfer_timing, unable_to_verify_identity, verify_my_identity, verify_source_of_funds, verify_top_up, virtual_card_not_working, visa_or_mastercard, why_verify_identity, wrong_amount_of_cash_received, wrong_exchange_rate_for_cash_withdrawal
    """
    text = dspy.InputField(desc="User query about banking")
    label = dspy.OutputField(desc="The specific intent category label from the valid list")

class ClassificationLoader(TaskLoader):
    def __init__(self):
        super().__init__("classification")
        
    def load_data(self, split="train", limit=None):
        # Load Banking77
        dataset = load_dataset("banking77", split=split)
        
        # Get label list
        labels = dataset.features["label"].names
        
        examples = []
        # Shuffle specifically for training
        if split == "train":
            dataset = dataset.shuffle(seed=42)
            
        for item in dataset:
            text = item["text"]
            label_id = item["label"]
            label_name = labels[label_id]
            
            examples.append(dspy.Example(
                text=text,
                label=label_name
            ).with_inputs("text"))
            
            if limit and len(examples) >= limit:
                break
        return examples

    def get_signature(self):
        return Banking77Signature

    def get_metric(self):
        def accuracy_metric(example, prediction, trace=None):
            # Normalize strings for comparison
            return prediction.label.strip().lower() == example.label.strip().lower()
        return accuracy_metric

# ==================================================================================
# 2. Math Reasoning (GSM8K)
# ==================================================================================

class GSM8KSignature(dspy.Signature):
    """Solve the math word problem. Give the reasoning steps and the final answer."""
    question = dspy.InputField()
    reasoning = dspy.OutputField(desc="Step-by-step calculation")
    answer = dspy.OutputField(desc="The final numerical answer")

class MathLoader(TaskLoader):
    def __init__(self):
        super().__init__("math")
        
    def load_data(self, split="train", limit=None):
        dataset = load_dataset("gsm8k", "main", split=split)
        examples = []
        
        if split == "train":
            dataset = dataset.shuffle(seed=42)
            
        for item in dataset:
            question = item["question"]
            # GSM8K answer usually contains "#### <number>"
            full_answer = item["answer"]
            answer_parts = full_answer.split("####")
            if len(answer_parts) > 1:
                final_answer = answer_parts[-1].strip()
            else:
                final_answer = full_answer.strip()
                
            examples.append(dspy.Example(
                question=question,
                answer=final_answer
            ).with_inputs("question"))
            
            if limit and len(examples) >= limit:
                break
        return examples

    def get_signature(self):
        return GSM8KSignature

    def get_metric(self):
        def gsm8k_metric(example, prediction, trace=None):
            # precise numeric matching
            try:
                # remove commas and misc chars
                pred = prediction.answer.replace(",", "").strip()
                gold = example.answer.replace(",", "").strip()
                # Basic float equality check
                return float(pred) == float(gold)
            except:
                return False
        return gsm8k_metric

# ==================================================================================
# 3. QA (HotPotQA)
# ==================================================================================

class HotPotQASignature(dspy.Signature):
    """Answer the question based on common knowledge or context."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Short answer to the question")

class QALoader(TaskLoader):
    def __init__(self):
        super().__init__("qa")
        
    def load_data(self, split="train", limit=None):
        # using distractor setting for multi-hop
        dataset = load_dataset("hotpot_qa", "distractor", split=split, trust_remote_code=True)
        examples = []
        
        if split == "train":
            dataset = dataset.shuffle(seed=42)
            
        for item in dataset:
            question = item["question"]
            answer = item["answer"]
            
            examples.append(dspy.Example(
                question=question,
                answer=answer
            ).with_inputs("question"))
            
            if limit and len(examples) >= limit:
                break
        return examples

    def get_signature(self):
        return HotPotQASignature

    def get_metric(self):
        def exact_match_metric(example, prediction, trace=None):
            return prediction.answer.strip().lower() == example.answer.strip().lower()
        return exact_match_metric

# ==================================================================================
# 4. Extraction (CoNLL-2003)
# ==================================================================================

class NERSignature(dspy.Signature):
    """Extract named entities from the text. Return entities as a JSON object with keys: PER (persons), ORG (organizations), LOC (locations), MISC (miscellaneous)."""
    text = dspy.InputField(desc="Text to extract entities from")
    entities = dspy.OutputField(desc="JSON object with keys PER, ORG, LOC, MISC, each containing a list of entity strings")

class ExtractionLoader(TaskLoader):
    def __init__(self):
        super().__init__("extraction")
        
    def load_data(self, split="train", limit=None):
        # Fallback chain for NER datasets
        dataset = None
        tag_col = "ner_tags"
        tokens_col = "tokens"
        
        candidates = [
            ("conll2003", "ner_tags"),
            ("eriktks/conll2003", "ner_tags"),
            ("wikiann", "ner_tags", "en") # wikiann requires config "en"
        ]
        
        for cand in candidates:
            try:
                name = cand[0]
                args = cand[2:]
                print(f"Trying to load NER dataset: {name}...")
                if cand[0] == "wikiann":
                    dataset = load_dataset(name, "en", split=split)
                else:
                    # Try without trust_remote_code first, then with if needed (but blocked by env)
                    try:
                        dataset = load_dataset(name, split=split, trust_remote_code=True)
                    except:
                        dataset = load_dataset(name, split=split)
                
                tag_col = cand[1]
                print(f"Successfully loaded {name}")
                break
            except Exception as e:
                print(f"Failed to load {name}: {e}")
                continue
                
        if dataset is None:
            raise RuntimeError("Could not load any NER dataset (conll2003, wikiann)")

        # tag_names might differ by dataset but for standard NER we map roughly
        # Wikiann: 0:O, 1:B-PER, 2:I-PER, 3:B-ORG, 4:I-ORG, 5:B-LOC, 6:I-LOC
        # CoNLL: 0:O, 1:B-PER, 2:I-PER, 3:B-ORG, 4:I-ORG, 5:B-LOC, 6:I-LOC, 7:B-MISC, 8:I-MISC
        # We will attempt to dynamic map or use standard mapping
        
        tag_names = dataset.features[tag_col].feature.names
        
        examples = []
        
        dataset_list = list(dataset) # iterate once
        if split == "train":
            random.Random(42).shuffle(dataset_list)
            
        count = 0
        for item in dataset_list:
            tokens = item[tokens_col]
            tags = item[tag_col]
            
            if len(tokens) < 5: continue
            
            text = " ".join(tokens)
            entities = {"PER": [], "ORG": [], "LOC": [], "MISC": []}
            current_entity = []
            current_type = None
            
            for token, tag_id in zip(tokens, tags):
                tag = tag_names[tag_id]
                # Map wikiann simplified tags if needed
                
                if tag.startswith("B-"):
                    if current_entity and current_type: entities[current_type].append(" ".join(current_entity))
                    current_type = tag[2:]
                    current_entity = [token]
                elif tag.startswith("I-") and current_type:
                    current_entity.append(token)
                else:
                    if current_entity and current_type: entities[current_type].append(" ".join(current_entity))
                    current_entity = []
                    current_type = None
            if current_entity and current_type: entities[current_type].append(" ".join(current_entity))
            
            # Clean up unknown types (wikiann doesn't have MISC usually)
            valid_keys = ["PER", "ORG", "LOC", "MISC"]
            entities = {k: v for k, v in entities.items() if k in valid_keys}

            if any(entities.values()): # Only keep positives for higher signal
                examples.append(dspy.Example(
                    text=text,
                    entities=json.dumps(entities)
                ).with_inputs("text"))
                count += 1
                
            if limit and count >= limit:
                break
        return examples

    def get_signature(self):
        return NERSignature

    def get_metric(self):
        def extraction_f1(example, prediction, trace=None):
            try:
                pred_text = prediction.entities
                if pred_text.startswith("```"):
                    pred_text = pred_text.strip("`").replace("json", "").strip()
                pred = json.loads(pred_text)
                gold = json.loads(example.entities)
                
                pred_flat = []
                gold_flat = []
                for k in ["PER", "ORG", "LOC", "MISC"]:
                    pred_flat.extend([f"{k}:{v}" for v in pred.get(k, [])])
                    gold_flat.extend([f"{k}:{v}" for v in gold.get(k, [])])
                
                pred_set, gold_set = set(pred_flat), set(gold_flat)
                if not pred_set and not gold_set: return True
                
                tp = len(pred_set & gold_set)
                fp = len(pred_set - gold_set)
                fn = len(gold_set - pred_set)
                
                f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
                return f1 >= 0.5
            except:
                return False
        return extraction_f1

# ==================================================================================
# 5. RAG (StackExchange) - Placeholder/Simplified
# ==================================================================================
# For RAG, we need a retriever. For this experiment, we might assume the context is 'retrieved' 
# or use a predefined set. Docs/PLAN.md says 'SemanticF1'.
# I'll implement a simplified "Context-based QA" loader for compatibility if strict RAG is complex locally.
# Actually, let's use a simpler RAG dataset or simulate it for now as "Given context C, answer Q".

class RAGSignature(dspy.Signature):
    """Answer the question using the provided context."""
    context = dspy.InputField(desc="Relevant information to answer the question")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="The answer, extracted from the context. Keep it concise.")

class RAGLoader(TaskLoader):
    def __init__(self):
        super().__init__("rag")
    
    def load_data(self, split="train", limit=None):
        # Using SQuAD as a proxy for RAG (Context + Question -> Answer) 
        # since actual RAG requires a retrieval index which is heavy for this step.
        # This matches the "Given context..." pattern suitable for local eval.
        dataset = load_dataset("squad", split=split)
        examples = []
        
        if split == "train":
            dataset = dataset.shuffle(seed=42)
            
        for item in dataset:
            context = item["context"]
            question = item["question"]
            answers = item["answers"]["text"]
            
            # SQuAD has multiple possible answers, take first for simplicity or handle in metric
            gold_answer = answers[0] if answers else ""
            
            examples.append(dspy.Example(
                context=context,
                question=question,
                answer=gold_answer
            ).with_inputs("context", "question"))
            
            if limit and len(examples) >= limit:
                break
        return examples

    def get_signature(self):
        return RAGSignature

    def get_metric(self):
        def semantic_match(example, prediction, trace=None):
            # Simple substring match for now, ideally use BERTScore or similar
            pred = prediction.answer.lower()
            gold = example.answer.lower()
            return gold in pred or pred in gold
        return semantic_match

# ==================================================================================
# 6. Analysis (BoolQ)
# ==================================================================================

class BoolQSignature(dspy.Signature):
    """Answer the question based on the passage with a simple Yes or No."""
    passage = dspy.InputField(desc="Context passage")
    question = dspy.InputField(desc="Question about the passage")
    answer = dspy.OutputField(desc="Yes or No")

class AnalysisLoader(TaskLoader):
    def __init__(self):
        super().__init__("analysis")
        
    def load_data(self, split="train", limit=None):
        dataset = load_dataset("boolq", split=split)
        examples = []
        
        if split == "train":
            dataset = dataset.shuffle(seed=42)
            
        for item in dataset:
            passage = item["passage"]
            question = item["question"]
            answer = "Yes" if item["answer"] else "No"
            
            examples.append(dspy.Example(
                passage=passage,
                question=question,
                answer=answer
            ).with_inputs("passage", "question"))
            
            if limit and len(examples) >= limit:
                break
        return examples

    def get_signature(self):
        return BoolQSignature

    def get_metric(self):
        def boolq_metric(example, prediction, trace=None):
            # Normalize to lower case yes/no
            pred = prediction.answer.strip().lower()
            gold = example.answer.strip().lower()
            # Handle model outputting "true"/"false" just in case
            if pred == "true": pred = "yes"
            if pred == "false": pred = "no"
            return pred == gold
        return boolq_metric

# ==================================================================================
# 7. Synthesis (XSum)
# ==================================================================================

class XSumSignature(dspy.Signature):
    """Summarize the document into a concise one-sentence summary."""
    document = dspy.InputField(desc="News article or document")
    summary = dspy.OutputField(desc="One-sentence summary")

class SynthesisLoader(TaskLoader):
    def __init__(self):
        super().__init__("synthesis")
        
    def load_data(self, split="train", limit=None):
        dataset = load_dataset("xsum", split=split)
        examples = []
        
        if split == "train":
            dataset = dataset.shuffle(seed=42)
            
        for item in dataset:
            document = item["document"]
            summary = item["summary"]
            
            examples.append(dspy.Example(
                document=document,
                summary=summary
            ).with_inputs("document"))
            
            if limit and len(examples) >= limit:
                break
        return examples

    def get_signature(self):
        return XSumSignature

    def get_metric(self):
        def synthesis_metric(example, prediction, trace=None):
            # F1 Score (Precision + Recall) to penalize copying
            pred = prediction.summary.strip().lower()
            gold = example.summary.strip().lower()
            
            if not pred: return 0.0
            
            # Simple character/word based LCS or fallback to set intersection for speed if needed
            # But let's stick to the LCS logic provided but fix the scoring
            
            m, n = len(pred), len(gold)
            if n == 0: return 0.0
            
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if pred[i - 1] == gold[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            
            lcs_length = dp[m][n]
            
            # Precision: How much of pred is relevant? (Penalizes long "copy-paste" output)
            precision = lcs_length / len(pred) if len(pred) > 0 else 0
            
            # Recall: How much of gold is captured?
            recall = lcs_length / len(gold) if len(gold) > 0 else 0
            
            if precision + recall == 0:
                return 0.0
                
            f1 = 2 * (precision * recall) / (precision + recall)
            return f1
        return synthesis_metric

# ==================================================================================
# 8. Agentic Reasoning (StrategyQA)
# ==================================================================================

class StrategyQASignature(dspy.Signature):
    """Answer the question that requires multi-step reasoning. Give the Answer as Yes or No."""
    question = dspy.InputField()
    reasoning = dspy.OutputField(desc="Step-by-step reasoning")
    answer = dspy.OutputField(desc="Yes or No")

class AgenticLoader(TaskLoader):
    def __init__(self):
        super().__init__("agentic")
        
    def load_data(self, split="train", limit=None):
        # StrategyQA doesn't have a standard HF dataset that is easy to use directly without processing sometimes
        # We'll use 'wics/strategy-qa' or similar if available, or 'metaeval/strategy-qa'
        try:
            # Always load 'train' because metaeval only has train
            dataset = load_dataset("metaeval/strategy-qa", split="train") 
        except:
             # Fallback
             try:
                 dataset = load_dataset("wics/strategy-qa", split="train")
             except:
                 print("Failed to load StrategyQA")
                 return []

        examples = []
        
        if split == "train":
            # Manual split: First 80% for train
            limit_idx = int(0.8 * len(dataset))
            dataset_list = dataset.select(range(limit_idx))
        elif split == "test" or split == "validation":
            # Manual split: Last 20% for eval
            limit_idx = int(0.8 * len(dataset))
            dataset_list = dataset.select(range(limit_idx, len(dataset)))
        else:
            dataset_list = dataset

        # Shuffle only if training
        if split == "train":
            dataset_list = dataset_list.shuffle(seed=42)
            
        for item in dataset_list:
            question = item["question"]
            # Answer is boolean in dataset
            answer = "Yes" if item["answer"] else "No"
            
            examples.append(dspy.Example(
                question=question,
                answer=answer
            ).with_inputs("question"))
            
            if limit and len(examples) >= limit:
                break
        return examples

    def get_signature(self):
        return StrategyQASignature

    def get_metric(self):
        def strategy_metric(example, prediction, trace=None):
             pred = prediction.answer.strip().lower()
             gold = example.answer.strip().lower()
             return pred == gold
        return strategy_metric

# ==================================================================================
# 9. Code Generation (HumanEval)
# ==================================================================================

class HumanEvalSignature(dspy.Signature):
    """Complete the python function based on the docstring."""
    prompt = dspy.InputField(desc="Function signature and docstring")
    completion = dspy.OutputField(desc="The implementation code")

class CodeLoader(TaskLoader):
    def __init__(self):
        super().__init__("code")
        
    def load_data(self, split="train", limit=None):
        examples = []
        
        if split == "train":
            # Use MBPP for training/optimization
            print("Loading MBPP for Code training...")
            try:
                dataset = load_dataset("mbpp", split="train", trust_remote_code=True)
            except:
                dataset = load_dataset("google-research-datasets/mbpp", split="train", trust_remote_code=True)
                
            dataset = dataset.shuffle(seed=42)
            
            for item in dataset:
                # MBPP has 'text' (prompt) and 'code' (solution)
                prompt = item["text"]
                code = item["code"]
                # We format it to match HumanEval signature approximately
                # MBPP doesn always have a function signature in 'text', it's a description.
                # But we can use it as the 'prompt'.
                
                examples.append(dspy.Example(
                    prompt=prompt,
                    completion=code
                ).with_inputs("prompt"))
                
                if limit and len(examples) >= limit:
                    break
                    
        else:
            # Use HumanEval for evaluation
            # It only has 'test' split usually
            print("Loading HumanEval for Code evaluation...")
            dataset = load_dataset("openai_humaneval", split="test", trust_remote_code=True)
            
            for item in dataset:
                prompt = item["prompt"]
                canonical_solution = item["canonical_solution"]
                test = item["test"]
                entry_point = item["entry_point"]
                
                examples.append(dspy.Example(
                    prompt=prompt,
                    canonical_solution=canonical_solution,
                    test=test,
                    entry_point=entry_point
                ).with_inputs("prompt"))
                
                if limit and len(examples) >= limit:
                    break
                    
        return examples

    def get_signature(self):
        return HumanEvalSignature

    def get_metric(self):
        def code_metric(example, prediction, trace=None):
            code = prediction.completion
            # Strip markdown formatting
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0]
            elif "```" in code:
                code = code.split("```")[1].split("```")[0]
            
            # 1. Syntax Check
            try:
                import ast
                ast.parse(code)
            except:
                return False
                
            # 2. Execution Check (Functional Correctness)
            try:
                import signal
                
                def handler(signum, frame):
                    raise TimeoutError("Execution timed out")
                
                # Set alarm for 5 seconds
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(5)
                
                try:
                    local_scope = {}
                    # example.test has assertions, example.entry_point is the function name
                    # We need to ensure we don't execute malicious code, but we assume local safe environment for now.
                    full_code = f"from typing import List, Dict, Any, Tuple, Optional\nimport math\n\n{code}\n\n{example.test}\ncheck({example.entry_point})"
                    
                    exec(full_code, {}, local_scope)
                    signal.alarm(0) # Disable alarm
                    return True
                except TimeoutError:
                    return False
                except Exception:
                    signal.alarm(0) # Disable alarm in case of other exceptions
                    return False
            except Exception:
                return False
        return code_metric

# ==================================================================================
# Registry with Extensibility
# ==================================================================================

LOADERS = {
    "classification": ClassificationLoader,
    "math": MathLoader,
    "qa": QALoader,
    "extraction": ExtractionLoader,
    "rag": RAGLoader,
    "analysis": AnalysisLoader,
    "synthesis": SynthesisLoader,
    "agentic": AgenticLoader,
    "code": CodeLoader
}

def get_loader(task_name):
    # 1. Standard Loaders
    if task_name in LOADERS:
        return LOADERS[task_name]()
    
    # 2. Custom/Generic Loaders from Registry
    try:
        from src.registry import Registry
        from src.tasks.generic_loader import GenericTaskLoader
        
        custom_tasks = Registry.load_custom_tasks()
        if task_name in custom_tasks:
            print(f"Loading custom task: {task_name}")
            return GenericTaskLoader(task_name, custom_tasks[task_name])
            
    except Exception as e:
        print(f"Warning: Failed to load custom task {task_name}: {e}")
        
    raise ValueError(f"Unknown task: {task_name}")
