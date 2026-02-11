# Technical Specification: ExperimentDSPy

## 1. Project Objective
The primary goal of this project is to experimentally evaluate the "Intelligence Arbitrage" hypothesis: **Can smaller, locally running models (via Ollama) match or exceed the performance of frontier API models (GPT-4o) when optimized using DSPy?**

## 2. Architecture & Stack
*   **Orchestration:** `DSPy` (Declarative Self-improving Language Programs).
*   **Local Inference:** `Ollama` (running Llama 3.2, Qwen 2.5, Mistral).
*   **API Inference:** `OpenAI` (GPT-4o, GPT-4o-mini), `Anthropic` (Claude).
*   **Data Handling:** `HuggingFace Datasets`, `Pandas`.

## 3. Supported Models
| Type | Models |
| :--- | :--- |
| **Local (Student)** | `llama3.2`, `qwen2.5:7b`, `mistral:latest` |
| **API (Teacher)** | `gpt-4o` |
| **API (Baseline)** | `gpt-4o`, `gpt-4o-mini` |

## 4. Task Registry
The system evaluates models across 9 cognitive domains.

| Task Domain | Dataset | Metric | Split Strategy |
| :--- | :--- | :--- | :--- |
| **Classification** | `Banking77` | Accuracy | Train / Test |
| **Math** | `GSM8K` | Exact Match | Train / Test |
| **QA** | `HotPotQA` | Exact Match | Train / Validation |
| **Extraction** | `CoNLL-2003` | F1 Score | Train / Test |
| **Analysis** | `BoolQ` | Accuracy | Train / Validation |
| **Synthesis** | `XSum` | F1 Score (w/ copy penalty) | Train / Test |
| **Agentic** | `StrategyQA` | Accuracy | Train / Test |
| **RAG** | Custom | Faithfulness | Train / Validation |
| **Code** | `MBPP` (Train) -> `HumanEval` (Eval) | Pass@1 | **Cross-Dataset** |
| **Professional** | **`openai/gdpval`** | **GPT-4o Judge** | **Train / Test** |

## 5. GDPVal Implementation Details
*   **Dataset Source:** `openai/gdpval` (HuggingFace)
    *   *Note: Using the "Gold Open-Sourced Set" (approx. 220 tasks).*
*   **Grading Strategy:** **GPT-4o as LLM-Judge**
    *   **Rationale:** Aligns with the "Teacher" architecture. Provides interpretable reasoning traces via DSPy. Avoids reliance on black-box external grading APIs.
    *   **Implementation:** A DSPy module `GDPValJudge` that takes `(task, reference, prediction)` and outputs a `score` and `rationale`.

## 6. Optimization Loop (`src/optimize.py`)
1.  **Teacher Bootstrapping:** `gpt-4o` generates examples.
2.  **MIPROv2:** Optimizes instructions/demonstrations.
3.  **Variance Reduction:** 3 seeds (10, 20, 30) -> Best of 3 on Dev Set.

## 7. Configuration
*   **Config File:** `src/config.py`
*   **Environment:** `.env` for API Keys.
*   **Caching:** `DSP_CACHEBOOL` to control persistence.
