# Intelligence Arbitrage

Can smaller, locally-running language models match or exceed frontier API models when optimized with DSPy? This project systematically tests the "intelligence arbitrage" hypothesis across 9 cognitive task domains.

## Hypothesis

DSPy's optimization pipeline (teacher bootstrapping + MIPROv2) can close the performance gap between local open-weight models (3B-7B parameters) and frontier API models (GPT-4o, GPT-5.2) on structured NLP tasks.

## Results

Benchmark results across 9 models and 9 task domains (50 samples each):

| Model | Agentic | Analysis | Classification | Code | Extraction | Math | QA | RAG | Synthesis |
|:------|:--------|:---------|:---------------|:-----|:-----------|:-----|:---|:----|:----------|
| gpt-5.2 | 86.0% | 88.0% | 64.0% | 90.0% | 44.0% | 90.0% | 22.0% | 88.0% | 39.9% |
| gpt-4o | 82.0% | 78.0% | 56.0% | 76.0% | 46.0% | 68.0% | 32.0% | 88.0% | 43.3% |
| gpt-4o-mini | 86.0% | 84.0% | 30.0% | 92.0% | 48.0% | 52.0% | 14.0% | 88.0% | 42.8% |
| phi4 | 74.0% | 80.0% | 26.0% | 90.0% | 52.0% | 48.0% | 8.0% | 86.0% | 41.2% |
| qwen2.5 (7B) | 74.0% | 78.0% | 18.0% | 84.0% | 40.0% | 36.0% | 10.0% | 82.0% | 43.1% |
| llama3.2 | 66.0% | 68.0% | 28.0% | 68.0% | 50.0% | 40.0% | 0.0% | 82.0% | 44.9% |
| mistral | 56.0% | 76.0% | 10.0% | 56.0% | 50.0% | 2.0% | 0.0% | 84.0% | 44.6% |

Full per-model CSV results are in `results/`.

## Architecture

```
┌─────────────────────────────────────────────┐
│              Task Registry (9 domains)       │
│  classification · math · qa · extraction     │
│  analysis · synthesis · agentic · rag · code │
└──────────────────┬──────────────────────────┘
                   │
    ┌──────────────┼──────────────┐
    ▼              ▼              ▼
┌────────┐   ┌─────────┐   ┌──────────┐
│  API   │   │  Local   │   │ Optimized│
│Baseline│   │ Baseline │   │  (DSPy)  │
│ GPT-4o │   │ Ollama   │   │ MIPROv2  │
│ GPT-5.2│   │ models   │   │ Teacher  │
└────────┘   └─────────┘   └──────────┘
    │              │              │
    └──────────────┼──────────────┘
                   ▼
           results/*.csv
```

### Optimization Loop

1. **Teacher Bootstrapping**: GPT-4o generates high-quality examples
2. **MIPROv2**: Optimizes instructions and demonstrations for local models
3. **Variance Reduction**: 3 seeds (10, 20, 30), best-of-3 on dev set

## Task Domains

| Domain | Dataset | Metric |
|:-------|:--------|:-------|
| Classification | Banking77 | Accuracy |
| Math | GSM8K | Exact Match |
| QA | HotPotQA | Exact Match |
| Extraction | CoNLL-2003 | F1 Score |
| Analysis | BoolQ | Accuracy |
| Synthesis | XSum | F1 Score (w/ copy penalty) |
| Agentic | StrategyQA | Accuracy |
| RAG | Custom | Faithfulness |
| Code | MBPP → HumanEval | Pass@1 (cross-dataset) |

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) (for local model inference)
- OpenAI API key (for API baselines and teacher bootstrapping)
- Anthropic API key (optional, for Claude baselines)

## Setup

```bash
git clone https://github.com/imaglide/intelligence-arbitrage.git
cd intelligence-arbitrage

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API keys
```

### Pull Local Models

```bash
# Install required Ollama models
./setup_models.sh
```

## Running Experiments

```bash
# Run the full experiment suite
./run_experiment.sh

# Or run individual components:
source venv/bin/activate

# API baselines (GPT-4o, GPT-4o-mini)
python -m src.run_baseline_v2 --mode api

# Local baselines (Ollama models)
python -m src.run_baseline_v2 --mode local

# DSPy optimization + evaluation
python -m src.run_optimization
python -m src.run_optimized_eval
```

## Project Structure

```
intelligence-arbitrage/
├── src/
│   ├── config.py              # Model configs, task registry, pricing
│   ├── registry.py            # Custom model registration
│   ├── api_eval.py            # API model evaluation
│   ├── local_eval.py          # Local model evaluation
│   ├── optimize.py            # DSPy optimization pipeline
│   ├── run_baseline_v2.py     # Baseline runner
│   ├── run_optimization.py    # Optimization orchestrator
│   ├── run_optimized_eval.py  # Optimized model evaluation
│   ├── consolidate_results.py # Result aggregation
│   └── tasks/                 # Per-domain task definitions
├── experiments/               # Experiment configurations
├── results/                   # CSV results and golden summaries
│   ├── golden/               # Consolidated benchmark results
│   ├── api_baseline_*.csv    # Per-model API results
│   ├── local_baseline_*.csv  # Per-model local results
│   └── local_optimized_*.csv # DSPy-optimized results
├── tests/                     # Test suites
├── TECHNICAL_SPEC.md          # Detailed technical specification
├── run_experiment.sh          # Full experiment runner
└── setup_models.sh            # Ollama model setup
```

## Tech Stack

- **DSPy** — Declarative Self-improving Language Programs
- **Ollama** — Local model inference (Llama 3.2, Qwen 2.5, Mistral, Phi-4)
- **OpenAI API** — Frontier baselines and teacher model
- **HuggingFace Datasets** — Standard benchmark datasets
- **Pandas** — Result analysis and aggregation

## License

MIT

---

*Built with AI-assisted development tools.*
