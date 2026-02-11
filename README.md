# Intelligence Arbitrage

Can smaller, locally-running language models match or exceed frontier API models when optimized with DSPy? This project systematically tests the "intelligence arbitrage" hypothesis across 9 cognitive task domains.

## Hypothesis

DSPy's optimization pipeline (teacher bootstrapping + MIPROv2) can close the performance gap between local open-weight models (3B-7B parameters) and frontier API models (GPT-4o, GPT-5.2) on structured NLP tasks.

## Key Findings

1. **Yes, on specific tasks.** Optimized local models matched or exceeded GPT-5.2 baseline on 5 of 9 task categories
2. **Optimization response is task-specific.** Tasks determine WHETHER optimization helps (2x more variance than models)
3. **MIPROv2 sometimes hurts.** In 12/54 model-task pairs, optimization produced negative deltas
4. **175 total evaluation runs** across 5 experiment phases

### Optimization Lift by Model

| Model | Params | Avg Baseline | Avg Optimized | Avg Delta | Tasks Improved |
|-------|--------|-------------|--------------|-----------|----------------|
| qwen2.5:7b | 7B | 51.7% | 65.9% | **+14.2pp** | 8/9 |
| mistral | 7B | 42.1% | 55.7% | **+13.6pp** | 6/9 |
| llama3.2 | 3B | 49.7% | 60.2% | **+10.5pp** | 7/9 |
| gpt-5.2 | API | 68.0% | 77.1% | **+9.1pp** | 5/9 |
| phi4 | 14B | 56.1% | 65.0% | **+8.8pp** | 5/9 |
| gpt-4o | API | 63.3% | 71.2% | **+7.9pp** | 5/9 |

See [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) for the full write-up with all 5 phases, and `results/` for per-model CSVs.

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
│   ├── demo_curve_eval.py     # Demo count curve analysis
│   ├── results_writer.py      # Structured result output
│   ├── tasks/                 # Per-domain task definitions
│   └── tipping_point/         # Tipping point experiment modules
├── experiments/               # Experiment configurations
├── results/                   # CSV results and golden summaries
│   ├── golden/               # Consolidated benchmark results
│   ├── demo_curves/          # Demo count optimization curves
│   ├── tipping_point/        # Tipping point experiment results
│   ├── api_baseline_*.csv    # Per-model API results
│   ├── local_baseline_*.csv  # Per-model local results
│   ├── local_optimized_*.csv # DSPy-optimized results
│   └── api_optimized_*.csv   # API-optimized results
├── tests/                     # Test suites
├── EXPERIMENT_RESULTS.md      # Full experiment write-up (5 phases)
├── TECHNICAL_SPEC.md          # Detailed technical specification
├── run_experiment.sh          # Full experiment runner
└── setup_models.sh            # Ollama model setup
```

## Tech Stack

- **DSPy** — Declarative Self-improving Language Programs
- **Ollama** — Local model inference (Llama 3.2, Qwen 2.5, Mistral, Phi-4, DeepSeek-R1, Qwen3)
- **OpenAI API** — Frontier baselines and teacher model
- **HuggingFace Datasets** — Standard benchmark datasets
- **Pandas** — Result analysis and aggregation

## License

MIT

---

*Built with AI-assisted development tools.*
