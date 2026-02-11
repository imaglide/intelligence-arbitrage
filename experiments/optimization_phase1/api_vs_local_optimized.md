# Intelligence Arbitrage Report
> Analysis generated: 2026-01-23 12:56:32.009526

## 1. Intelligence Arbitrage (Wins & Catch-ups)
Where local models either beat the API or significantly closed the gap.

### 🏆 Wins (Local > API)
| Task | Winner (Local) | Score | vs API Best | Delta |
|---|---|---|---|---|
| extraction | **mistral** | 53.8% | gpt-5.2-chat-latest (52.0%) | +1.8% |
| extraction | **llama3.2** | 56.8% | gpt-5.2-chat-latest (52.0%) | +4.8% |
| rag | **llama3.2** | 91.7% | gpt-4o (88.0%) | +3.7% |
| synthesis | **llama3.2** | 44.8% | gpt-4o (43.3%) | +1.5% |

### 🚀 Gap Closers (High Improvement)
| Task | Contender (Local) | Opt Score | Previous Baseline | vs API Best | Note |
|---|---|---|---|---|---|
| analysis | llama3.2 | **78.4%** | 68.0% | gpt-5.2 (-9.6%) | Closed gap by 10.4% |
| classification | mistral | **53.7%** | 10.0% | gpt-5.2-chat-latest (-18.3%) | Closed gap by 43.7% |
| math | mistral | **35.6%** | 2.0% | gpt-5.2 (-54.4%) | Closed gap by 33.6% |
| math | llama3.2 | **71.9%** | 40.0% | gpt-5.2 (-18.1%) | Closed gap by 31.9% |
| qa | mistral | **18.9%** | 0.0% | gpt-4o (-13.1%) | Closed gap by 18.9% |
| qa | llama3.2 | **15.4%** | 0.0% | gpt-4o (-16.6%) | Closed gap by 15.4% |

## 2. Detailed Internal Deltas (Optimized vs Unoptimized)
Performance improvement for every model/task pair.
| Task           | Model      | Baseline | Optimized | Internal Delta | vs API Best |
| -------------- | ---------- | -------- | --------- | -------------- | ----------- |
| agentic        | llama3.2   | 66.0%    | **65.7%** | 📉 -0.3%        | -26.3%      |
| agentic        | mistral    | 56.0%    | **65.5%** | 📈 +9.5%        | -26.5%      |
| agentic        | phi4       | 74.0%    | -         | -              | -           |
| agentic        | qwen2.5_7b | 74.0%    | -         | -              | -           |
| analysis       | llama3.2   | 68.0%    | **78.4%** | 📈 +10.4%       | -9.6%       |
| analysis       | mistral    | 76.0%    | **80.8%** | 📈 +4.8%        | -7.2%       |
| analysis       | phi4       | 80.0%    | -         | -              | -           |
| analysis       | qwen2.5_7b | 78.0%    | -         | -              | -           |
| classification | llama3.2   | 28.0%    | **31.1%** | 📈 +3.1%        | -40.9%      |
| classification | mistral    | 10.0%    | **53.7%** | 📈 +43.7%       | -18.3%      |
| classification | phi4       | 26.0%    | -         | -              | -           |
| classification | qwen2.5_7b | 18.0%    | -         | -              | -           |
| code           | llama3.2   | 68.0%    | **50.6%** | 📉 -17.4%       | -41.4%      |
| code           | mistral    | 56.0%    | **34.8%** | 📉 -21.2%       | -57.2%      |
| code           | phi4       | 90.0%    | -         | -              | -           |
| code           | qwen2.5_7b | 84.0%    | -         | -              | -           |
| extraction     | llama3.2   | 50.0%    | **56.8%** | 📈 +6.8%        | 🏆 +4.8%     |
| extraction     | mistral    | 50.0%    | **53.8%** | 📈 +3.8%        | 🏆 +1.8%     |
| extraction     | phi4       | 52.0%    | -         | -              | -           |
| extraction     | qwen2.5_7b | 40.0%    | -         | -              | -           |
| math           | llama3.2   | 40.0%    | **71.9%** | 📈 +31.9%       | -18.1%      |
| math           | mistral    | 2.0%     | **35.6%** | 📈 +33.6%       | -54.4%      |
| math           | phi4       | 48.0%    | -         | -              | -           |
| math           | qwen2.5_7b | 36.0%    | -         | -              | -           |
| qa             | llama3.2   | 0.0%     | **15.4%** | 📈 +15.4%       | -16.6%      |
| qa             | mistral    | 0.0%     | **18.9%** | 📈 +18.9%       | -13.1%      |
| qa             | phi4       | 8.0%     | -         | -              | -           |
| qa             | qwen2.5_7b | 10.0%    | -         | -              | -           |
| rag            | llama3.2   | 82.0%    | **91.7%** | 📈 +9.7%        | 🏆 +3.7%     |
| rag            | mistral    | 84.0%    | **80.8%** | 📉 -3.2%        | -7.2%       |
| rag            | phi4       | 86.0%    | -         | -              | -           |
| rag            | qwen2.5_7b | 82.0%    | -         | -              | -           |
| synthesis      | llama3.2   | 44.9%    | **44.8%** | 📉 -0.1%        | 🏆 +1.5%     |
| synthesis      | mistral    | 44.6%    | -         | -              | -           |
| synthesis      | phi4       | 41.2%    | -         | -              | -           |
| synthesis      | qwen2.5_7b | 43.1%    | -         | -              | -           |

## 3. Analysis of Regressions (Why did some scores drop?)
Investigating the negative deltas in **Code** (-17% to -21%) and **Agentic** tasks.
- **Cause:** The Optimized Programs for these tasks (`llama3.2_code_mipro_v2.json`, `mistral_code_mipro_v2.json`) were found to be **empty** (containing 0 few-shot examples).
- **Reason:** The small local models likely failed to generate *any* passing code during the strict training phase (MBPP dataset). Because the optimizer couldn't find any high-scoring examples to use as demos, it defaulted to a 0-shot prompt.
- **Conclusion:** Evaluation-based optimization (Student-as-Teacher) fails for hard tasks where the student cannot solve the problem 0-shot. These tasks require a **Teacher (GPT-4)** to generate the initial training traces.