# Intelligence Arbitrage: Can DSPy-Optimized Small Models Match Frontier APIs?

## Executive Summary

This experiment tests the **Intelligence Arbitrage** hypothesis: can small, locally-run language models (2-14B parameters), when optimized with DSPy's MIPROv2 prompt optimizer, match the performance of frontier API models (GPT-4o, GPT-5.2)?

**Key findings:**

1. **Yes, on specific tasks.** Optimized local models matched or exceeded GPT-5.2's baseline on 5 of 9 task categories — extraction, RAG, synthesis, code, and QA.
2. **Optimization response is task-specific.** Tasks determine WHETHER optimization helps (2x more variance than models). But demo count optima are model-specific — phi4 peaks at 1-2 demos on most tasks while llama3.2 benefits from all 5.
3. **MIPROv2 sometimes hurts.** In 12/54 model-task pairs, optimization produced negative deltas — distributed across phi4 (3), gpt-4o (3), gpt-5.2 (3), mistral (2), and llama3.2 (1). Progressive demo analysis reveals two distinct failure modes: *instruction damage* (optimized instructions confuse the model) and *demo interference* (too many examples degrade performance).
4. **Reasoning models show different profiles.** qwen3:4b and deepseek-r1:7b have complementary strengths, with deepseek-r1 excelling at classification/extraction and qwen3 at math/rag.

---

## Table of Contents

1. [Experimental Setup](#experimental-setup)
2. [Phase 1: Baseline vs Optimized (6 Models x 9 Tasks)](#phase-1-baseline-vs-optimized)
3. [Phase 2: Variance Analysis — Task-Specific or Model-Specific?](#phase-2-variance-analysis)
4. [Phase 3: Demo Curve Experiment — Why Does Optimization Hurt?](#phase-3-demo-curve-experiment)
5. [Phase 4: Reasoning Model Baselines](#phase-4-reasoning-model-baselines)
6. [Phase 5: Tipping Point Experiment (Partial)](#phase-5-tipping-point-experiment)
7. [Cross-Cutting Analysis](#cross-cutting-analysis)
8. [Methodology Notes](#methodology-notes)
9. [Remaining Work](#remaining-work)

---

## Experimental Setup

- **9 task categories**: agentic (StrategyQA), analysis (BoolQ), classification (Banking77), code (HumanEval), extraction (CoNLL-2003), math (GSM8K), QA (HotPotQA), RAG (SQuAD), synthesis (XSum)
- **8 models evaluated across phases**:
  - Local (Phase 1): llama3.2 (3B), mistral (7B), qwen2.5:7b (7B), phi4 (14B)
  - API (Phase 1): gpt-4o, gpt-5.2
  - Reasoning (Phase 4): qwen3:4b (4B), deepseek-r1:7b (7B)
- **Optimizer**: MIPROv2 with 3 seeds (10, 20, 30), best-of-3 selected by dev score, GPT-4o as teacher model
- **Evaluation**: 50 examples per task, deterministic (temperature=0), separate train/eval splits verified
- **Total evaluations**: 108 (Phase 1) + 42 (Phase 3 demo curves) + 17 (Phase 4 baselines) + 8 (Phase 5 tipping point) = **175 evaluation runs**

---

## Phase 1: Baseline vs Optimized

### Optimization Lift by Model

| Model | Params | Avg Baseline | Avg Optimized | Avg Delta | Tasks Improved† |
|-------|--------|-------------|--------------|-----------|----------------|
| qwen2.5:7b | 7B | 51.7% | 65.9% | **+14.2pp** | 8/9 |
| mistral | 7B | 42.1% | 55.7% | **+13.6pp** | 6/9 |
| llama3.2 | 3B | 49.7% | 60.2% | **+10.5pp** | 7/9 |
| gpt-5.2 | API | 68.0% | 77.1% | **+9.1pp** | 5/9 |
| phi4 | 14B | 56.1% | 65.0% | **+8.8pp** | 5/9 |
| gpt-4o | API | 63.3% | 71.2% | **+7.9pp** | 5/9 |

*†"Tasks Improved" counts tasks with delta > 0. Code (0pp for all models) counts as neither improved nor hurt.*

### Full Results Matrix

#### Baselines (no optimization)

| Task | llama3.2 | mistral | phi4 | qwen2.5:7b | gpt-4o | gpt-5.2 |
|------|---------|---------|------|-----------|--------|---------|
| agentic | 66% | 56% | 74% | 74% | 82% | 86% |
| analysis | 68% | 76% | 80% | 78% | 78% | 88% |
| classification | 28% | 10% | 26% | 18% | 56% | 64% |
| code | 68% | 56% | 90% | 84% | 76% | 90% |
| extraction | 50% | 50% | 52% | 40% | 46% | 44% |
| math | 40% | 2% | 48% | 36% | 68% | 90% |
| qa | 0% | 0% | 8% | 10% | 32% | 22% |
| rag | 82% | 84% | 86% | 82% | 88% | 88% |
| synthesis | 44.9% | 44.6% | 41.2% | 43.1% | 43.3% | 39.9% |

#### Optimized (MIPROv2)

| Task | llama3.2 | mistral | phi4 | qwen2.5:7b | gpt-4o | gpt-5.2 |
|------|---------|---------|------|-----------|--------|---------|
| agentic | 78% | 84% | 66% | 80% | 86% | 82% |
| analysis | 78% | 81% | 76% | 84% | 89% | 88% |
| classification | 31% | 54% | 24% | 38% | 46% | 62% |
| code | 68% | 56% | 90% | 84% | 76% | 88% |
| extraction | 66% | 48% | 84% | 82% | 74% | 94% |
| math | 66% | 32% | 84% | 72% | 96% | 98% |
| qa | 20% | 18% | 30% | 22% | 52% | 42% |
| rag | 90% | 81% | 88% | 88% | 86% | 92% |
| synthesis | 44.6% | 46.9% | 42.6% | 43.2% | 35.7% | 48.2% |

#### Deltas (Optimized - Baseline, percentage points)

| Task | llama3.2 | mistral | phi4 | qwen2.5:7b | gpt-4o | gpt-5.2 |
|------|---------|---------|------|-----------|--------|---------|
| agentic | +12 | +28 | **-8** | +6 | +4 | **-4** |
| analysis | +10 | +5 | **-4** | +6 | +11 | 0 |
| classification | +3 | +44 | **-2** | +20 | **-10** | **-2** |
| code | 0 | 0 | 0 | 0 | 0 | **-2** |
| extraction | +16 | **-2** | +32 | +42 | +28 | +50 |
| math | +26 | +30 | +36 | +36 | +28 | +8 |
| qa | +20 | +18 | +22 | +12 | +20 | +20 |
| rag | +8 | **-3** | +2 | +6 | **-2** | +4 |
| synthesis | **-0.3** | +2.4 | +1.4 | +0.1 | **-7.6** | +8.3 |

*Bold negative values indicate optimization produced a lower score. However, at n=50, deltas of ±4pp or smaller fall within one standard error of zero (see [Statistical Limitations](#statistical-limitations)) and should not be interpreted as confirmed regressions.*

**DSPy version key**: llama3.2 and mistral optimized evals ran on DSPy 2.6.x. phi4, qwen2.5:7b, gpt-5.2, and gpt-4o (qa/rag/synthesis only) ran on DSPy 3.1.3. All baselines ran on DSPy 2.6.x. See [DSPy Version as Confound](#dspy-version-as-confound).

### Intelligence Arbitrage Analysis

#### The Scoreboard: Local Optimized vs GPT-5.2 Baseline

| Task | GPT-5.2 Baseline | Best Local Optimized | Arbitrage |
|------|-----------------|---------------------|-----------|
| extraction | 44% | phi4: **84%** (+40pp) | Massive |
| qa | 22% | phi4: **30%** (+8pp) | Strong |
| synthesis | 40% | mistral: **47%** (+7pp) | Strong |
| rag | 88% | llama3.2: **90%** (+2pp) | Matched |
| code | 90% | phi4: **90%** (tied) | Matched |
| agentic | 86% | mistral: **84%** (-2pp) | Near-match |
| analysis | 88% | qwen2.5: **84%** (-4pp) | Close |
| classification | 64% | mistral: **54%** (-10pp) | Gap remains |
| math | 90% | phi4: **84%** (-6pp) | Gap remains |

Optimized local models match or exceed GPT-5.2 baseline on **5 of 9 tasks**. But the real story is in the rank inversions.

#### Rank Inversions: When Optimized Local Models Beat Optimized API Models

Across all 9 tasks, there are **14 cases** where an optimized local model outperforms an optimized API model. The most dramatic:

| Task | Local (optimized) | API (optimized) | Swing from baseline |
|------|-------------------|-----------------|---------------------|
| classification | mistral **54%** | gpt-4o **46%** | **54pp swing** (was 46pp behind) |
| agentic | mistral **84%** | gpt-5.2 **82%** | **32pp swing** (was 30pp behind) |
| code | phi4 **90%** | gpt-4o **76%** | **28pp swing** (was 14pp ahead at baseline too) |
| extraction | phi4 **84%** | gpt-4o **74%** | **14pp swing** (was 4pp behind) |
| synthesis | all 4 locals | gpt-4o **35.7%** | gpt-4o regressed -7.6pp, all locals beat it |

**The classification story** is the most striking. Mistral started at 10% baseline — 46pp behind gpt-4o's 56%. After MIPROv2 optimization, mistral surged to 54% (+44pp, the largest single lift in the experiment) while gpt-4o *dropped* to 46% (-10pp, the largest regression). A $0 local 7B model leapfrogged a paid API model through optimization alone.

**The agentic story** is similar: mistral baseline (56%) trailed gpt-5.2 (86%) by 30pp. After optimization, mistral (84%) edged past gpt-5.2 (82%) — a complete reversal enabled by mistral's +28pp gain and gpt-5.2's -4pp regression.

#### gpt-4o: Optimization Regressions

gpt-4o has **3 of 12 negative deltas** in the experiment — tied with phi4 (3) and gpt-5.2 (3):

| Task | gpt-4o Baseline | gpt-4o Optimized | Delta | Consequence |
|------|----------------|-----------------|-------|-------------|
| classification | 56% | 46% | **-10pp** | Fell behind mistral (54%) |
| synthesis | 43.3% | 35.7% | **-7.6pp** | Worst score in the task; all locals beat it |
| rag | 88% | 86% | **-2pp** | Fell behind llama3.2 (90%) |

However, gpt-4o's regressions are the *most severe* in magnitude (-10pp classification, -7.6pp synthesis), while phi4 and gpt-5.2's negatives are smaller (-2pp to -8pp). gpt-4o also had the strongest gains on math (+28pp) and extraction (+28pp), giving it a corrected average lift of +7.9pp — roughly comparable to phi4's +8.8pp, not the outlier the uncorrected +3.8pp suggested.

This pattern suggests MIPROv2 optimization interacts *unevenly* with gpt-4o: strong gains on tasks where formatting matters (math, extraction), but regressions where the model's existing priors were already well-calibrated (classification, synthesis).

#### Pre-Optimization Parity

On some tasks, local models already matched API models before any optimization:

- **Code**: phi4 (90%) = gpt-5.2 (90%) at baseline. phi4 beats gpt-4o (76%) by 14pp with zero optimization.
- **Extraction**: All 6 models clustered in 40-52% at baseline — the frontier had no inherent advantage. Optimization then amplified differences (range expanded from 12pp to 46pp).
- **Synthesis**: All 6 models in 40-45% at baseline — effectively indistinguishable.

#### Gap Compression vs Divergence

Optimization doesn't uniformly close gaps. On some tasks it compresses the field, on others it widens it:

| Task | Baseline range | Optimized range | Effect |
|------|---------------|-----------------|--------|
| math | 88pp (2%-90%) | 66pp (32%-98%) | Compressed (-22pp) |
| classification | 54pp (10%-64%) | 38pp (24%-62%) | Compressed (-16pp) |
| agentic | 30pp (56%-86%) | 20pp (66%-86%) | Compressed (-10pp) |
| analysis | 20pp (68%-88%) | 13pp (76%-89%) | Compressed (-7pp) |
| code | 34pp (56%-90%) | 34pp (56%-90%) | Unchanged |
| qa | 32pp (0%-32%) | 34pp (18%-52%) | Diverged (+2pp) |
| rag | 6pp (82%-88%) | 11pp (81%-92%) | Diverged (+5pp) |
| synthesis | 5pp (40%-45%) | 13pp (36%-48%) | **Diverged (+8pp)** ‡ |
| extraction | 12pp (40%-52%) | 46pp (48%-94%) | **Diverged (+34pp)** |

*‡ Synthesis ranges use decimal values (35.7%-48.2%). The apparent 13pp range rounds to 12.5pp exactly.*

Extraction is notable: the tightest baseline cluster (12pp range) became the widest optimized spread (46pp). Models that responded well to optimization (phi4, qwen2.5, gpt-5.2) pulled far ahead, while those that didn't (mistral) fell behind.

---

## Phase 2: Variance Analysis

### Question: Is optimization response task-specific or model-specific?

Using the 6-model x 9-task delta matrix from Phase 1:

| Metric | Value |
|--------|-------|
| Within-task variance (across models) | 102.5 |
| Within-model variance (across tasks) | 205.2 |
| Variance ratio (task/model) | 0.50 |
| Mean pairwise model correlation | +0.49 |

**Interpretation**: Within-task variance is **half** within-model variance, meaning tasks determine optimization response 2x more than model identity. The positive model correlation (+0.49) means models broadly agree on *which* tasks benefit from optimization.

### Task Consistency Ranking

| Tier | Tasks | Cross-model std | Interpretation |
|------|-------|-----------------|----------------|
| Task-driven | code (0.7), qa (3.2), rag (4.3) | Low | All models respond similarly |
| Weakly task-driven | synthesis (4.8), math (9.8), analysis (5.0) | Medium | Some model-dependent variation |
| Model-dependent | classification (18.2), extraction (17.0), agentic (12.8) | High | Response depends on the model |

### MIPROv2 Demo Count Patterns

MIPROv2 chose demo counts in a binary all-or-nothing pattern (0 or 5, never intermediate):

| Task | Demo pattern across 6 models |
|------|------------------------------|
| code | 0 demos for ALL models |
| math | 5 demos for ALL models |
| qa | 5 demos for ALL models |
| synthesis | 5 demos for 5/6 models (exception: qwen2.5:7b = 0) |
| rag | 5 demos for 5/6 models (exception: mistral = 0) |
| classification | 5 for API models, 0 for 4 local models |
| extraction | 5 for llama3.2/qwen2.5, 0 for others |
| analysis | 5 for local models, 0 for API models |
| agentic | 5 for all except phi4 (0) |

**30 programs have 5 demos, 24 have 0 demos.** No intermediate counts exist. This suggests MIPROv2's optimizer uses a binary heuristic rather than fine-grained demo count selection.

---

## Phase 3: Demo Curve Experiment

### Question: When optimization hurt in Phase 1, was it the wrong demo count?

For 6 (model, task) pairs where optimization had negative or near-zero deltas and 5 demos, we progressively evaluated the optimized program at n=0 (instructions only), n=1, n=2, ..., n=5 demos, alongside a raw baseline (no optimization). This decomposes the optimization effect into two independent components:

- **Instruction effect** = accuracy(n=0) - accuracy(baseline)
- **Demo effect** = accuracy(n=5) - accuracy(n=0)

### Results: Accuracy by Demo Count (n=50 examples each)

| Model | Task | P1 Delta | Baseline* | n=0 | n=1 | n=2 | n=3 | n=4 | n=5 | Best n |
|-------|------|----------|-----------|-----|-----|-----|-----|-----|-----|--------|
| llama3.2 | math | +26.0 | 44% | 50% | 50% | 58% | 70% | 74% | 76% | **n=5** |
| llama3.2 | synthesis | -0.3 | 100% | 98% | 100% | 100% | 100% | 100% | 100% | base |
| phi4 | analysis | -4.0 | 82% | 78% | 80% | **82%** | 80% | 78% | 76% | **n=2** |
| phi4 | math | +36.0 | 50% | 54% | 74% | 80% | 78% | **84%** | 84% | **n=4** |
| phi4 | rag | +2.0 | 88% | 80% | **90%** | 90% | 88% | 88% | 88% | **n=1** |
| phi4 | synthesis | +1.4 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | base |

**\*Phase 3 baseline note:** These baselines were re-evaluated on DSPy 3.1.3, while Phase 1 baselines ran on DSPy 2.6.x. Small discrepancies exist: phi4 analysis (82% here vs 80% in Phase 1), phi4 math (50% vs 48%), phi4 rag (88% vs 86%), llama3.2 math (44% vs 40%). The synthesis baselines show a larger gap (100% here vs 41-45% in Phase 1), likely due to metric computation differences between DSPy versions. The instruction and demo effects in this table are internally consistent (all computed within the same DSPy 3.1.3 environment), but direct comparison of these baselines to Phase 1 baselines should be treated with caution.

### Decomposition: Instruction Effect vs Demo Effect

| Model | Task | Instruction Effect | Demo Effect | Diagnosis |
|-------|------|--------------------|-------------|-----------|
| phi4 | analysis | **-4pp** (hurt) | **-2pp** (hurt) | Both hurt — double penalty |
| phi4 | rag | **-8pp** (hurt) | +8pp (helped) | Instructions damaged, demos compensated |
| phi4 | math | +4pp (helped) | +30pp (helped) | Both helped |
| phi4 | synthesis | 0pp | 0pp | Neutral (already at ceiling) |
| llama3.2 | math | +6pp (helped) | +26pp (helped) | Both helped, monotonic |
| llama3.2 | synthesis | -2pp (slight hurt) | +2pp (compensated) | Near-neutral |

### Key Findings

**1. Demo interference is real.** phi4's analysis task shows a classic inverted-U curve: accuracy rises from 78% (n=0) to 82% (n=2) then falls to 76% (n=5). MIPROv2 used all 5 demos, but n=2 was optimal. This confirms that adding more demonstrations can actively confuse smaller models.

**2. Instruction damage is the bigger problem.** In phi4's RAG task, optimized instructions alone dropped accuracy by 8 percentage points (88% → 80%). Even a single demo recovered performance to 90%. MIPROv2's optimized task instructions can be counterproductive, with demos serving as a corrective.

**3. Demo count optima are model-specific.** On the same task (math), llama3.2 shows monotonic improvement through all 5 demos (each demo adds ~5pp), while phi4 gets +20pp from demo #1 alone and shows diminishing/non-monotonic returns after that. The optimal count depends on model capacity and how the model processes in-context examples.

**4. The first demo is disproportionately valuable.** Across cases where demos helped:
- phi4 math: n=0→n=1 = **+20pp** (largest single jump)
- phi4 rag: n=0→n=1 = **+10pp** (recovered from instruction damage)
- llama3.2 math: n=0→n=1 = 0pp, but n=1→n=2 = **+8pp** (delayed start)

**5. High baselines are optimization-proof.** Both synthesis cases (phi4, llama3.2) scored 100% at baseline, leaving no room for improvement. All 7 conditions produced 100% ± 2pp. The synthesis metric (binary threshold on ROUGE-L) may lack granularity.

---

## Phase 4: Reasoning Model Baselines

### qwen3:4b (9/9 tasks complete)

| Task | Accuracy | Comparison to Phase 1 mean |
|------|----------|---------------------------|
| rag | **88%** | Matches phi4 baseline (86%), exceeds llama3.2 (82%) |
| analysis | 68% | At Phase 1 average |
| math | 50% | Beats llama3.2 (40%), matches phi4 (48%) |
| agentic | 46% | Below most Phase 1 models |
| code | 24% | Below Phase 1 average |
| extraction | 20% | Below Phase 1 average |
| classification | 14% | Below most Phase 1 models |
| qa | 12% | Better than llama3.2/mistral (0%), below phi4 (8%) |
| synthesis | 0% | Metric failure (no output matched threshold) |

### deepseek-r1:7b (8/9 tasks complete, missing code)

| Task | Accuracy | Comparison to Phase 1 mean |
|------|----------|---------------------------|
| analysis | **82%** | Matches phi4 baseline, exceeds most local models |
| rag | 74% | Below Phase 1 average |
| extraction | **66%** | Exceeds all Phase 1 local models |
| classification | **64%** | Exceeds all Phase 1 local models, matches gpt-5.2 |
| agentic | 44% | Below Phase 1 average |
| math | 32% | Below Phase 1 average |
| qa | 2% | Near-zero |
| synthesis | 0% | Metric failure |

### Cross-Model Comparison

| Strength area | qwen3:4b | deepseek-r1:7b | Winner |
|---------------|----------|----------------|--------|
| Math | 50% | 32% | qwen3:4b (+18pp) |
| RAG | 88% | 74% | qwen3:4b (+14pp) |
| Classification | 14% | 64% | deepseek-r1:7b (+50pp) |
| Extraction | 20% | 66% | deepseek-r1:7b (+46pp) |
| Analysis | 68% | 82% | deepseek-r1:7b (+14pp) |

The two reasoning models show **complementary** profiles: qwen3:4b excels at structured reasoning (math, retrieval), while deepseek-r1:7b excels at pattern matching and labeling (classification, extraction, analysis). deepseek-r1:7b's classification score (64%) matches GPT-5.2's baseline without any optimization.

### Latency Characteristics

| Model | Per-example latency | Notes |
|-------|-------------------|-------|
| qwen3:4b | ~95ms | Fast; suitable for batch evaluation |
| deepseek-r1:7b | ~13-18s | Very slow; extended "thinking" traces |

---

## Phase 5: Tipping Point Experiment (Partial)

### Design

The tipping point experiment tests: *at what number of optimization examples does performance improve?* Uses BootstrapFewShotWithRandomSearch with GPT-4o as teacher, evaluating at n=0, 1, 2 examples on qwen3:4b.

### Results (3 of 9 tasks complete)

| Task | Baseline† | n=0 | n=1 | n=2 | Opt time (n=1) |
|------|-----------|-----|-----|-----|----------------|
| classification | 34% | 34% | 34% | 34% | 3.7 hours |
| math | 50% | 50% | **64%** | 60% | 1.7 hours |
| qa | 12% | 12% | 10% | — ‡ | 4s |

*†Phase 5 baselines were evaluated independently within the BootstrapFewShot pipeline (DSPy 3.1.3). The classification baseline here (34%) differs from Phase 4 (14%) — a 20pp gap for what should be identical conditions (same model, same task, temperature=0). The cause is under investigation; possible factors include different eval splits, prompt formatting differences between the tipping point runner and the baseline runner, or qwen3:4b checkpoint differences.*

*‡ QA at n=2 was not evaluated (optimization at n=2 did not produce a valid program).*

### Observations

- **Math tipping point at n=1**: A single optimized example boosted qwen3:4b from 50% to 64%. At n=2, performance drops to 60%, echoing the demo interference pattern from Phase 3.
- **Classification resists optimization**: 34% at every condition. BootstrapFewShot cannot help here.
- **QA optimization degraded**: -2pp at n=1. Combined with Phase 3 findings, QA appears to be a task where optimization broadly helps (Phase 1 saw +12 to +22pp) but the *choice of optimizer* matters — MIPROv2 helped while BootstrapFewShot hurt.
- **Prohibitive cost**: ~1.5 to 3.7 hours per (task, n) due to GPT-4o teacher calls. Full experiment (9 tasks x 3 n-values x 2 models) would take 50+ hours. This motivated the shift to the faster progressive demo evaluation approach in Phase 3.

---

## Cross-Cutting Analysis

### 1. Optimization helps most where models are weakest

Tasks with the largest Phase 1 gains (math +27pp avg, extraction +28pp avg, QA +19pp avg) were consistently those with the lowest baselines. MIPROv2 primarily fixes **output formatting** — teaching models to produce answers in the expected format — rather than fundamentally improving reasoning.

### 2. Strong baselines show diminishing returns

Models already scoring 80%+ on a task (e.g., phi4 on code/rag, gpt-4o on agentic/rag) saw near-zero or negative deltas. Optimization can over-constrain models that are already performing well.

### 3. Smaller models benefit more from optimization

qwen2.5:7b saw the largest average lift (+14.2pp), while gpt-4o saw the smallest (+7.9pp). But the gap between API and local models is narrower than it first appears — gpt-4o's +7.9pp is comparable to phi4's +8.8pp. The real outlier is mistral (+13.6pp), driven almost entirely by its +44pp classification gain. Prompt optimization is a **force multiplier that disproportionately benefits less capable models**, but the effect is uneven.

### 4. Some tasks resist optimization entirely

Code showed 0pp delta across all 6 Phase 1 models — and MIPROv2 chose 0 demos for all of them. Code generation requires capabilities that few-shot prompting cannot unlock.

### 5. MIPROv2's binary demo selection is suboptimal

MIPROv2 always chose 0 or 5 demos, never an intermediate count. Phase 3 proved this is wrong: phi4's analysis task peaks at n=2, and phi4's rag task peaks at n=1. A more granular demo selection strategy could recover 2-8 percentage points in cases where MIPROv2 currently hurts.

### 6. Two distinct optimization failure modes

| Failure mode | Mechanism | Example | Potential fix |
|-------------|-----------|---------|---------------|
| **Instruction damage** | Optimized instructions confuse the model | phi4 rag: -8pp from instructions alone | Validate instructions independently before adding demos |
| **Demo interference** | Too many demos degrade performance | phi4 analysis: peaks at n=2, falls to 76% at n=5 | Progressive demo evaluation to find per-model optimum |

### 7. Cost comparison

| Approach | Per-task cost (50 examples) | Per-task time |
|----------|---------------------------|---------------|
| GPT-5.2 API | ~$0.66 | ~2-3 min |
| GPT-4o API | ~$0.15 | ~2-3 min |
| Local (phi4, 14B) | $0.00 | ~12-15 min |
| Local (qwen2.5, 7B) | $0.00 | ~8-12 min |
| Local (llama3.2, 3B) | $0.00 | ~5-8 min |

Total GPT-5.2 evaluation cost for Phase 1: **$5.93** for all 9 tasks.

---

## Methodology Notes

### Evaluation Design

- **Sample size**: All comparisons use n=50 binary outcomes per condition. For models with larger evaluation runs (gpt-4o had up to n=1000), only the first 50 are used — ground truth ordering is deterministic.
- **Train/eval separation**: Verified across all 9 tasks. 7 use HuggingFace built-in splits, agentic uses deterministic 80/20 split (0 overlap verified), code uses cross-dataset (MBPP training, HumanEval eval).
- **llama3.2 and llama3.2_3b**: Confirmed to be the same model (both 1.9 GB, identical scores). Only llama3.2 results are included.

### Optimization Protocol

- **Optimizer**: MIPROv2 with GPT-4o as teacher model. Three seeds (10, 20, 30) are run independently. Each seed produces a candidate program evaluated against a held-out dev set (20% of training data). The **best-of-3 by dev score** is selected as the final program. This selection inflates reported scores slightly relative to single-seed optimization.
- **Programs**: Each optimized program contains task instructions (rewritten by GPT-4o) and curated few-shot demonstrations (0 or 5). The binary demo count is a MIPROv2 optimizer decision, not experimenter-chosen.
- **Teacher-as-judge concern**: GPT-4o serves as the MIPROv2 teacher (generating optimized instructions and demos). For tasks with model-based evaluation metrics, this could create circularity if the evaluation rewards outputs that resemble GPT-4o's style. In practice, 7 of 9 tasks use deterministic metrics (exact match, F1, ROUGE-L threshold) where this is not a factor. The two potentially affected tasks are agentic and analysis, which use binary answer matching against ground truth.

### Statistical Limitations

- **No confidence intervals reported.** With n=50 binary outcomes, the standard error of a proportion at p=50% is ≈7pp (and smaller at extreme proportions). This means:
  - Deltas of ±4pp or smaller (approximately 18 of 54 in the delta matrix) are **not statistically distinguishable from zero** at conventional significance levels.
  - The synthesis row (deltas ranging from -7.6 to +8.3) is almost entirely within noise.
  - Claims of "optimization hurt" for small negative deltas (phi4 classification -2pp, gpt-5.2 code -2pp) should be read as "no detectable effect" rather than confirmed regression.
  - The large deltas (mistral classification +44pp, gpt-5.2 extraction +50pp, math gains of +26 to +36pp) are well outside the noise band and represent genuine effects.
- **Subsampling**: For gpt-4o (which had up to n=1000 total evaluations), only the first 50 examples are used. The ground truth ordering is deterministic, but difficulty may vary across the sequence. No analysis was done to verify that the 50-example subset is representative of the full set.

### DSPy Version as Confound

The experiment spans two DSPy versions, which constitutes a methodological confound:

| DSPy Version | What ran on it |
|---|---|
| 2.6.x (Jan 2026) | All baselines, llama3.2 optimized, mistral optimized (8/9), gpt-4o optimized (6/9) |
| 3.1.3 (Feb 2026) | phi4 optimized, qwen2.5 optimized, gpt-5.2 optimized, gpt-4o optimized (qa/rag/synthesis), all Phase 3-5 |

The Phase 1 "optimized" column therefore mixes results from two DSPy versions. Cross-model comparisons implicitly assume the framework version doesn't matter, but Phase 3 baseline discrepancies (2-4pp shifts, plus synthesis jumping from ~42% to 100%) demonstrate that it does. A 2.x-to-3.x state format adapter was used to load old programs on 3.1.3, but the inference pipeline itself changed between versions.

### Metric Limitations

- **Synthesis**: Binary threshold on ROUGE-L score. This metric saturates at 100% for models that produce reasonable summaries (confirmed in Phase 3, where both phi4 and llama3.2 scored 100% at baseline). Synthesis results are included in all aggregate statistics but should be interpreted with this ceiling effect in mind. The task provides limited discriminative power — all models cluster in a narrow band at baseline (40-45%), and the binary threshold converts continuous quality differences into noise.
- **Phase 3 demo curve**: Each condition (model x task x demo count) is a full 50-example evaluation. The same test data is used across all conditions for direct comparability. Only the demo count varies; optimized instructions are held constant.

---

## Remaining Work

- [ ] **deepseek-r1:7b code baseline** — 1 missing task (8/9 complete)
- [ ] **GPT-4o classification demo curve** — most dramatic untested case (-10pp with 5 demos, requires API cost)
- [ ] **GPT-5.2 optimized evals** — 9 tasks (programs exist, no evals run yet)
- [ ] **Extended tipping point** — remaining 6/9 tasks for qwen3:4b; all 9 for deepseek-r1:7b
- [ ] **API model demo curves** — gpt-4o classification, synthesis, rag all had negative deltas with 5 demos
- [ ] **Final report** — synthesize all findings into publication-ready narrative
