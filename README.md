<div align="center">

<img src="assets/toolweave-mark.svg" alt="ToolWeave mark" width="130">

# ToolWeave

**🧵 Boundary-Guided Verified Tool-Use Synthesis and Agentic Reinforcement Learning for Multi-Turn Tool-Calling Agents.**

ToolWeave trains multi-turn tool-use agents through a staged curriculum, then closes the loop between policy learning and verified online data synthesis by detecting capability-boundary tasks and generating new executable tool-use trajectories.

[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Stage1%20%7C%20Stage2%20%7C%20Stage3-yellow)](#models)
[![Training Data](https://img.shields.io/badge/%F0%9F%A4%97%20Training%20Data-RODS_EnvTuning-yellow)](https://github.com/inclusionAI/AWorld-RL/tree/main/EnvTuning/data)
[![Eval Data](https://img.shields.io/badge/%F0%9F%A4%97%20Eval%20Data-RODS_BFCL_V3-yellow)](https://github.com/inclusionAI/AWorld-RL/tree/main/EnvTuning/data)
[![Code](https://img.shields.io/badge/GitHub-Code-181717?logo=github&logoColor=white)](https://github.com/Muradil-mamat-211/ToolWeave)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Agentic RL](https://img.shields.io/badge/Agentic-RL-6D28D9)](#stage-3-training-branch)
[![Tool Calling](https://img.shields.io/badge/Tool-Calling-0EA5E9)](#overview)
[![BFCL](https://img.shields.io/badge/BFCL-Multi--Turn-F59E0B)](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)

🤗 [ToolWeave Stage 1 Model](https://huggingface.co/muradil211/stage1) |
🤗 [ToolWeave Stage 2 Model](https://huggingface.co/muradil211/stage2) |
🤗 [ToolWeave Stage 3 Reference](https://huggingface.co/muradil211/stage3)

</div>

> [!IMPORTANT]
> ToolWeave is a project-level framework. It is **not** the official RODS implementation. The project distinguishes upstream EnvTuning/RODS concepts, reused public BFCL/EnvTuning infrastructure, and ToolWeave-specific extensions and robustness layers.

## Table of Contents

- [Overview](#overview)
- [Training Pipeline](#training-pipeline)
- [Models](#models)
- [Data](#data)
- [Stage 1 and Stage 2 Training](#stage-1-and-stage-2-training)
- [Reward and Diagnostic Contract](#reward-and-diagnostic-contract)
- [Checkpoint Validation](#checkpoint-validation)
- [Selected Checkpoints on Eval-400](#selected-checkpoints-on-eval-400)
- [Stage 3 Training Branch](#stage-3-training-branch)
- [Why ToolWeave?](#why-toolweave)
- [Open Resources](#open-resources)
- [Quick Start](#quick-start)
- [Repository Layout](#repository-layout)
- [Roadmap](#roadmap)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Overview

ToolWeave studies how an agent can improve multi-turn tool use by learning from executable environment interaction and by generating new training situations exactly where its current capability is uncertain. The central loop is:

```text
Train → Detect Boundary → Generate → Validate → Replay → Train
```

<div align="center">
<img src="assets/toolweave-pipeline.svg" alt="ToolWeave implementation-aligned Stage 1 to Stage 3 training, synthesis, validation, replay, and policy-optimization pipeline" width="100%">
</div>

The framework combines:

- multi-turn tool calling with real environment interaction;
- a Stage 1 → Stage 2 → Stage 3 curriculum;
- Progress Reward for task-level learning;
- capability-boundary detection from grouped rollouts;
- online synthetic data generation;
- strict execution and semantic validation;
- RODS-style global trajectory learning; and
- MatchTIR-inspired local tool-call credit.

### Scope and provenance

| Layer | Role in ToolWeave |
|---|---|
| Upstream EnvTuning / RODS concepts | Environment-tuning curriculum, progress-based learning, boundary-focused online synthesis, and dynamic replay ideas |
| Reused public infrastructure | BFCL multi-turn data/environment components, EnvTuning interfaces, and the veRL training stack |
| ToolWeave-specific implementation | Boundary lifecycle integration, deterministic semantic guards, fresh-VM verification, replay admission rules, and the global-plus-local credit adaptation |

These layers are intentionally documented separately. In particular, project guards and the local credit branch must not be read as claims about the official RODS algorithm.

## Training Pipeline

| Stage | Name | Training signal | Starting model | Selected output |
|---|---|---|---|---|
| Stage 1 | Tool-Use Cold Start | EnvTuning format + executable tool-call score | Qwen3-4B | ToolWeave Stage 1, update 25 |
| Stage 2 | Progress-Reward RL | Fixed-denominator terminal Progress Reward | ToolWeave Stage 1, update 25 | ToolWeave Stage 2, update 25 |
| Stage 3 | Boundary-Guided Online RL | Unchanged global Progress Reward + tool-local residual | ToolWeave Stage 2, update 25 | Implemented branch; formal training and release pending |

The starting checkpoint is recorded by the workspace as `Qwen/Qwen3-4B`; the README keeps that exact model identity rather than silently relabeling it.

## Models

| Model | Stage | Description | Status | Link |
|---|---|---|---|---|
| `ToolWeave-Stage1-4B` | Stage 1 | Selected merged update-25 checkpoint after the Stage 1 gate | Public checkpoint available; release documentation pending | [Hugging Face repository](https://huggingface.co/muradil211/stage1) |
| `ToolWeave-Stage2-4B` | Stage 2 | Merged update-25 checkpoint initialized from Stage 1 update 25 | Public checkpoint available; release documentation pending | [Hugging Face repository](https://huggingface.co/muradil211/stage2) |
| `ToolWeave-Stage3-4B` | Stage 3 reference | Public `Qwen3-4B-RODS` checkpoint currently stored in the Stage 3 repository; formal ToolWeave Stage 3 training is a separate pending step | Public reference checkpoint; not a ToolWeave final model | [Hugging Face repository](https://huggingface.co/muradil211/stage3) |

Stage 3 is not presented as a released final ToolWeave model. Its link points to the public `Qwen3-4B-RODS` reference checkpoint currently stored in that repository; formal ToolWeave Stage 3 training and certification remain pending. The model links do not imply that the full ToolWeave training code or reproducibility package has already been released.

## Data

ToolWeave does **not** rehost upstream EnvTuning or RODS BFCL training data in this repository. The project references the public sources and keeps generated candidates release-gated until their provenance and validation contract are ready.

| Resource | Source | Role in ToolWeave |
|---|---|---|
| BFCL V3 Multi-Turn source | [BFCL dataset](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) and [BFCL repository data](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard/bfcl_eval/data) | Original benchmark source for the four multi-turn categories |
| Stage 1/2 training data | [AWorld-RL `bfcl_train_base.parquet`](https://github.com/inclusionAI/AWorld-RL/blob/main/EnvTuning/data/bfcl_train_base.parquet) | 100 Base multi-turn interaction rows; the canonical public training split for Stage 1/2 |
| Stage 3 seed training data | [AWorld-RL `bfcl_train.parquet`](https://github.com/inclusionAI/AWorld-RL/blob/main/EnvTuning/data/bfcl_train.parquet) | 400 human seed rows: 100 each for Base, Missing Function, Missing Parameter, and Long Context |
| EnvTuning validation data | [AWorld-RL `bfcl_val.parquet`](https://github.com/inclusionAI/AWorld-RL/blob/main/EnvTuning/data/bfcl_val.parquet) | Validation only; not a training dataset |
| RODS / BFCL V3 held-in evaluation split | [AWorld-RL `bfcl_val.parquet`](https://github.com/inclusionAI/AWorld-RL/blob/main/EnvTuning/data/bfcl_val.parquet) + [`bfcl_test.parquet`](https://github.com/inclusionAI/AWorld-RL/blob/main/EnvTuning/data/bfcl_test.parquet) | 400 rows total: 100 each for Base, Missing Function, Missing Parameter, and Long Context |
| EnvTuning infrastructure | [AWorld-RL / EnvTuning](https://github.com/inclusionAI/AWorld-RL/tree/main/EnvTuning) | Public multi-turn environment-tuning reference and infrastructure |
| RODS resources and benchmark setup | [AWorld-RL / RODS](https://github.com/inclusionAI/AWorld-RL/tree/main/RODS) | Stage 3 boundary detection, synthesis, and replay reference |
| Generated ToolWeave candidates | ToolWeave project | Online validated Stage 3 replay data; **release pending** |

These are **RL interaction training datasets**, not supervised trajectory corpora. Their parquet entries provide prompts, tool/environment metadata, and reward-side ground truth for executable multi-turn interaction; ToolWeave does not claim to train by trajectory imitation here.

### Local training data used by ToolWeave

The local preparation preserves the official EnvTuning sample membership while adding the project’s executable protocol alignment and, where noted, a deterministic row-order shuffle:

| Stage | Local training file | Composition | SHA256 | Status |
|---|---|---|---|---|
| Stage 1 | `bfcl_stage1_train_base_100_shuffled_seed42.parquet` | 100 Base rows | `d02122551606f616c5d9d6b2915113e8266872078906c58cdefbc97ea198bf5d` | Used for the selected Stage 1 training run |
| Stage 2 | `bfcl_stage1_train_base_100_shuffled_seed42.parquet` | The same 100 Base rows | `d02122551606f616c5d9d6b2915113e8266872078906c58cdefbc97ea198bf5d` | Used for Stage 2 Progress-Reward RL |
| Stage 3 | `bfcl_stage3_train_all_400_shuffled_seed42.parquet` | 400 rows, 100 per category | `fee03852fefed510e4022a7f44894518ef0af6790807e35655b0baf9979ef2d6` | Prepared seed pool; formal Stage 3 training is pending |

The local Stage 1/2 training membership matches upstream `bfcl_train_base.parquet`, and the prepared Stage 3 seed membership matches upstream `bfcl_train.parquet`. Stage 3’s later replay pool will additionally contain validated online candidates; those generated candidates are not yet released here.

RODS does not publish a separate RODS-only evaluation artifact or standalone evaluation ID manifest in its public `RODS/` directory. Its paper defines the in-distribution protocol as 800 BFCL V3 Multi-Turn samples: 400 training samples (100 per category) and the remaining 400 held-in evaluation samples (100 per category). In the public AWorld-RL processed-data layout, those 400 evaluation rows are represented by the 100-row validation file plus the 300-row test file linked above.

### Local evaluation dataset used by ToolWeave

The Stage 1 final gate and the Stage 2 final comparison both use the same canonical local `eval_400` dataset, recorded as `val_400_combined.parquet` in the workspace:

| Category | Rows |
|---|---:|
| Base | 100 |
| Missing Function | 100 |
| Missing Parameter | 100 |
| Long Context | 100 |
| **Total** | **400** |

An ID-level audit shows that this local 400-row set has exactly the union of the sample IDs in upstream `bfcl_val.parquet` and `bfcl_test.parquet`. Local rows additionally carry project-side protocol-alignment metadata, so the local parquet is not rehosted in this documentation-only repository. The separately prepared local RODS-style audit subset contains 100 rows (25 per category) sampled from this canonical `eval_400`; it is an audit subset, not the main Stage 1/2 evaluation set.

## Stage 1 and Stage 2 Training

This section is reconstructed from the selected run configurations, reward implementations, interaction code, and retained validation artifacts—not from intended settings alone. Both stages use veRL GRPO and the real stateful BFCL multi-turn interaction stack, but they optimize different environment rewards.

### Shared interaction and optimization setup

| Setting | Stage 1 | Stage 2 |
|---|---|---|
| Starting model | `Qwen/Qwen3-4B` | merged Stage 1 update 25 |
| RL training rows | 100 Base rows | the same 100 Base rows |
| Prompt groups per update | 20 | 20 |
| Rollouts per prompt, `K` | 16 | 16 |
| Trajectories per optimizer update | 320 | 320 |
| Prompt / response limit | 8,192 / 10,000 tokens | 8,192 / 10,000 tokens |
| Optimizer | AdamW through veRL, learning rate `1e-6`, warmup ratio `0.03` | same |
| PPO epochs | 1 | 1 |
| PPO mini-batch | 20 **prompt groups** | 20 **prompt groups** |
| PPO micro-batch | 2 per GPU with dynamic token batching | same |
| Parallelism | 2 GPUs, Ulysses sequence parallel size 2 | same |
| Gradient checkpointing | enabled | enabled |
| FSDP residency | actor parameters resident; optimizer and reference offloaded | same |
| GRPO group normalization | normalize by group standard deviation | same |
| PPO clipping | low `0.20`, high `0.28`, dual-clip constant `10` | same |
| Entropy coefficient | `0.001` | `0.001` |
| Selected checkpoint position | update 25 = 5 passes over 100 rows | update 25 = 5 passes over 100 rows |

`PPO mini-batch = 20` is expressed in pre-expansion prompt groups in this local veRL configuration; it is not a flattened 320-trajectory mini-batch label. The Stage 1 profile allowed an early-stopped schedule of up to 20 epochs; the selected and released checkpoint is update 25. Stage 2 was explicitly budgeted for five epochs (25 updates).

### Stage 1 — format and executable tool-use cold start

Stage 1 uses the EnvTuning interaction profile and the native format/tool reward. For trajectory `i`, let `C_i` be its ordered diagnostic-code sequence, `T_i = |C_i|`, and `n_{i,c}` the count of code `c`. Define:

$$
I_i^{\mathrm{tool}}=\mathbf{1}[n_{i,-1}+n_{i,-2}>0],
\qquad
F_i=\frac{T_i-n_{i,-3}}{T_i},
$$

$$
Q_i=
\begin{cases}
\dfrac{n_{i,-1}}{n_{i,-1}+n_{i,-2}}, & I_i^{\mathrm{tool}}=1,\\
0, & \text{otherwise},
\end{cases}
\qquad
S_i^{\mathrm{Stage1}}=I_i^{\mathrm{tool}}(F_i+Q_i).
$$

Thus the Stage 1 environment score is in `[0, 2]`: it rewards parser-compatible actions and executable tool calls, not fixed-denominator task completion. GRPO normalizes these sequence scores within each 16-rollout prompt group. Training also uses a reward-side adaptive KL controller (`initial β = 0.1`, target `0.1`, horizon `10,000`); the validation tables below report the unpenalized environment metrics.

### Stage 2 — fixed-denominator Progress-Reward RL

Stage 2 starts from the merged Stage 1 update-25 model and switches to the standard, non-augmented BFCL environment. For sample `i`, let `U_i` be the number of expected BFCL user turns, taken from `len(ground_truth)`. Only terminal code `1` is a successful turn:

$$
R_{P,i}=\frac{n_{i,1}}{U_i}.
$$

This denominator is fixed. Truncated, early-terminated, or otherwise missing terminal turns remain failures instead of disappearing from the denominator. Stage 2 GRPO uses only `R_P` as its global reward; reward-side KL is disabled. A low-variance reference-KL term with coefficient `0.01` is applied in the actor loss.

The two stages therefore answer different questions:

- Stage 1 score: “Was the trajectory formatted correctly, and were attempted tool calls executable?”
- Stage 2 `R_P`: “What fraction of all expected user turns were actually solved?”

Their raw `1.x` and `0.x` values are not directly comparable.

The implementation returns zero for any otherwise undefined empty denominator (`T_i=0`, no tool attempts, no observed terminal turns, or `U_i=0`) and raises if a trajectory produces more terminal turns than its ground-truth contract permits.

## Reward and Diagnostic Contract

The multi-turn environment emits a diagnostic event for every assistant action or closed user turn:

| Code | Exact role |
|---:|---|
| `-3` | Response-format/parser failure. The required reasoning/action serialization could not be parsed. |
| `-2` | A parsed tool call reached execution but failed in the environment/VM. |
| `-1` | A parsed tool call executed without an execution error. This is an intermediate tool event, **not** terminal task success. |
| `0` | Terminal user-turn failure. On a Missing task turn with empty GT, incorrectly calling a tool also closes the turn with failure. |
| `1` | Terminal user-turn success. Normal turns require both state and response checks; a correct no-tool answer closes an empty-GT Missing turn successfully. |

Only `0` and `1` are terminal codes. The following diagnostics are used throughout the tables:

$$
P_i^{\mathrm{observed}}=\frac{n_{i,1}}{n_{i,0}+n_{i,1}},
\qquad
\mathrm{Coverage}_i=\frac{n_{i,0}+n_{i,1}}{U_i},
$$

$$
\mathrm{MissingTurns}_i=U_i-(n_{i,0}+n_{i,1}),
\qquad
\mathrm{Incomplete}_i=\mathbf{1}[n_{i,0}+n_{i,1}<U_i].
$$

`Observed progress` is retained only as a Stage 1 diagnostic: it divides by terminal turns that happened to be observed. `R_P` uses all expected turns and is the Stage 2/3 training reward. Reported dataset metrics are sample-macro means, `N^{-1}\sum_i m_i`. `Tool-use rate` is the mean of `I_i^tool`; `rounds` is the mean of `T_i`. Action-parser rates are pooled valid actions divided by all actions in the indicated position bucket.

## Checkpoint Validation

These are deterministic, one-rollout validation passes saved during the selected training runs. They are training-time validation metrics, not official BFCL leaderboard accuracy.

### Validation dataset identity

| Run | Validation file | Rows | Composition | SHA256 |
|---|---|---:|---|---|
| Stage 1 | `val_100_stratified_seed42.parquet` | 100 | 25 Base + 25 Long Context + 25 Missing Function + 25 Missing Parameter | `479f28404af7a878a637ae71b1f83911c614b38e00c8aa42217db975f88492cc` |
| Stage 2 | `checkpoint_gate_eval/val_base_100.parquet` | 100 | 100 Base | `5f1908c1f2766a5aea38d0a3cbc32f3fe39e132adc372b907c0e49301fa5f664` |

The Stage 1 directory retains validation outputs for updates 10, 15, 20, and 25. No update-5 validation JSON was retained, so it is not reconstructed or imputed here.

### Stage 1 retained checkpoints

| Update | Epoch | Stage 1 score ↑ | Observed progress ↑ | Format ↑ | Tool-call execution ↑ | Tool-use rate ↑ | Mean rounds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 2 | 1.7122 | 0.3631 | 0.8646 | 0.8622 | 0.8700 | 10.95 |
| 15 | 3 | 1.6986 | 0.3495 | 0.8519 | 0.8629 | 0.8700 | 11.79 |
| 20 | 4 | 1.7463 | 0.3525 | 0.8719 | 0.8777 | 0.8900 | 11.11 |
| 25 | 5 | 1.7428 | 0.3489 | 0.8788 | 0.8740 | 0.8900 | 11.51 |

Because this validation split is balanced, its per-category Stage 1 score / observed-progress pairs are also available:

| Update | Base | Long Context | Missing Function | Missing Parameter |
|---:|---:|---:|---:|---:|
| 10 | 1.8046 / 0.4293 | 1.7299 / 0.3547 | 1.7235 / 0.4221 | 1.5908 / 0.2464 |
| 15 | 1.8702 / 0.3813 | 1.8785 / 0.3993 | 1.4750 / 0.2972 | 1.5706 / 0.3200 |
| 20 | 1.8163 / 0.4127 | 1.9040 / 0.3513 | 1.5721 / 0.3310 | 1.6930 / 0.3150 |
| 25 | 1.8649 / 0.4873 | 1.8092 / 0.3460 | 1.6457 / 0.2993 | 1.6512 / 0.2630 |

### Stage 2 retained checkpoints

The Stage 2 reward wrapper directly emits `R_P`, terminal coverage, incomplete-trajectory status, expected turns, missing terminals, and code counts. Metrics marked `†` below are deterministic post-derivations from each row’s saved diagnostic counts using the Stage 1 formulas; they did not alter training reward.

| Update | `R_P` ↑ | Terminal coverage ↑ | Incomplete rate ↓ | Missing turns ↓ | Format† ↑ | Tool execution† ↑ | Tool-use† ↑ | Rounds† |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.4463 | 0.9195 | 0.1100 | 0.25 | 0.8801 | 0.9364 | 0.9500 | 11.66 |
| 10 | 0.4668 | 0.8903 | 0.1200 | 0.35 | 0.8364 | 0.8910 | 0.9100 | 10.40 |
| 15 | 0.5267 | 0.8900 | 0.1200 | 0.33 | 0.8319 | 0.8850 | 0.9000 | 10.06 |
| 20 | 0.5590 | 0.9700 | 0.0300 | 0.10 | 0.8718 | 0.9466 | 0.9600 | 10.90 |
| 25 | 0.5565 | 0.9200 | 0.0800 | 0.27 | 0.8629 | 0.9018 | 0.9100 | 10.03 |

The mean expected-turn count is `3.68` for every row above. Mean diagnostic-event counts per trajectory are:

| Update | `n(-3)` | `n(-2)` | `n(-1)` | `n(0)` | `n(1)` |
|---:|---:|---:|---:|---:|---:|
| 5 | 1.04 | 0.08 | 7.11 | 1.70 | 1.73 |
| 10 | 0.96 | 0.10 | 6.01 | 1.56 | 1.77 |
| 15 | 0.97 | 0.08 | 5.66 | 1.34 | 2.01 |
| 20 | 1.32 | 0.10 | 5.90 | 1.48 | 2.10 |
| 25 | 0.79 | 0.04 | 5.79 | 1.26 | 2.15 |

Stage 2 update 20 has the highest retained Base-100 `R_P`; update 25 is the terminal selected checkpoint and the model used for the downstream `eval_400` and Stage 3 initialization. Intermediate validation JSON does not imply that every intermediate weight checkpoint was retained.

## Selected Checkpoints on Eval-400

The selected Stage 1 and Stage 2 update-25 models were evaluated on the same canonical 400-row set (`100 ×` Base, Long Context, Missing Function, and Missing Parameter):

```text
val_400_combined.parquet
SHA256: ee1eba107d6e8ae0602abb3eb8db566d965515ea0da10fdd9eff264ad481423f
```

Each run uses deterministic `n=1` decoding and the real multi-turn interaction environment. These tables report internal reward/protocol metrics—not the BFCL leaderboard’s AST/functional accuracy percentage.

### Stage 1 update 25 under its native reward protocol

| Split | Rows | Stage 1 score ↑ | Format ↑ | Tool execution ↑ | Tool-use rate ↑ | Observed progress ↑ | Final-answer closure† ↑ | Rounds | Parser ≥2 actions ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 100 | 1.7601 | 0.8633 | 0.9047 | 0.9200 | 0.4368 | 0.7300 | 11.61 | 0.9231 |
| Long Context | 100 | 1.7292 | 0.8511 | 0.8798 | 0.9000 | 0.4177 | 0.5400 | 10.63 | 0.9158 |
| Missing Function | 100 | 1.7584 | 0.8627 | 0.9057 | 0.9200 | 0.3889 | 0.7400 | 13.07 | 0.9140 |
| Missing Parameter | 100 | 1.5550 | 0.7646 | 0.7972 | 0.8100 | 0.3499 | 0.7300 | 10.36 | 0.9111 |
| **Overall** | **400** | **1.7007** | **0.8354** | **0.8718** | **0.8875** | **0.3983** | **0.6850** | **11.42** | **0.9161** |

`†` The legacy Stage 1 artifact named this field `terminal_coverage`, but its evaluator computes whether a transcript ends in a valid non-tool answer. It is documented here as **final-answer closure**, not as the fixed expected-turn coverage defined above.

The pooled action-parser success rates for Stage 1 update 25 are `0.9841` at action 1, `0.8939` at action 2, and `0.9182` at action 3 or later. With 2,000 sample-level bootstrap resamples (seed 42), overall 95% intervals are: score `[1.6398, 1.7611]`, format `[0.8042, 0.8650]`, tool execution `[0.8417, 0.9027]`, tool-use rate `[0.8575, 0.9175]`, and observed progress `[0.3628, 0.4330]`. Truncation and early-termination rates are both `0`. A legacy transcript-final approximation classifies 35 rows as `-3`, 91 as `-1`, and 274 as `1`; these are **not** raw per-event code counts and are not used as reward.

### Direct update-25 comparison under the Stage 2 fixed-denominator protocol

To compare actual task progress on one scale, the Stage 1 update-25 model was also re-evaluated with the exact Stage 2 fixed-denominator reward wrapper:

| Model | Overall `R_P` ↑ | Base ↑ | Long Context ↑ | Missing Function ↑ | Missing Parameter ↑ |
|---|---:|---:|---:|---:|---:|
| Stage 1 update 25 | 0.3728 | 0.4457 | 0.3429 | 0.3725 | 0.3302 |
| **Stage 2 update 25** | **0.4567** | **0.6027** | **0.3952** | **0.4515** | **0.3774** |

The paired overall improvement is `+0.0839`, with a paired 95% bootstrap interval of `[+0.0536, +0.1146]` over the same 400 sample IDs.

### Stage 2 update 25 detailed eval-400 diagnostics

| Split | `R_P` ↑ | Coverage ↑ | Incomplete ↓ | Format† ↑ | Tool execution† ↑ | Tool-use† ↑ | Rounds | Parser ≥2 actions ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 0.6027 | 0.9575 | 0.0500 | 0.8960 | 0.9547 | 0.9600 | 10.43 | 0.9374 |
| Long Context | 0.3952 | 0.7730 | 0.2600 | 0.8577 | 0.9170 | 0.9300 | 8.62 | 0.8737 |
| Missing Function | 0.4515 | 0.8868 | 0.1300 | 0.8582 | 0.9195 | 0.9300 | 10.52 | 0.9127 |
| Missing Parameter | 0.3774 | 0.8781 | 0.1400 | 0.8207 | 0.8786 | 0.8900 | 10.41 | 0.9279 |
| **Overall** | **0.4567** | **0.8739** | **0.1450** | **0.8582** | **0.9174** | **0.9275** | **10.00** | **0.9149** |

`†` As in the checkpoint table, format/tool metrics are post-derived diagnostics. They do not enter `R_P`. Complete mean diagnostic-event counts are:

| Split | Expected turns | `n(-3)` | `n(-2)` | `n(-1)` | `n(0)` | `n(1)` |
|---|---:|---:|---:|---:|---:|---:|
| Base | 3.68 | 0.86 | 0.02 | 6.02 | 1.24 | 2.29 |
| Long Context | 3.68 | 1.04 | 0.07 | 4.72 | 1.35 | 1.44 |
| Missing Function | 4.68 | 1.00 | 0.06 | 5.23 | 2.01 | 2.22 |
| Missing Parameter | 4.68 | 1.14 | 0.08 | 5.05 | 2.30 | 1.84 |
| **Overall** | **4.18** | **1.0100** | **0.0575** | **5.2550** | **1.7250** | **1.9475** |

The Stage 2 overall pooled action-parser rates are `0.9816` at action 1, `0.9030` at action 2, `0.9163` at action 3 or later, and `0.9149` over all actions at position 2 or later.

## Stage 3 Training Branch

The current Stage 3 branch has been implemented and deterministic/smoke-tested locally, but **formal five-epoch Stage 3 training has not been launched**. The description below follows the final audited code path.

### Global branch: Stage 2 reward remains unchanged

For each prompt, `K=16` rollouts retain the fixed-denominator `R_P`. Standard GRPO first computes the group-normalized global advantage:

$$
A_{\mathrm{RODS},i}=\frac{R_{P,i}-\overline{R_P}}{s(R_P)+10^{-6}}.
$$

MatchTIR-derived quantities never enter `R_P`, global GRPO normalization, or boundary classification. Disabling local credit or setting its weight to zero returns the original global advantage tensor exactly.

### Tool-local branch: one match per BFCL user turn

Within one `(prompt, rollout, BFCL user turn)` scope, every individual predicted call from every tool policy step is flattened. For predicted call `p` and GT call `g`, different function names force similarity zero. Otherwise:

$$
S(p,g)=\frac{1+J_{\mathrm{multiset}}(\mathrm{argNames}_p,\mathrm{argNames}_g)
+\sum_{k\in\mathrm{args}_g}\mathbf{1}[p_k=g_k]}
{2+|\mathrm{args}_g|}.
$$

Exactly one maximum-weight Hungarian assignment is run over the entire user turn. Unmatched calls receive `0`. Calls are then mapped back to their original policy steps:

$$
r_s=\frac{1}{|C_s|}\sum_{c\in C_s}r_c,
\qquad
G_s=r_s+0.9G_{s+1}.
$$

Discounting is confined to the same BFCL user turn and occurs over policy steps, never individual call indices. At each `(prompt, user turn, policy-step depth)`, returns are normalized only across rollouts that actually contain that depth:

$$
A_{\mathrm{local},s}=\frac{G_s-\mu_{\mathcal S_s}}{\sigma_{\mathcal S_s}+10^{-6}}.
$$

No absent rollout is padded with a fake zero. Empty-GT Missing turns, support smaller than two, and zero-variance groups receive local advantage zero. The local value is assigned only to the exact actor tool-call token span; environment/tool-observation tokens remain masked out.

The final actor advantage is:

$$
\boxed{A_{\mathrm{new}}=A_{\mathrm{RODS}}+1.0\,A_{\mathrm{local}}}.
$$

This is a ToolWeave BFCL adaptation of MatchTIR-style local credit. It is not part of the original RODS global reward.

### PPO/GRPO and KL path

The actor uses the stored rollout policy log-probability as `old_log_prob` and computes:

$$
\rho_t=\exp\!\left(\operatorname{clip}(\log\pi_t-\log\pi_{\mathrm{old},t},-20,20)\right).
$$

The existing veRL sequence-mean/token-mean PPO loss, asymmetric clip range (`0.20/0.28`), dual-clip constant `10`, entropy coefficient `0.001`, and reference low-variance KL loss coefficient `0.01` are preserved. Stage 3 does not add local reward to the reference KL or optimizer state.

### Boundary selection and dynamic lifecycle

After an optimizer step, lifecycle selection groups **only `R_P`** by prompt:

$$
\bar r_P=\frac{1}{K}\sum_{k=1}^{K}R_{P,k},
\qquad
\phi=4\bar r_P(1-\bar r_P).
$$

| Region | Rule |
|---|---|
| Too hard | `mean R_P < 0.20` |
| Boundary | `0.20 ≤ mean R_P ≤ 0.85` |
| Mastered | `mean R_P > 0.85` |

Boundary candidates pass a sample-identity cooldown, are partitioned into the four BFCL types, ranked by descending `φ` within type, clipped by each type quota `M_τ`, and finally bounded by total `M`. There is no automatic quota redistribution when a type has too few candidates.

RODS specifies the `M`, `M_τ`, and cooldown mechanisms but does not publish one unique numeric default. The formal Stage 3 configuration therefore leaves them `null`, sets `require_seed_selection_config: true`, and fails closed rather than silently emitting zero seeds. The completed online smoke profile used project choices `M=16`, `M_τ=4/4/4/4`, and cooldown `c=13`; these are **not claimed as paper defaults**.

Validated candidates generated in epoch `n` are staged and become eligible only from epoch `n+1`. At an epoch boundary:

$$
N_{\mathrm{new}}\le
\left\lfloor0.20\,|D_{\mathrm{active,before}}|\right\rfloor,
$$

while the generated sub-pool is capped at 400. The original 400 BFCL seeds are protected from retirement. Generated rows receive a one-observation trial; rows below `0.20` can be evicted as too hard, rows above `0.95` can retire as mastered, and stale retirement remains disabled because no reproducible paper-default window is available.

### Boundary-to-generator contract

The Training Branch stops at a durable selected-seed queue. A separate Data-Generation Branch consumes that contract and performs:

```text
selected boundary seed
  → feedback-conditioned planner
  → function/parameter proposal
  → real BFCL VM execution (GT first)
  → per-class query generation and verification
  → whole-conversation rewrite
  → Missing Function / Missing Parameter transform where required
  → deterministic semantic and fresh-VM gates
  → Quality Judge (at most one refinement cycle)
  → validated candidate queue
  → rate-limited next-epoch admission
```

One shared Gemma-4-31B vLLM service is the project’s synthesis-backbone substitution; RODS reports Qwen3-32B. The active audited catalog contains 128 functions. Public sources do not expose a deterministic HIGH_LEVEL-to-BOTTOM_LEVEL decomposition map, so unsupported HIGH_LEVEL plans fail closed rather than allowing the LLM to invent executable GT. Project semantic guards, exact-content deduplication, terminal journal ordering, file locking, and crash reconciliation protect candidate precision and durability without changing the Training Branch reward mathematics.

### Audited implementation map

The first repository release remains documentation-only, but this README was checked against the following workspace sources; these paths identify the implementation intended for the later code release.

| Responsibility | Audited source |
|---|---|
| Stage 1 run profile | Audited Stage 1 YAML training profile from the workspace |
| Stage 1 reward | `env_tuning/format_reward.py::compute_score` |
| Stage 2 run profile | `stage1_format_rl/configs/stage2_qwen3_4b_k16_base_progress_batch20_plain_env.yaml` |
| Fixed-denominator Progress Reward | `stage1_format_rl/rewards/rods_stage2_progress_reward.py::compute_score` |
| Stage 3 run profile | `stage1_format_rl/configs/stage3_rods_matchtir_v1_training_branch.yaml` |
| Global GRPO then residual fusion | `verl/verl/trainer/ppo/ray_trainer.py::compute_advantage` |
| Local similarity and Hungarian assignment | `env_tuning/rods_matchtir_v1/matching.py` |
| Step rewards, returns, ragged normalization, and fusion | `env_tuning/rods_matchtir_v1/advantage.py` |
| Boundary selection and dynamic replay | `env_tuning/rods_matchtir_v1/lifecycle.py` |
| Verified synthesis branch | `env_tuning/rods_data_generation_v1/` |

## Why ToolWeave?

### 🎯 Learn at the Boundary

Online synthesis focuses on tasks where the current policy is neither always failing nor already mastered.

### 🛠 Execute Before You Train

Synthetic ground truth must execute in the real BFCL environment before entering the replay pool.

### 🧵 Credit the Tool Call

Global task progress is complemented with fine-grained local credit for relevant tool actions.

### 🔒 Validate Before Replay

Only candidates passing semantic, execution, coherence, and fresh-VM checks are admitted.

## Open Resources

| Resource | Status | Link |
|---|---|---|
| ToolWeave documentation and branding | Repository initialized | [this repository](https://github.com/Muradil-mamat-211/ToolWeave) |
| Stage 1 model | Public checkpoint repository; release documentation pending | [Hugging Face](https://huggingface.co/muradil211/stage1) |
| Stage 2 model | Public checkpoint repository; release documentation pending | [Hugging Face](https://huggingface.co/muradil211/stage2) |
| Stage 3 reference checkpoint | Public RODS checkpoint; ToolWeave final release pending | [Hugging Face](https://huggingface.co/muradil211/stage3) |
| Stage 1/2 training data | Upstream processed training split | [`bfcl_train_base.parquet`](https://github.com/inclusionAI/AWorld-RL/blob/main/EnvTuning/data/bfcl_train_base.parquet) |
| Stage 3 seed training data | Upstream processed training split; formal ToolWeave Stage 3 pending | [`bfcl_train.parquet`](https://github.com/inclusionAI/AWorld-RL/blob/main/EnvTuning/data/bfcl_train.parquet) |
| EnvTuning data and environment | Upstream | [AWorld-RL / EnvTuning](https://github.com/inclusionAI/AWorld-RL/tree/main/EnvTuning) |
| RODS / BFCL V3 held-in evaluation split | Upstream processed files; not rehosted here | [`bfcl_val.parquet`](https://github.com/inclusionAI/AWorld-RL/blob/main/EnvTuning/data/bfcl_val.parquet) + [`bfcl_test.parquet`](https://github.com/inclusionAI/AWorld-RL/blob/main/EnvTuning/data/bfcl_test.parquet) |
| RODS resources | Upstream | [AWorld-RL / RODS](https://github.com/inclusionAI/AWorld-RL/tree/main/RODS), [paper](https://arxiv.org/abs/2606.19047) |

## Quick Start

This first public release is intentionally documentation-only. It contains the project identity, method overview, provenance boundaries, and release status; it does not yet contain training or data-generation source code.

To orient yourself:

1. Read the [Overview](#overview), [Stage 1 and Stage 2 Training](#stage-1-and-stage-2-training), and [Stage 3 Training Branch](#stage-3-training-branch).
2. Review the upstream [EnvTuning](https://github.com/inclusionAI/AWorld-RL/tree/main/EnvTuning) and [RODS](https://github.com/inclusionAI/AWorld-RL/tree/main/RODS) implementations.
3. Use the model links in [Models](#models) with the documented status: Stage 1/2 are public checkpoints, while Stage 3 currently points to a public RODS reference checkpoint rather than a completed ToolWeave final model.

Training commands, data-generation commands, and a reproducibility package will be added in a later release after the public code boundary is reviewed.

## Repository Layout

```text
ToolWeave/
├── README.md
└── assets/
    ├── toolweave-mark.svg
    └── toolweave-pipeline.svg
```

Training, data-generation, evaluation, and reproducibility code will be added in the next release step.

## Roadmap

- [x] Stage 1 training
- [x] Stage 2 training
- [x] Stage 3 data-generation pipeline validation
- [x] Generator precision hardening
- [ ] Final Stage 3 formal training
- [ ] Stage 3 BFCL evaluation
- [x] Stage 1/2 model release
- [ ] ToolWeave final model release
- [ ] Full training code release
- [ ] Reproducibility package

## Acknowledgements

ToolWeave builds on and adapts ideas or public infrastructure from the following projects. Their authors and teams are not implied to be contributors to ToolWeave.

- [AWorld-RL](https://github.com/inclusionAI/AWorld-RL)
- [Environment Tuning](https://github.com/inclusionAI/AWorld-RL/tree/main/EnvTuning) and its [paper](https://arxiv.org/abs/2510.10197)
- [RODS](https://github.com/inclusionAI/AWorld-RL/tree/main/RODS) and its [paper](https://arxiv.org/abs/2606.19047)
- [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
- [veRL](https://github.com/volcengine/verl)
- [Qwen](https://huggingface.co/Qwen/Qwen3-4B)
- [MatchTIR](https://github.com/quchangle1/MatchTIR) and its [paper](https://arxiv.org/abs/2601.10712)

## Citation

A formal ToolWeave citation will be added with the public technical report.

For upstream work used as reference, please cite the original sources:

### RODS

```bibtex
@article{fang2026rods,
  title={RODS: Reward-Driven Online Data Synthesis for Multi-Turn Tool-Use Agents},
  author={Fang, Ruishan and Lu, Siyuan and Zhuang, Chenyi and Lin, Tao},
  journal={arXiv preprint arXiv:2606.19047},
  year={2026}
}
```

### Environment Tuning

```bibtex
@article{lu2025don,
  title={Don't Just Fine-tune the Agent, Tune the Environment},
  author={Lu, Siyuan and Wang, Zechuan and Zhang, Hongxuan and Wu, Qintong and Gan, Leilei and Zhuang, Chenyi and Gu, Jinjie and Lin, Tao},
  journal={arXiv preprint arXiv:2510.10197},
  year={2025}
}
```
