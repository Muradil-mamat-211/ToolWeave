<div align="center">

<img src="assets/toolweave-mark.svg" alt="ToolWeave mark" width="130">

# ToolWeave

**🧵 Boundary-Guided Verified Tool-Use Synthesis and Agentic Reinforcement Learning for Multi-Turn Tool-Calling Agents.**

ToolWeave trains multi-turn tool-use agents through a staged curriculum, then closes the loop between policy learning and verified online data synthesis by detecting capability-boundary tasks and generating new executable tool-use trajectories.

[![Code](https://img.shields.io/badge/GitHub-Code-181717?logo=github&logoColor=white)](https://github.com/Muradil-mamat-211/ToolWeave)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Agentic RL](https://img.shields.io/badge/Agentic-RL-6D28D9)](#toolweave-method)
[![Tool Calling](https://img.shields.io/badge/Tool-Calling-0EA5E9)](#overview)
[![BFCL](https://img.shields.io/badge/BFCL-Multi--Turn-F59E0B)](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
[![Qwen3-4B](https://img.shields.io/badge/Qwen3--4B-7C3AED)](https://huggingface.co/Qwen/Qwen3-4B)

</div>

> [!IMPORTANT]
> ToolWeave is a project-level framework. It is **not** the official RODS implementation. The project distinguishes upstream EnvTuning/RODS concepts, reused public BFCL/EnvTuning infrastructure, and ToolWeave-specific extensions and robustness layers.

## Table of Contents

- [Overview](#overview)
- [Training Pipeline](#training-pipeline)
- [Models](#models)
- [Data](#data)
- [ToolWeave Method](#toolweave-method)
- [Why ToolWeave?](#why-toolweave)
- [Evaluation](#evaluation)
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
<img src="assets/toolweave-pipeline.svg" alt="ToolWeave staged training and closed-loop replay pipeline" width="100%">
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

| Stage | Name | Objective | Starting Model | Output |
|---|---|---|---|---|
| Stage 1 | Tool-Use Cold Start | Learn stable response format and reliable tool interaction | Qwen3-4B | ToolWeave Stage 1, update 25 |
| Stage 2 | Progress-Reward RL | Learn multi-turn task completion in the standard environment | ToolWeave Stage 1, update 25 | ToolWeave Stage 2, update 25 |
| Stage 3 | Boundary-Guided Online RL | Expand capability with verified online synthesis and dynamic replay | ToolWeave Stage 2, update 25 | Formal training and release pending |

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
| BFCL multi-turn data | [BFCL data](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard/bfcl_eval/data) via [AWorld-RL](https://github.com/inclusionAI/AWorld-RL) | Stage 1/2 environment tasks and evaluation protocol |
| EnvTuning infrastructure | [AWorld-RL / EnvTuning](https://github.com/inclusionAI/AWorld-RL/tree/main/EnvTuning) | Public multi-turn environment-tuning reference and infrastructure |
| RODS resources and benchmark setup | [AWorld-RL / RODS](https://github.com/inclusionAI/AWorld-RL/tree/main/RODS) | Stage 3 boundary detection, synthesis, and replay reference |
| Generated ToolWeave candidates | ToolWeave project | Online validated Stage 3 replay data; **release pending** |

## ToolWeave Method

### 1. Stage 1 — Tool-Use Cold Start

Stage 1 uses EnvTuning-style environment interaction to establish stable tool-call formatting, parser compatibility, and basic multi-turn execution behavior. Its purpose is to reduce the cold-start burden before Progress-Reward learning. This stage reuses and adapts public infrastructure; it is not presented as a new EnvTuning algorithm.

### 2. Stage 2 — Progress-Reward Learning

Stage 2 trains on the real multi-turn BFCL environment with the standard non-augmented Base environment and a fixed-denominator Progress Reward. The objective is to move beyond formatting competence toward actual task progress and terminal task completion:

```text
Formatting / tool competence → actual task progress
```

### 3. Stage 3 — Capability-Boundary Detection

For each prompt, the intended Stage 3 lifecycle obtains `K` rollouts and uses Progress Reward to estimate the current capability region:

```text
hard        → usually unsuccessful
boundary    → mixed success and failure; informative for synthesis
mastered    → usually successful
```

Only capability-boundary tasks trigger online synthesis. This is a RODS-style use of reward information, while the lifecycle integration and project policy are ToolWeave-specific.

### 4. Verified Online Data Synthesis

ToolWeave’s project-specific synthesis and validation path is organized as:

```text
Boundary Seed
  → Planner
  → Function Sampling
  → Parameter Generation
  → Real BFCL VM Execution
  → Query Generation
  → Verification
  → Coherence Rewrite
  → Missing-Function / Missing-Parameter Transformation
  → Semantic Guards
  → Fresh VM Replay
  → Quality Judge
  → Validated Candidate
```

The current correctness contract includes:

- argument provenance;
- execution-result semantics;
- unit semantics;
- relational ambiguity checks;
- genuine Missing Parameter validation;
- global coherence;
- observation-grounded consistency;
- action minimality;
- fresh-VM verification; and
- exact-content deduplication.

These deterministic and project-level semantic guards are **ToolWeave-specific robustness layers**. They are not claimed to be part of the official original RODS implementation.

### 5. Dynamic Replay

Generated candidates follow an epoch boundary:

```text
candidate generated in epoch n
  → validated
  → staged
  → eligible from epoch n+1
```

This prevents same-epoch leakage and keeps the policy from training on a candidate before the generation step has completed its validation contract.

### 6. Policy Optimization

The intended Stage 3 policy branch combines task-level and tool-call-level signals:

```math
A_{\text{ToolWeave}}
= A_{\text{global}}
+ \lambda_{\text{local}} A_{\text{tool-local}}.
```

- The global branch uses task-level Progress Reward in a RODS-style trajectory advantage.
- The local branch uses MatchTIR-inspired matching between predicted and ground-truth tool calls.
- The local residual applies only to relevant tool-call action tokens.

This global-plus-local construction is a **ToolWeave-specific adaptation**. It should not be interpreted as the official RODS algorithm or as a claim that RODS includes the ToolWeave project guards.

## Why ToolWeave?

### 🎯 Learn at the Boundary

Online synthesis focuses on tasks where the current policy is neither always failing nor already mastered.

### 🛠 Execute Before You Train

Synthetic ground truth must execute in the real BFCL environment before entering the replay pool.

### 🧵 Credit the Tool Call

Global task progress is complemented with fine-grained local credit for relevant tool actions.

### 🔒 Validate Before Replay

Only candidates passing semantic, execution, coherence, and fresh-VM checks are admitted.

## Evaluation

ToolWeave does not claim SOTA. Stage 3 final evaluation is pending, and no Stage 3 final model is presented here.

### Upstream RODS reference

The following numbers are reproduced from the public [RODS report](https://github.com/inclusionAI/AWorld-RL/tree/main/RODS) for context only. They are **not ToolWeave measurements**.

| Model | Overall | Base | Miss Func | Miss Param | Long Context |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B-Instruct | 22.13 | 26.50 | 21.00 | 15.50 | 25.50 |
| + EnvTuning | 50.50 | 64.00 | 52.00 | 35.00 | 51.00 |
| + RODS | 56.00 | 68.00 | 59.00 | 44.00 | 53.00 |
| ToolWeave Stage 3 | **Evaluation pending** | — | — | — | — |

### Internal Stage-wise Evaluation

The following recorded workspace measurements document Stage 1/2 development; detailed artifacts are not included in this documentation-only repository release.

| Checkpoint / comparison | Protocol | Result |
|---|---|---:|
| Stage 1 update 20 | Base-100 gate score, deterministic `n=1` | 1.7689 |
| Stage 1 update 25 | Base-100 gate score, deterministic `n=1` | 1.7601 |
| Stage 1 update 20 | `val-400` overall score | 1.7077 |
| Stage 1 update 25 | `val-400` overall score | 1.7007 |
| Stage 2 update 25 | `eval_400` overall Progress Reward | 0.4567 |
| Stage 2 update 25 | `eval_400` Base-100 Progress Reward | 0.6027 |

The Stage 1 step20/step25 rows are a direct comparison under the same gate protocol. Stage 1 score and Stage 2 Progress Reward use different reward contracts and must not be compared numerically as one benchmark. Stage 2 update 20 did not have a complete retained weight checkpoint for a full `eval_400` rerun.

## Open Resources

| Resource | Status | Link |
|---|---|---|
| ToolWeave documentation and branding | Repository initialized | [this repository](https://github.com/Muradil-mamat-211/ToolWeave) |
| Stage 1 model | Public checkpoint repository; release documentation pending | [Hugging Face](https://huggingface.co/muradil211/stage1) |
| Stage 2 model | Public checkpoint repository; release documentation pending | [Hugging Face](https://huggingface.co/muradil211/stage2) |
| Stage 3 reference checkpoint | Public RODS checkpoint; ToolWeave final release pending | [Hugging Face](https://huggingface.co/muradil211/stage3) |
| EnvTuning data and environment | Upstream | [AWorld-RL / EnvTuning](https://github.com/inclusionAI/AWorld-RL/tree/main/EnvTuning) |
| RODS resources | Upstream | [AWorld-RL / RODS](https://github.com/inclusionAI/AWorld-RL/tree/main/RODS), [paper](https://arxiv.org/abs/2606.19047) |

## Quick Start

This first public release is intentionally documentation-only. It contains the project identity, method overview, provenance boundaries, and release status; it does not yet contain training or data-generation source code.

To orient yourself:

1. Read the [Overview](#overview) and [ToolWeave Method](#toolweave-method).
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
