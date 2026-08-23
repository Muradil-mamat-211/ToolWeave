<div align="center">

<img src="assets/toolweave-mark.svg" alt="ToolWeave mark" width="130">

# ToolWeave

**🧵 Boundary-Guided Verified Tool-Use Synthesis and Agentic Reinforcement Learning for Multi-Turn Tool-Calling Agents.**

ToolWeave trains a multi-turn tool-use policy through a three-stage curriculum, then couples policy optimization with asynchronous, execution-verified data evolution around the current policy's capability boundary.

[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Stage1%20%7C%20Stage2%20%7C%20Stage3-yellow)](#models)
[![Training Data](https://img.shields.io/badge/%F0%9F%A4%97%20Training%20Data-RODS_EnvTuning-yellow)](https://github.com/inclusionAI/AWorld-RL/tree/main/EnvTuning/data)
[![Eval Data](https://img.shields.io/badge/%F0%9F%A4%97%20Eval%20Data-RODS_BFCL_V3-yellow)](https://github.com/inclusionAI/AWorld-RL/tree/main/EnvTuning/data)
[![Code](https://img.shields.io/badge/GitHub-Code-181717?logo=github&logoColor=white)](https://github.com/Muradil-mamat-211/ToolWeave)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Agentic RL](https://img.shields.io/badge/Agentic-RL-6D28D9)](#stage-3-toolweave)
[![Tool Calling](https://img.shields.io/badge/Tool-Calling-0EA5E9)](#overview)
[![BFCL](https://img.shields.io/badge/BFCL-Multi--Turn-F59E0B)](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)

🤗 [ToolWeave Stage 1 Model](https://huggingface.co/muradil211/ToolWeave_stage1) |
🤗 [ToolWeave Stage 2 Model](https://huggingface.co/muradil211/ToolWeave_stage2) |
🤗 [ToolWeave Stage 3 Model](https://huggingface.co/muradil211/ToolWeave_stage3)

</div>

> [!IMPORTANT]
> ToolWeave is a project-level framework. It is **not** the official RODS or MatchTIR implementation. This README distinguishes public EnvTuning/RODS concepts, MatchTIR-derived local credit, reused BFCL infrastructure, and ToolWeave-specific adaptations and semantic hardening.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Method](#method)
  - [1. Problem Formulation](#1-problem-formulation)
  - [2. Three-Stage Curriculum](#2-three-stage-curriculum)
- [Stage 3: ToolWeave](#stage-3-toolweave)
  - [3.1 Reward Modeling](#31-reward-modeling)
  - [3.2 Dual-Level Advantage Estimation](#32-dual-level-advantage-estimation)
  - [3.3 Policy Optimization](#33-policy-optimization)
    - [Real Rollout Case Study](#real-rollout-case-study-interaction-level-credit-assignment)
      - [Why Dual-Level Credit Assignment](#why-dual-level-credit-assignment)
      - [Full K=16 User Turn 3 Interaction Audit](#full-k16-user-turn-3-interaction-audit)
      - [Four Credit-Assignment Regimes](#four-credit-assignment-regimes-in-one-real-k16-group)
  - [3.4 Boundary-Guided Online Data Evolution](#34-boundary-guided-online-data-evolution)
- [Verified Online Data Synthesis](#verified-online-data-synthesis)
- [Experiments and Results](#experiments-and-results)
- [Models](#models)
- [Data](#data)
- [Repository Layout](#repository-layout)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Overview

Multi-turn tool use has two coupled difficulties. A policy must learn *whether and how* to call tools over a stateful interaction, while the training distribution must continue to expose tasks near the policy's evolving capability boundary. ToolWeave addresses both through a staged curriculum:

1. establish parser-compatible, executable tool use;
2. optimize fixed-denominator multi-turn progress; and
3. combine trajectory-level progress with interaction-level local credit derived from call-level matching, while asynchronously synthesizing verified future training data.

<div align="center">
<img src="assets/toolweave-pipeline.svg" alt="Overall ToolWeave three-stage curriculum with parallel policy-learning and asynchronous data-evolution lanes" width="100%">
</div>

Stage 3 deliberately has two non-blocking branches after the same rollout-derived Progress Reward. The policy branch performs the current optimizer update. The data branch selects boundary seeds, synthesizes and validates candidates asynchronously, and stages them for a later epoch. A candidate generated from epoch `n` is never fed back into the current update and is first eligible at epoch `n+1`.

### Method provenance at a glance

| Source | Role in ToolWeave |
|---|---|
| [EnvTuning](https://arxiv.org/abs/2510.10197) | Multi-turn interaction protocol, public BFCL environment path, and Stage 1 diagnostic semantics |
| [RODS](https://arxiv.org/abs/2606.19047) | Progress Reward, boundary-driven online synthesis, and dynamic-replay concepts |
| [MatchTIR](https://arxiv.org/abs/2601.10712) | Tool-call similarity, one-to-one local matching, and dual-level credit inspiration |
| ToolWeave | BFCL user-turn-local return, ragged same-runtime-depth normalization, additive residual fusion, deterministic semantic hardening, and durable online lifecycle safeguards |

**Formal-training provenance.** The released Stage 3 checkpoint, the frozen `runtime_interaction_final` implementation selected by the [formal reference profile](stage1_format_rl/configs/layers/profiles/stage3_reference.yaml), and the real K=16 trajectory evidence belong to the confirmed ToolWeave Stage 3 formal-training provenance chain. The current source reproduces the published interaction-credit evidence deterministically; the released model is the final formal Stage 3 checkpoint.

## Getting Started

The public repository separates portable source/configuration from machine-local models, datasets, outputs, and credentials. Use Python 3.10+ with the project runtime dependencies already installed; the repository does not currently provide a one-command environment bootstrap.

1. Clone the repository and create an ignored local environment file:

   ```bash
   git clone https://github.com/Muradil-mamat-211/ToolWeave.git
   cd ToolWeave
   cp environment/env.template.sh environment/env.local.sh
   ```

2. Edit `environment/env.local.sh` so `TOOLWEAVE_ASSET_ROOT`, `TOOLWEAVE_DATA_ROOT`, and `TOOLWEAVE_PYTHON` point to your local assets and compatible Python environment, then load it:

   ```bash
   source environment/env.local.sh
   ```

   The exact required model/data files and hashes are declared in the [Stage 3 asset layer](stage1_format_rl/configs/layers/assets/stage3_reference.yaml). Upstream datasets and released model links are listed in [Data](#data) and [Models](#models).

3. Resolve and validate the formal Stage 3 reference profile:

   ```bash
   python -m stage1_format_rl.infrastructure.cli \
     --profile stage1_format_rl/configs/layers/profiles/stage3_reference.yaml \
     resolve

   python -m stage1_format_rl.infrastructure.cli \
     --profile stage1_format_rl/configs/layers/profiles/stage3_reference.yaml \
     preflight --check-assets
   ```

4. Inspect the generated launch without starting training:

   ```bash
   python -m stage1_format_rl.infrastructure.cli \
     --profile stage1_format_rl/configs/layers/profiles/stage3_reference.yaml \
     launch
   ```

   `launch` is a dry run by default. Execution is intentionally explicit and guard-protected:

   ```bash
   ALLOW_RODS_MATCHTIR_STAGE3_TRAINING=1 \
   python -m stage1_format_rl.infrastructure.cli \
     --profile stage1_format_rl/configs/layers/profiles/stage3_reference.yaml \
     launch --execute
   ```

The reference profile targets the audited 2×96 GiB topology. Hardware, runtime placement, assets, algorithm settings, and qualification requirements remain separate layers; see the [layered configuration guide](stage1_format_rl/configs/layers/README.md) and [infrastructure documentation](docs/infrastructure-decoupling.md).

## Method

### 1. Problem Formulation

Let `q` denote one BFCL prompt. The current policy produces a group of `K` multi-turn rollouts,

$$
\mathcal{T}_q=\lbrace\tau_i\rbrace_{i=1}^{K}.
$$

Each rollout contains several distinct levels. Keeping them separate is essential to the method:

| Level | Symbol | Meaning |
|---|---|---|
| Prompt | $q$ | One BFCL sample and its initial environment/tool contract |
| Rollout | $\tau_i$ | One sampled stateful interaction for prompt $q$ |
| BFCL user turn | $u$ | One expected user task inside the multi-turn sample |
| Non-answer runtime interaction | $j\in\mathcal{D}_{i,u}$ | One assistant generation followed by parser/environment handling, before valid answer/turn closure |
| Parsed calls | $P_{i,u,j}$ | Successfully parsed structured calls in runtime interaction $j$; empty when no structured calls are parsed |
| User-turn call sequence | $P_{i,u}$ | All successfully parsed calls across the non-answer runtime interactions of user turn $u$ |
| Ground-truth calls | $G_u$ | The expected executable calls for user turn $u$ |
| Observation | $o_{i,u,j+1}$ | The parser/environment feedback produced after runtime interaction $j$ |

The formal local sequence and its domain are

$$
\mathcal{I}_{i,u}=(I_{i,u,j})_{j\in\mathcal{D}_{i,u}},
\qquad
\mathcal{D}_{i,u}=\mathrm{dom}(\mathcal{I}_{i,u})=\lbrace 0,\ldots,J_{i,u}-1\rbrace.
$$

The complete predicted-call sequence for a user turn is the duplicate-preserving concatenation

$$
P_{i,u}=\biguplus_{j\in\mathcal{D}_{i,u}}P_{i,u,j}.
$$

A valid final answer/turn closure is excluded from $\mathcal{D}_{i,u}$ and receives global credit only. Every temporally reliable non-answer runtime interaction remains in the local sequence, including parser-rejected or unclassified malformed actions. The historical `tool_attempt_index` field is retained only as diagnostic metadata; it does not control matching, discount distance, peer grouping, normalization, advantage alignment, or token broadcast. ToolWeave matches at the **individual-call** level, accumulates temporal credit at the **runtime-interaction** level, and applies policy gradients at the **trainable actor-token** level.

### 2. Three-Stage Curriculum

#### Stage 1 — Tool-Use Cold Start

Stage 1 starts from `Qwen/Qwen3-4B` and uses the public EnvTuning interaction path to establish strict response formatting and executable tool calls. For rollout $i$, let $C_i$ be its diagnostic-code sequence, $T_i=|C_i|$, and $n_{i,c}$ the count of code $c$. Define

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

The Stage 1 score lies in `[0,2]`. It measures parser-compatible behavior and executable attempted calls; it is not a fixed-denominator task-completion rate. Empty denominators return zero. The scalar score is group-normalized across the 16 rollouts of each prompt. This stage uses the audited reward-side adaptive KL controller (`initial beta = 0.1`, target `0.1`, horizon `10,000`).

ToolWeave's diagnostic-code semantics follow the public EnvTuning implementation used by this project. Where downstream prose is ambiguous, the executable upstream implementation is treated as the implementation reference.

#### Stage 2 — Progress-Reward Learning

Stage 2 begins from Stage 1 update 25 and replaces the cold-start score with the RODS fixed-denominator Progress Reward. If user turn $u$ is successful only when both its state and response conditions pass, then

$$
R_P^{(i)}=
\frac{1}{U_i}
\sum_{u=1}^{U_i}
\mathbf{1}[\text{user turn }u\text{ succeeds in }\tau_i].
$$

$U_i$ is the number of expected BFCL user turns. Missing terminal outcomes, truncation, and early termination remain failures in the denominator. A zero expected-turn denominator returns zero, while more terminal outcomes than the GT contract permits raise an error. Stage 2 uses $R_P$ as its only global reward, disables reward-side KL, and applies the audited loss-side `low_var_kl` reference term with coefficient `0.01`.

#### Stage 3 — Boundary-Guided Fine-Grained Online RL

Stage 3 retains the same global Progress Reward and adds a MatchTIR-derived local residual over the ordered non-answer runtime-interaction timeline. In parallel, grouped $R_P$ statistics drive boundary selection and asynchronous verified synthesis. Local credit changes neither $R_P$ nor the boundary selector.

| Stage | Starting model | Primary signal | Data behavior |
|---|---|---|---|
| Stage 1 | `Qwen/Qwen3-4B` | Format + executable-call score | Fixed 100-row Base training split |
| Stage 2 | Stage 1 update 25 | Fixed-denominator $R_P$ | Same fixed 100-row Base split |
| Stage 3 | Stage 2 update 25 | $A^{g}+A^{\ell}$ in the actor update | 400 original rows plus next-epoch validated candidates |

## Stage 3: ToolWeave

<div align="center">
<img src="assets/toolweave-stage3.svg" alt="ToolWeave Stage 3 method: multi-turn RL, tool matching, dual-level advantage, and asynchronous online data evolution" width="100%">
</div>

For a rollout $\tau_i$, ToolWeave uses the following consistent notation:

| Quantity | Meaning |
|---|---|
| $R_P^{(i)}$ | Scalar fixed-denominator Progress Reward |
| $A_i^{g}$ | Group-normalized global advantage for rollout $i$ |
| $r_{i,u,j}$ | Local reward of non-answer runtime interaction $j$ in BFCL user turn $u$ |
| $R_{i,u,j}^{\ell}$ | User-turn-local discounted return over real runtime depth |
| $A_{i,u,j}^{\ell}$ | Ragged same-runtime-depth normalized local advantage |
| $A_{i,u,j}^{\mathrm{TW}}$ | Advantage used on actor tokens belonging to runtime interaction $j$ |

Implementation fields map to this notation as follows: prompt identity `uid` corresponds to $q$; `user_turn_id` to $u$; and `runtime_interaction_index` to $j$. The backward-compatible field `policy_step_id` remains a legacy alias for $j$, while `tool_attempt_index` is diagnostic only. The package path `rods_matchtir_v1` is likewise a legacy module name; the active configuration selects `runtime_interaction_final`. This is the frozen ToolWeave Stage-3 formal-training credit-assignment algorithm.

Throughout the method, $A^{\mathrm{TW}}\equiv A^{\mathrm{ToolWeave}}$. Numeric artifact tables retain the compatibility labels $A_{\mathrm{RODS}}\equiv A^g$, $A_{\mathrm{local}}\equiv A^{\ell}$, and $A_{\mathrm{TW}}\equiv A^{\mathrm{ToolWeave}}$.

### 3.1 Reward Modeling

#### Global Task Reward

MatchTIR combines process-level interaction reward with a final-answer outcome term. ToolWeave does **not** use MatchTIR's final-answer F1 as its global task reward. Its only global task reward is the RODS Progress Reward $R_P^{(i)}$ defined above. Local matching quantities never enter $R_P$.

#### Tool-Call Matching Matrix

For predicted call $p$ and GT call $g$, first define a tool-name score:

$$
S_{\mathrm{tn}}(p,g)=
\begin{cases}
1, & p,g\text{ are valid and their function names match case-insensitively},\\
0, & \text{otherwise}.
\end{cases}
$$

Let $N_p$ and $N_g$ be the predicted and GT argument-name collections. MatchTIR's paper defines the parameter-name component with ordinary set Jaccard:

$$
S_{\mathrm{pn}}^{\mathrm{paper}}(p,g)=
\frac{|\mathrm{set}(N_p)\cap\mathrm{set}(N_g)|}
{|\mathrm{set}(N_p)\cup\mathrm{set}(N_g)|}.
$$

ToolWeave's audited implementation instead follows the upstream-code-compatible `Counter`-based **multiset** form. For multiplicity count $C_N(x)$,

$$
I_{\mathrm{multi}}(N_p,N_g)=
\sum_x \min(C_{N_p}(x),C_{N_g}(x)),
$$

$$
S_{\mathrm{pn}}(p,g)=
\frac{I_{\mathrm{multi}}(N_p,N_g)}
{|N_p|+|N_g|-I_{\mathrm{multi}}(N_p,N_g)}.
$$

When both argument-name collections are empty, the implementation returns $S_{\mathrm{pn}}=1$. The parameter-content component counts exact structured equality only over GT argument keys:

$$
S_{\mathrm{pc}}(p,g)=
\sum_{k\in N_g}
\mathbf{1}[k\in N_p\ \text{and}\ p_k=g_k].
$$

The final ToolWeave call similarity is

$$
S(p,g)=
S_{\mathrm{tn}}(p,g)
\frac{S_{\mathrm{tn}}(p,g)+S_{\mathrm{pn}}(p,g)+S_{\mathrm{pc}}(p,g)}{2+|N_g|}.
$$

Invalid calls and function-name mismatches therefore score zero. Parameter values are compared after conversion to stable JSON-compatible built-ins; no fuzzy value matching is introduced.

> **Paper/implementation distinction.** MatchTIR's paper uses set Jaccard for parameter names. ToolWeave documents and executes the multiset form found in the audited source path; the README does not silently replace implementation behavior with the paper equation.

#### Hard Bipartite Assignment

For one rollout and one BFCL user turn, let $m=|P_{i,u}|$, $n=|G_u|$, and form the complete matrix

$$
\mathbf{S}\in\mathbb{R}^{m\times n},
\qquad
\mathbf{S}_{ab}=S(p_a,g_b).
$$

ToolWeave solves the maximum-weight one-to-one assignment

$$
\max_{x}\sum_{a=1}^{m}\sum_{b=1}^{n}x_{ab}\mathbf{S}_{ab},
$$

subject to

$$
x_{ab}\in\lbrace 0,1\rbrace,
\qquad
\sum_b x_{ab}\le 1,
\qquad
\sum_a x_{ab}\le 1.
$$

The call reward is

$$
r_{p_a}=
\begin{cases}
\mathbf{S}_{ab}, & x_{ab}=1\text{ and }\mathbf{S}_{ab}>0,\\
0, & \text{otherwise}.
\end{cases}
$$

The implementation uses `unmatched_penalty = 0.0`. One-to-one assignment prevents duplicate predicted calls from repeatedly claiming the same GT call, which independent per-call maxima would allow.

#### Call-Level Reward to Interaction Reward

A single runtime interaction may contain more than one parsed call. Individual-call rewards are averaged back to that one interaction:

$$
r_{i,u,j}=
\begin{cases}
\dfrac{1}{|P_{i,u,j}|}\displaystyle\sum_{p\in P_{i,u,j}}r_p,
& |P_{i,u,j}|>0,\\
0, & |P_{i,u,j}|=0.
\end{cases}
$$

Only successfully parsed calls participate in matching. Any unparsed non-answer runtime interaction therefore remains one temporal step with $P_{i,u,j}=\varnothing$ and $r_{i,u,j}=0$. A successfully parsed call remains in matching even if its later environment execution fails: the local branch measures tool-call semantic correctness, while the global Progress Reward measures stateful task success. Multiple calls inside one action are never converted into multiple temporal steps.

This matching and interaction-return construction applies to BFCL user turns with a non-empty executable ground-truth call structure $G_u$. When $G_u=\varnothing$, the local branch abstains: $A_{i,u,j}^{\ell}=0$, and actor tokens remain global-only. ToolWeave does not invent a local clarification or final-answer reward for empty-GT turns.

### 3.2 Dual-Level Advantage Estimation

#### Global Advantage

For the $K$ rollouts sampled from the same prompt, ToolWeave computes the unchanged GRPO advantage from $R_P$:

$$
\mu_q^g=\frac{1}{K}\sum_{k=1}^{K}R_P^{(k)},
$$

$$
\sigma_q^g=
\sqrt{
\frac{1}{K-1}
\sum_{k=1}^{K}
\left(R_P^{(k)}-\mu_q^g\right)^2
},
$$

$$
A_i^g=
\frac{R_P^{(i)}-\mu_q^g}{\sigma_q^g+\epsilon},
\qquad \epsilon=10^{-6}.
$$

This scalar is the trajectory-level signal and is shared by trainable actor tokens under the existing response/loss mask. The implementation internally retains the compatibility labels `A_RODS` and `rods_advantages`; the public notation $A^g$ avoids implying that RODS defines a separately named advantage estimator.

#### Local Discounted Return

Within one BFCL user turn $u$, rewards are accumulated backward over the complete non-answer runtime-interaction sequence:

$$
R_{i,u,j}^{\ell}=
\sum_{h=j}^{J_{i,u}-1}
\gamma^{h-j}r_{i,u,h},
\qquad \gamma=0.9.
$$

> **ToolWeave adaptation.** Parser-rejected and otherwise unparsed non-answer interactions remain real discount steps at $r=0$; they are never deleted before computing the return. The accumulator resets at the next BFCL user turn, so local reward never propagates across that boundary. A valid final answer/turn closure is excluded rather than appended as a zero-reward local step.

#### Ragged Same-Runtime-Depth Normalization

Different rollouts can contain different numbers of non-answer runtime interactions. Define

$$
\mathcal{D}_{i,u}=\mathrm{dom}(\mathcal{I}_{i,u})=\lbrace 0,\ldots,J_{i,u}-1\rbrace,
$$

and the ragged same-runtime-depth peer set

$$
\mathcal{S}_{q,u,j}=
\left\lbrace
i\in\lbrace 1,\ldots,K\rbrace
:
j\in\mathcal{D}_{i,u}
\right\rbrace.
$$

Peer membership depends only on actual interaction existence at the same prompt, user turn, and runtime depth. Missing late interactions are absent, not zero-valued samples. Let

$$
n_{q,u,j}=|\mathcal{S}_{q,u,j}|.
$$

For $n_{q,u,j}\ge 2$, define

$$
\mu_{q,u,j}^{\ell}=
\frac{1}{n_{q,u,j}}
\sum_{i\in\mathcal{S}_{q,u,j}}R_{i,u,j}^{\ell},
$$

$$
s_{q,u,j}^{\ell}=
\sqrt{
\frac{1}{n_{q,u,j}-1}
\sum_{i\in\mathcal{S}_{q,u,j}}
\left(R_{i,u,j}^{\ell}-\mu_{q,u,j}^{\ell}\right)^2
},
$$

The local advantage is defined for every support size by

$$
A_{i,u,j}^{\ell}=
\frac{R_{i,u,j}^{\ell}-\mu_{q,u,j}^{\ell}}
{s_{q,u,j}^{\ell}+\epsilon}
\quad
\text{when }n_{q,u,j}\ge 2\text{ and }0<s_{q,u,j}^{\ell}<\infty,
\qquad \epsilon=10^{-6}.
$$

Otherwise,

$$
A_{i,u,j}^{\ell}=0.
$$

$s_{q,u,j}^{\ell}$ is the unbiased sample standard deviation and is defined only for support of at least two. The implementation key is `(uid, user_turn_id, runtime_interaction_index)`. `tool_attempt_index` never enters discounting, peer grouping, normalization, or advantage alignment.

### 3.3 Policy Optimization

The core Stage 3 advantage is

$$
\boxed{A_{i,u,j}^{\mathrm{TW}}=A_i^g+\lambda_{\mathrm{local}}A_{i,u,j}^{\ell}}
\qquad \lambda_{\mathrm{local}}=1.0.
$$

For a token inside an actor-span-reliable non-answer runtime interaction, the actor uses $A_i^g+A_{i,u,j}^{\ell}$. Every other trainable actor token uses $A_i^g$. There is no divide-by-two fusion, post-fusion normalization, RMS rescaling, or adaptive local weighting.

#### Token Assignment

The local scalar is broadcast over the trainable actor tokens inside the originating runtime-interaction assistant span, intersected with the existing actor response/loss mask. This includes parser-rejected or unclassified malformed non-answer generations when temporal identity and actor span are reliable. ToolWeave does **not** define a finer per-call JSON-token-subspan objective. User messages, tool observations, environment tokens, and other non-actor positions receive zero local residual, and the implementation asserts that local values cannot leak outside the actor mask.

#### Implementation Invariants

| Condition | Local branch behavior |
|---|---|
| Empty GT / Missing turn | $A^{\ell}=0$; no clarification or final-answer local reward is invented |
| Valid final answer / turn closure | Excluded from the local sequence; answer actor tokens receive global-only credit |
| Temporally reliable unparsed non-answer interaction | Enters the local sequence with no parsed calls and $r=0$ |
| Unreliable user-turn ownership or runtime ordering | User-turn local branch fails closed |
| Reliable temporal provenance but unreliable actor span | Remains in the return chain, but receives no token-level local residual |
| Successfully parsed call with environment execution failure | Participates in semantic call matching; stateful failure remains visible through $A^g$ |
| No rollout-level provenance or batch misalignment | Exact global baseline |
| Ragged peer support below 2 | $A^{\ell}=0$ |
| Zero variance or non-finite sample std | $A^{\ell}=0$ |
| Local disabled or weight set to zero | Original global tensors are returned unchanged |

At every new BFCL user turn, `runtime_interaction_index` and the local discount accumulator restart from zero. `tool_attempt_index` may also reset for diagnostics, but it has no formal credit-assignment role.

#### PPO/GRPO Actor Update

The fused advantage is passed to the existing actor-only PPO/GRPO path. For actor-token position $z$,

$$
\rho_{i,z}=
\exp\left(
\log\pi_{\theta}(a_{i,z})-
\log\pi_{\mathrm{old}}(a_{i,z})
\right),
$$

$$
\ell_{i,z}^{(1)}=-A_{i,z}^{\mathrm{TW}}\rho_{i,z},
\qquad
\ell_{i,z}^{(2)}=
-A_{i,z}^{\mathrm{TW}}
\mathrm{clip}(\rho_{i,z},1-\epsilon_{\mathrm{low}},1+\epsilon_{\mathrm{high}}).
$$

The unchanged implementation applies the configured clipped/dual-clipped surrogate, response mask, reference KL, backward pass, and optimizer step. Local credit is not added to reward-side KL or to boundary statistics.

<details>
<summary>Implementation-level PPO/KL details</summary>

- `epsilon_low = 0.20`, `epsilon_high = 0.28`, and dual-clip constant `10`.
- The log-ratio is clamped to `[-20,20]` before exponentiation.
- The reference term uses `kl_loss_type = low_var_kl` with coefficient `0.01`; reward-side KL is disabled.
- The masked actor loss uses the existing `seq-mean-token-mean` aggregation.
- For GRPO tensor-contract consistency, the implementation mirrors the local residual into `returns_new = returns_global + lambda_local * A_local`; the actor-only path consumes `advantages`, not a critic return. This does not introduce critic learning.

</details>

#### Real Rollout Case Study: Interaction-Level Credit Assignment

**Full trajectory and reproducible group statistics:** [Hugging Face dataset](https://huggingface.co/datasets/muradil211/ToolWeave-BFCL-Rollout-Case-Study)

> [!IMPORTANT]
> The **current implementation** deterministically reproduces this case study from parser/provenance through trainer fusion. Every temporally reliable non-answer runtime interaction remains in the local sequence; unparsed interactions have $P_{i,u,j}=\varnothing$ and $r_{i,u,j}=0$, while only successfully parsed calls enter matching. The stored rollout is immutable Stage 3 formal-training evidence.

The audited record is JSONL line 10 (`trajectory_index=9`) from that artifact at `global_step=2`, `batch_index=1`, and `epoch=0`. It is one complete stateful BFCL sample with five user turns:

| BFCL user turn | Ground-truth calls |
|---:|---|
| 0 | `get_flight_cost`, `book_flight` |
| 1 | `retrieve_invoice` |
| 2 | `contact_customer_support` |
| 3 | `ticket_login`, `create_ticket` |
| 4 | `edit_ticket` |

The original sample ID is `multi_turn_base_156`. Its group/prompt UID is `1b94ddc9-3612-48c4-acf2-7b755d72330f`, shared by all $K=16$ rollouts. The individual rollout ID is `8516d0df-e6fb-4a67-969d-637bfd967e77`, with `rollout_offset=9`. The group UID is therefore not a unique rollout identifier.

For this local-credit audit, $j$ indexes every non-answer runtime interaction before valid answer/turn closure. The current provenance writes `runtime_interaction_index=j`; legacy `policy_step_id` is a compatible alias, while `tool_attempt_index` is diagnostic metadata only. Terminal answers remain in the complete raw trajectory but are excluded from the local sequence and receive global credit only.

The replay below applies exactly the reward, runtime-depth return, ragged peer normalization, and additive fusion defined in Sections 3.1–3.3.

##### Why Dual-Level Credit Assignment?

A trajectory-only global estimator assigns the same signal to every interaction in one rollout:

$$
A_{i,u,j}^{\mathrm{trajectory}}=A_i^g.
$$

It can distinguish which rollout was better overall, but it cannot distinguish a strong interaction from a weak interaction inside that same rollout. ToolWeave adds the interaction-level residual:

$$
A_{i,u,j}^{\mathrm{TW}}
=A_i^g+A_{i,u,j}^{\ell}.
$$

This permits $A_{i,u,j_0}^{\mathrm{TW}}\ne A_{i,u,j_1}^{\mathrm{TW}}$ and can even produce opposite signs at two runtime depths within one globally successful trajectory.

For User Turn 3, 15 peer rollouts use two parsed one-call actions—`ticket_login` followed by `create_ticket`—before their terminal answer action. The special rollout instead makes five parser-rejected tool attempts and then self-corrects with one legal tool action containing a JSON array of two calls:

```text
Efficient non-answer runtime chain
j=0  ticket_login  ── r=1.000000
                         ↓
j=1  create_ticket ── r=1.000000

Special recovery chain
j=0  parse_error ── r=0
         ↓
j=1  parse_error ── r=0
         ↓
j=2  parse_error ── r=0
         ↓
j=3  parse_error ── r=0
         ↓
j=4  parse_error ── r=0
         ↓
j=5  ONE valid tool-call action ── r=1.000000
       ├── ticket_login       call reward=1.000000
       └── create_ticket      call reward=1.000000
```

The five errors and final action were replayed through the current runtime parser. The final two calls were scored with the current ToolWeave similarity and one true SciPy maximum-weight Hungarian assignment over the complete User Turn 3 call set.

<!-- TOOLWEAVE_CASE_STUDY_CORE_TABLE_BEGIN -->
| Runtime depth $j$ | Runtime outcome | Parsed calls | Call rewards | $r_j$ | $R_j$ | Peer support | Peer mean $R$ | Peer sample std $R$† | $A_{\mathrm{local}}$ | $A_{\mathrm{RODS}}$ | $A_{\mathrm{TW}}$ |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | Parse error | — | — | 0.000000 | 0.590490 | 16 | 1.813468 | 0.326664 | -3.7438 | -0.4967 | -4.2405 |
| 1 | Parse error | — | — | 0.000000 | 0.656100 | 16 | 0.973298 | 0.087103 | -3.6416 | -0.4967 | -4.1383 |
| 2 | Parse error | — | — | 0.000000 | 0.729000 | 1 | 0.729000 | 0.000000 | 0.0000 | -0.4967 | -0.4967 |
| 3 | Parse error | — | — | 0.000000 | 0.810000 | 1 | 0.810000 | 0.000000 | 0.0000 | -0.4967 | -0.4967 |
| 4 | Parse error | — | — | 0.000000 | 0.900000 | 1 | 0.900000 | 0.000000 | 0.0000 | -0.4967 | -0.4967 |
| 5 | Valid two-call action | `ticket_login`, `create_ticket` | `[1.000000, 1.000000]` | 1.000000 | 1.000000 | 1 | 1.000000 | 0.000000 | 0.0000 | -0.4967 | -0.4967 |
<!-- TOOLWEAVE_CASE_STUDY_CORE_TABLE_END -->

`†` For peer support below two, the unbiased sample standard deviation is mathematically undefined. The production diagnostic records `0.000000` as a sentinel, and the estimator abstains with $A_{\mathrm{local}}=0$.

The special rollout closes four of five expected BFCL user turns, so the source-of-truth fixed-denominator wrapper gives $R_P=4/5=0.8$. Across the 16-rollout group, the recomputed Progress Rewards have mean `0.925000` and unbiased sample standard deviation `0.251661`, producing $A_{\mathrm{RODS}}=-0.496698$ for this rollout.

ToolWeave abstains from local relative credit when no same-runtime-depth peer exists; the global RODS advantage remains active. Thus the late self-correction is not assigned fabricated singleton credit, while the earlier inefficient/error interactions at $j=0$ and $j=1$ are sharply distinguished from their peers.

> **Scope of the peer summary.** Runtime patterns, immediate rewards, and discounted returns below describe **User Turn 3 only**. Full-rollout $R_P$ and $A_{\mathrm{RODS}}$ are trajectory-level quantities computed over **all five BFCL user turns**. A rollout can therefore have locally perfect User Turn 3 calls yet have $R_P=0$ when none of its five user turns receives terminal success. Here $A_{\mathrm{RODS}}$ is the same-prompt, $K=16$ normalized global advantage derived from full-rollout $R_P$, not a User Turn 3 local advantage.

<!-- TOOLWEAVE_CASE_STUDY_K16_TABLE_BEGIN -->
| Offset | Runtime-interaction pattern *(User Turn 3)* | Immediate rewards *(User Turn 3)* | Discounted returns *(User Turn 3)* | Full-rollout $R_P$ *(5 turns)* | Full-rollout $A_{\mathrm{RODS}}$ |
|---:|---|---|---|---:|---:|
| 0 | parsed → parsed | `[1.000000, 1.000000]` | `[1.900000, 1.000000]` | 1.000000 | 0.2980 |
| 1 | parsed → parsed | `[1.000000, 1.000000]` | `[1.900000, 1.000000]` | 1.000000 | 0.2980 |
| 2 | parsed → parsed | `[1.000000, 1.000000]` | `[1.900000, 1.000000]` | 0.000000 | -3.6756 |
| 3 | parsed → parsed | `[1.000000, 1.000000]` | `[1.900000, 1.000000]` | 1.000000 | 0.2980 |
| 4 | parsed → parsed | `[1.000000, 1.000000]` | `[1.900000, 1.000000]` | 1.000000 | 0.2980 |
| 5 | parsed → parsed | `[1.000000, 1.000000]` | `[1.900000, 1.000000]` | 1.000000 | 0.2980 |
| 6 | parsed → parsed | `[1.000000, 1.000000]` | `[1.900000, 1.000000]` | 1.000000 | 0.2980 |
| 7 | parsed → parsed | `[1.000000, 1.000000]` | `[1.900000, 1.000000]` | 1.000000 | 0.2980 |
| 8 | parsed → parsed | `[1.000000, 1.000000]` | `[1.900000, 1.000000]` | 1.000000 | 0.2980 |
| **9** | **5× parse error → parsed[2 calls]** | **`[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000]`** | **`[0.590490, 0.656100, 0.729000, 0.810000, 0.900000, 1.000000]`** | **0.800000** | **-0.4967** |
| 10 | parsed → parsed | `[1.000000, 1.000000]` | `[1.900000, 1.000000]` | 1.000000 | 0.2980 |
| 11 | parsed → parsed | `[1.000000, 1.000000]` | `[1.900000, 1.000000]` | 1.000000 | 0.2980 |
| 12 | parsed → parsed | `[1.000000, 1.000000]` | `[1.900000, 1.000000]` | 1.000000 | 0.2980 |
| 13 | parsed → parsed | `[1.000000, 1.000000]` | `[1.900000, 1.000000]` | 1.000000 | 0.2980 |
| 14 | parsed → parsed | `[1.000000, 0.916667]` | `[1.825000, 0.916667]` | 1.000000 | 0.2980 |
| 15 | parsed → parsed | `[1.000000, 1.000000]` | `[1.900000, 1.000000]` | 1.000000 | 0.2980 |
<!-- TOOLWEAVE_CASE_STUDY_K16_TABLE_END -->

##### Full K=16 User Turn 3 Interaction Audit

The following table is generated directly by deterministic replay through the production implementation. It expands the 15 non-special peer rollouts into one row per actual User Turn 3 non-answer runtime interaction; the special offset 9 is intentionally not repeated here because its six interactions are shown above. The complete 36-row K=16 artifact, including offset 9, is available as [`user_turn3_k16_full_interaction_advantage.json`](https://huggingface.co/datasets/muradil211/ToolWeave-BFCL-Rollout-Case-Study/blob/main/analysis/user_turn3_k16_full_interaction_advantage.json) and [`CSV`](https://huggingface.co/datasets/muradil211/ToolWeave-BFCL-Rollout-Case-Study/blob/main/analysis/user_turn3_k16_full_interaction_advantage.csv).

<details>
<summary><b>Full K=16 interaction-level audit (30 peer rows)</b></summary>

<!-- TOOLWEAVE_CASE_STUDY_FULL_INTERACTION_TABLE_BEGIN -->
| Offset | $j$ | Runtime outcome | Parsed calls | Call rewards | $r_j$ | $R_j$ | Peer support | Peer mean $R$ | Peer sample std $R$† | $A_{\mathrm{local}}$ | $A_{\mathrm{RODS}}$ | $A_{\mathrm{TW}}$ |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | Valid tool action | ticket_login | [1.000000] | 1.000000 | 1.900000 | 16 | 1.813468 | 0.326664 | 0.2649 | 0.2980 | 0.5629 |
| 0 | 1 | Valid tool action | create_ticket | [1.000000] | 1.000000 | 1.000000 | 16 | 0.973298 | 0.087103 | 0.3066 | 0.2980 | 0.6046 |
| 1 | 0 | Valid tool action | ticket_login | [1.000000] | 1.000000 | 1.900000 | 16 | 1.813468 | 0.326664 | 0.2649 | 0.2980 | 0.5629 |
| 1 | 1 | Valid tool action | create_ticket | [1.000000] | 1.000000 | 1.000000 | 16 | 0.973298 | 0.087103 | 0.3066 | 0.2980 | 0.6046 |
| 2 | 0 | Valid tool action | ticket_login | [1.000000] | 1.000000 | 1.900000 | 16 | 1.813468 | 0.326664 | 0.2649 | -3.6756 | -3.4107 |
| 2 | 1 | Valid tool action | create_ticket | [1.000000] | 1.000000 | 1.000000 | 16 | 0.973298 | 0.087103 | 0.3066 | -3.6756 | -3.3690 |
| 3 | 0 | Valid tool action | ticket_login | [1.000000] | 1.000000 | 1.900000 | 16 | 1.813468 | 0.326664 | 0.2649 | 0.2980 | 0.5629 |
| 3 | 1 | Valid tool action | create_ticket | [1.000000] | 1.000000 | 1.000000 | 16 | 0.973298 | 0.087103 | 0.3066 | 0.2980 | 0.6046 |
| 4 | 0 | Valid tool action | ticket_login | [1.000000] | 1.000000 | 1.900000 | 16 | 1.813468 | 0.326664 | 0.2649 | 0.2980 | 0.5629 |
| 4 | 1 | Valid tool action | create_ticket | [1.000000] | 1.000000 | 1.000000 | 16 | 0.973298 | 0.087103 | 0.3066 | 0.2980 | 0.6046 |
| 5 | 0 | Valid tool action | ticket_login | [1.000000] | 1.000000 | 1.900000 | 16 | 1.813468 | 0.326664 | 0.2649 | 0.2980 | 0.5629 |
| 5 | 1 | Valid tool action | create_ticket | [1.000000] | 1.000000 | 1.000000 | 16 | 0.973298 | 0.087103 | 0.3066 | 0.2980 | 0.6046 |
| 6 | 0 | Valid tool action | ticket_login | [1.000000] | 1.000000 | 1.900000 | 16 | 1.813468 | 0.326664 | 0.2649 | 0.2980 | 0.5629 |
| 6 | 1 | Valid tool action | create_ticket | [1.000000] | 1.000000 | 1.000000 | 16 | 0.973298 | 0.087103 | 0.3066 | 0.2980 | 0.6046 |
| 7 | 0 | Valid tool action | ticket_login | [1.000000] | 1.000000 | 1.900000 | 16 | 1.813468 | 0.326664 | 0.2649 | 0.2980 | 0.5629 |
| 7 | 1 | Valid tool action | create_ticket | [1.000000] | 1.000000 | 1.000000 | 16 | 0.973298 | 0.087103 | 0.3066 | 0.2980 | 0.6046 |
| 8 | 0 | Valid tool action | ticket_login | [1.000000] | 1.000000 | 1.900000 | 16 | 1.813468 | 0.326664 | 0.2649 | 0.2980 | 0.5629 |
| 8 | 1 | Valid tool action | create_ticket | [1.000000] | 1.000000 | 1.000000 | 16 | 0.973298 | 0.087103 | 0.3066 | 0.2980 | 0.6046 |
| 10 | 0 | Valid tool action | ticket_login | [1.000000] | 1.000000 | 1.900000 | 16 | 1.813468 | 0.326664 | 0.2649 | 0.2980 | 0.5629 |
| 10 | 1 | Valid tool action | create_ticket | [1.000000] | 1.000000 | 1.000000 | 16 | 0.973298 | 0.087103 | 0.3066 | 0.2980 | 0.6046 |
| 11 | 0 | Valid tool action | ticket_login | [1.000000] | 1.000000 | 1.900000 | 16 | 1.813468 | 0.326664 | 0.2649 | 0.2980 | 0.5629 |
| 11 | 1 | Valid tool action | create_ticket | [1.000000] | 1.000000 | 1.000000 | 16 | 0.973298 | 0.087103 | 0.3066 | 0.2980 | 0.6046 |
| 12 | 0 | Valid tool action | ticket_login | [1.000000] | 1.000000 | 1.900000 | 16 | 1.813468 | 0.326664 | 0.2649 | 0.2980 | 0.5629 |
| 12 | 1 | Valid tool action | create_ticket | [1.000000] | 1.000000 | 1.000000 | 16 | 0.973298 | 0.087103 | 0.3066 | 0.2980 | 0.6046 |
| 13 | 0 | Valid tool action | ticket_login | [1.000000] | 1.000000 | 1.900000 | 16 | 1.813468 | 0.326664 | 0.2649 | 0.2980 | 0.5629 |
| 13 | 1 | Valid tool action | create_ticket | [1.000000] | 1.000000 | 1.000000 | 16 | 0.973298 | 0.087103 | 0.3066 | 0.2980 | 0.6046 |
| 14 | 0 | Valid tool action | ticket_login | [1.000000] | 1.000000 | 1.825000 | 16 | 1.813468 | 0.326664 | 0.0353 | 0.2980 | 0.3333 |
| 14 | 1 | Valid tool action | create_ticket | [0.916667] | 0.916667 | 0.916667 | 16 | 0.973298 | 0.087103 | -0.6502 | 0.2980 | -0.3521 |
| 15 | 0 | Valid tool action | ticket_login | [1.000000] | 1.000000 | 1.900000 | 16 | 1.813468 | 0.326664 | 0.2649 | 0.2980 | 0.5629 |
| 15 | 1 | Valid tool action | create_ticket | [1.000000] | 1.000000 | 1.000000 | 16 | 0.973298 | 0.087103 | 0.3066 | 0.2980 | 0.6046 |
<!-- TOOLWEAVE_CASE_STUDY_FULL_INTERACTION_TABLE_END -->

</details>

##### Four Credit-Assignment Regimes in One Real K=16 Group

| Observed behavior | Global $A^g$ | Local $A^{\ell}$ | Fused $A^{\mathrm{TW}}$ | Credit-assignment effect |
|---|:---:|:---:|:---:|---|
| Efficient + globally successful | + | + | stronger + | Strengthens efficient, correct interactions |
| Globally successful + locally weaker | + | − | can become − | Can suppress a weaker interaction despite positive trajectory credit |
| Locally correct + globally failed | − | + | still − here, but softened | Preserves local evidence while whole-trajectory consistency remains decisive |
| Repeated failure + delayed recovery | − | early strong − | early stronger negative | Suppresses supported early errors while preserving the final correction's immediate reward |

The four rows are all observed in this group:

- **Offset 0 — efficient and globally successful.** Both User Turn 3 calls match perfectly, and positive local residuals align with positive full-rollout credit. The fused signal is therefore stronger at both runtime depths.

- **Offset 14 — globally successful but locally weaker.** The second `create_ticket` call scores `0.916667`. Although $A^g=+0.298019$, its local residual is `-0.650158`, flipping the fused interaction advantage to `-0.352139`. Trajectory-only supervision would instead assign the same positive global value to both interactions.

- **Offset 2 — locally correct but globally failed.** `ticket_login` and `create_ticket` each match at `1.000000`, producing positive local residuals. The full rollout nevertheless has $R_P=0$ and $A^g=-3.675562$, so both fused values remain negative, but are softened by the correct local evidence.

  The runtime replay identifies the cause precisely. In User Turn 0, the model emitted only `get_flight_cost` with `travel_from='SAN'`, while the GT uses `travel_from='SFO'` and also requires `book_flight`. At the User Turn 3 terminal check, `state_checker` reports `multi_turn:instance_state_mismatch` in `booking_record` and `credit_card_list`: the model has an empty booking record and card balance `6000`, whereas the GT state contains booking `3426812` and balance `5000`. The TicketAPI state itself matches. Thus these User Turn 3 actions are locally correct, not locally bad; the negative fused value comes from the stateful global branch.

- **Offset 9 — repeated parser failure with delayed recovery.** Parser failures occupy five real runtime depths at $r=0$. The first two depths have supported negative local residuals; later singleton depths abstain. The final valid two-call action retains $r=1$, and its negative fused value comes only from $A^g=-0.496698$, not a fabricated local penalty.

This real K=16 formal-training group shows strictly finer interaction-level credit resolution than trajectory-only supervision: the estimator can strengthen efficient interactions, suppress a relatively weaker interaction inside a successful rollout, preserve positive local evidence inside a globally failed rollout, and distinguish direct execution from delayed recovery. This is evidence about credit-assignment behavior, not a controlled claim of superior final-policy performance. Full arguments, runtime messages, parser provenance, rollout identities, exact floating-point values, and the complete trajectory are available in the [dataset record](https://huggingface.co/datasets/muradil211/ToolWeave-BFCL-Rollout-Case-Study/blob/main/data/multi_turn_base_156_rollout_offset_9.json), [K=16 summary](https://huggingface.co/datasets/muradil211/ToolWeave-BFCL-Rollout-Case-Study/blob/main/analysis/user_turn3_k16_credit_summary.json), and [full interaction audit](https://huggingface.co/datasets/muradil211/ToolWeave-BFCL-Rollout-Case-Study/blob/main/analysis/user_turn3_k16_full_interaction_advantage.json).

> **Matching provenance.** The MatchTIR paper describes maximum-weight Hungarian/KM assignment. At audited public commit [`975c453`](https://github.com/quchangle1/MatchTIR/commit/975c4535fbb86a49f21ff7d291a1fa822f827684), the helper named `hungarian_assignment` performs greedy sorted-edge matching. ToolWeave's implemented solver uses the paper-style one-to-one objective with SciPy's true `linear_sum_assignment(..., maximize=True)`.

### 3.4 Boundary-Guided Online Data Evolution

The same $K$ rollout rewards also produce a prompt-level mean

$$
\bar r_P(q)=\frac{1}{K}\sum_{i=1}^{K}R_P^{(i)}.
$$

RODS-style capability regions are

| Region | Rule |
|---|---|
| Too hard | $\bar r_P(q)<0.20$ |
| Boundary | $0.20\le\bar r_P(q)\le0.85$ |
| Mastered | $\bar r_P(q)>0.85$ |

Boundary examples are prioritized by

$$
\phi(q)=4\bar r_P(q)\left(1-\bar r_P(q)\right).
$$

After a sample-identity cooldown, candidates are partitioned into Base, Missing Function, Missing Parameter, and Long Context buckets. Each bucket is ranked by descending $\phi$, limited by its quota $M_{\tau}$, and bounded by total budget $M$, with

$$
\sum_{\tau}M_{\tau}=M.
$$

There is no quota redistribution when one type has too few eligible samples. The paper specifies the mechanism but does not publish one unique numeric default for $M$, $M_{\tau}$, or cooldown $c$; ToolWeave's formal training configuration supplies these project hyperparameters explicitly, and validation fails fast if they are omitted.

ToolWeave formal training fixes these project choices to

$$
M=16,
\qquad
M_{\tau}=4
\quad \forall \tau\in
\lbrace
\mathrm{Base},\mathrm{MF},\mathrm{MP},\mathrm{LC}
\rbrace,
\qquad
c=13.
$$

These are ToolWeave project hyperparameters and are not attributed to RODS. The four per-type quotas sum exactly to the total selection budget.

This branch consumes only $R_P$. It never consumes $A^{\ell}$, call similarities, or the fused advantage. Seed dispatch occurs after a successful optimizer update, and the separate generator proceeds asynchronously while policy learning can continue.

Validated candidates generated in epoch `n` are staged and become eligible only at epoch `n+1`. ToolWeave implements a reproducible adaptation of RODS-style lifecycle management, not a claim of a verbatim unpublished RODS lifecycle. It protects all original rows, limits new admission to at most `floor(0.20 × active_pool_before)`, caps the generated sub-pool at 400, supports trial eviction and drift retirement for generated rows, and persists deferred validated candidates for later epochs.

<details>
<summary>Audited lifecycle constants and edge behavior</summary>

- One observation is required before a generated row leaves its trial state.
- Trial rows below `0.20` can be evicted as too hard.
- Observed generated rows above `0.95` can retire as mastered; the lower retirement boundary is `0.20`.
- Stale retirement is an optional disabled hook because no reproducible paper-default stale window is available.
- If restored state is already above the 400-row generated cap, observed generated rows with the lowest available $\phi$ are pruned first. The implementation does not fabricate priorities for unobserved trial rows.

</details>

### ToolWeave Stage 3 at a Glance

```text
Input: current policy, active pool, K rollouts per prompt,
       per-user-turn executable GT call structures and environment contract

For each training update:
  1. Sample K stateful BFCL rollouts for each prompt.
  2. Compute fixed-denominator Progress Reward R_P.
  3. Group-normalize R_P to obtain A_global.
  4. Build each BFCL user turn's ordered non-answer runtime-interaction sequence.
  5. Match successfully parsed calls to GT once per BFCL user turn.
  6. Average call rewards inside each runtime interaction; unparsed interactions receive r=0.
  7. Compute discounted local returns over real runtime depth within each BFCL user turn.
  8. Normalize over ragged (prompt, user turn, runtime depth) peers to obtain A_local.
  9. Form A_ToolWeave = A_global + A_local on reliable non-answer actor spans.
 10. Run the unchanged PPO/GRPO actor update.
 11. Select up to M=16 boundary seeds from grouped R_P only (4/type, cooldown c=13).
 12. Synthesize and validate executable candidate trajectories.
 13. Admit eligible candidates from the next epoch onward.
```

<details>
<summary>Audited Stage 3 implementation map and local-credit constants</summary>

| Responsibility | Public implementation source |
|---|---|
| Global GRPO followed by residual fusion | [`ray_trainer.py`](code/AWorld-RL-stage1-worktree/EnvTuning/verl/verl/trainer/ppo/ray_trainer.py) |
| Parser classification and runtime indexing | [`utils.py`](code/AWorld-RL-stage1-worktree/EnvTuning/env_tuning/interaction/utils.py), [`response_handler.py`](code/AWorld-RL-stage1-worktree/EnvTuning/env_tuning/interaction/response_handler.py), and [`new_multi_turn_fc.py`](code/AWorld-RL-stage1-worktree/EnvTuning/env_tuning/interaction/new_multi_turn_fc.py) |
| Structured rollout provenance and actor-span binding | [`provenance.py`](code/AWorld-RL-stage1-worktree/EnvTuning/env_tuning/rods_matchtir_v1/provenance.py) and [`schemas.py`](code/AWorld-RL-stage1-worktree/EnvTuning/verl/verl/workers/rollout/schemas.py) |
| Call similarity and maximum-weight assignment | [`matching.py`](code/AWorld-RL-stage1-worktree/EnvTuning/env_tuning/rods_matchtir_v1/matching.py) |
| Interaction reward, turn-local return, ragged normalization, token residual, and fusion | [`advantage.py`](code/AWorld-RL-stage1-worktree/EnvTuning/env_tuning/rods_matchtir_v1/advantage.py) |
| Boundary selection and next-epoch lifecycle | [`lifecycle.py`](code/AWorld-RL-stage1-worktree/EnvTuning/env_tuning/rods_matchtir_v1/lifecycle.py) |
| Verified synthesis | [`rods_data_generation_v1/`](code/AWorld-RL-stage1-worktree/EnvTuning/env_tuning/rods_data_generation_v1/) |

The `rods_matchtir_v1` directory name is retained for import compatibility. Its active Stage 3 local-credit mode is `runtime_interaction_final`; historical `tool_attempt_index` metadata is diagnostic only.

| Local-credit setting | Value |
|---|---:|
| enabled | `true` |
| weight | `1.0` |
| discount $\gamma$ | `0.9` |
| matching | `hard` |
| unmatched penalty | `0.0` |
| minimum peer support | `2` |
| normalization epsilon | `1e-6` |
| post-fusion normalization | none |
| local input to boundary selection | no |

</details>

## Verified Online Data Synthesis

The Data-Generation Branch is a separate queue consumer. It does not redo boundary selection and does not block the current optimizer step.

### 1. Boundary Seed Selection

The Training Branch validates and emits the selected `rods_boundary_seed.v1` records after a successful update. Each seed retains its original question, GT, available functions, initial configuration, mean $R_P$, priority $\phi$, source epoch/step, and BFCL type.

### 2. Planning and Function Construction

A RODS-derived planner concept proposes an ordered executable structure and latent narrative. Function sampling is constrained to the audited active 128-function catalog. Parameter generation is grounded in the schema, current environment configuration, earlier successful results, and dependency context. Hallucinated or blocked functions fail closed.

### 3. Real BFCL VM Execution

Ground truth is generated and executed before the natural-language query. User turns share state and execution history. Failures enter a 12-class taxonomy, and only eligible failures trigger deterministic configuration patching, cumulative blocklisting, and feedback-conditioned replanning. A seed is dropped after at most three complete pipeline attempts.

### 4. Query Construction and Whole-Conversation Rewrite

Each executed turn is converted into a class-conditioned natural user query and verified against its actual GT. A whole-conversation rewrite then improves cross-turn coherence without changing the executable intent. Missing Function and Missing Parameter transformations preserve their explicit no-call and recovery-turn protocols.

### 5. ToolWeave Semantic Hardening

These deterministic or structured project guards are ToolWeave additions, not claimed as unpublished official RODS rules:

| Guard | Acceptance contract |
|---|---|
| Argument provenance | Every provided required or optional argument must come from user context, a successful prior result, relevant visible state, an intentional Missing protocol, or a schema-defined default |
| Unit semantics | Numeric quantities must respect schema-audited units or an explicit executable conversion chain |
| Relational ambiguity | Singular extrema such as “latest” must resolve uniquely unless the query itself disambiguates |
| Genuine Missing Parameter | Clarification is legal only when the parameter is absent or ambiguous in policy-visible history |
| Observation entailment | Claims explicitly attributed to prior observations must be supported by those observations |
| Action minimality | Each GT call must be direct intent, a real prerequisite, or a dependency producer |
| Recursive result semantics | Nested/stringified explicit failures are hard failures; audited business negatives remain domain-negative outcomes |
| Exact-content novelty | Canonical training content is deduplicated across source seeds without embedding thresholds |

No deterministic gate can be overridden by the Quality Judge.

### 6. Fresh-VM, Judge, and Candidate Admission

The final trajectory is replayed in a distinct fresh VM initialized from the final configuration. Tool visibility and recursive parameter-complexity gates then run before the Quality Judge. At most one classified refinement cycle is allowed; GT-unfixable defects are dropped. Only records passing every gate, Judge acceptance, the frozen candidate schema, and the Training Branch validator receive `validated=true` and enter the durable candidate queue.

The append-only terminal journal stores a complete successful candidate before candidate-queue publication; restart reconciliation is idempotent. This provides exactly-once logical candidate identity even across crashes at durable boundaries.

RODS-derived high-level concepts include boundary-driven seed selection, planning, function/parameter generation, executable interaction, query construction, whole-trajectory rewriting, quality critique/refinement, and dynamic replay. ToolWeave does not claim the complete reconstructed synthesis branch as official RODS source code. One shared Gemma-4-31B vLLM service is the project's synthesis-backbone substitution; RODS reports Qwen3-32B.

Structural complexity profiles are used only as synthesis guidance and diagnostics where they are reproducibly recoverable. ToolWeave does **not** claim an unpublished official RODS structural-distance acceptance threshold. Public sources do not expose a deterministic HIGH_LEVEL-to-BOTTOM_LEVEL decomposition map, so unsupported HIGH_LEVEL plans fail closed rather than allowing an LLM to invent executable GT.

## Experiments and Results

Stage 1/2 results below come from deterministic one-rollout evaluation artifacts and report internal EnvTuning/RODS reward and protocol metrics. The final Stage 3 checkpoint is additionally reported with complete-entry BFCL Multi-Turn accuracy on the balanced 400-row held-in set. Stage 3 implementation validation and the real K=16 formal-training replay are documented separately. None of these results is presented as an official BFCL leaderboard submission.

### Evaluation Protocol

The environment emits one diagnostic event for each assistant action or closed user turn:

| Code | Executable meaning in this project |
|---:|---|
| `-3` | Response-format/parser failure |
| `-2` | Parsed tool call reached execution but failed in the VM |
| `-1` | Parsed tool call executed without an execution error; this is not terminal task success |
| `0` | Terminal user-turn failure |
| `1` | Terminal user-turn success |

Only `0` and `1` are terminal. Additional diagnostics are

$$
P_i^{\mathrm{observed}}=
\frac{n_{i,1}}{n_{i,0}+n_{i,1}},
\qquad
\mathrm{Coverage}_i=
\frac{n_{i,0}+n_{i,1}}{U_i},
$$

$$
\mathrm{MissingTurns}_i=U_i-(n_{i,0}+n_{i,1}),
\qquad
\mathrm{Incomplete}_i=
\mathbf{1}[n_{i,0}+n_{i,1}<U_i].
$$

`Observed progress` is a Stage 1 diagnostic that divides only by terminal turns that happened to be observed. $R_P$ divides by all expected turns and is the Stage 2/3 training reward. Dataset metrics are sample-macro means:

$$
\frac{1}{N}\sum_{i=1}^{N} m_i.
$$

`Tool-use rate` is the mean of $I_i^{\mathrm{tool}}$; `rounds` is the mean of $T_i$. For action bucket $b$, parser rate is `valid_actions_b / total_actions_b`.

#### Validation dataset identity

| Run | Validation file | Rows | Composition | SHA256 |
|---|---|---:|---|---|
| Stage 1 | `val_100_stratified_seed42.parquet` | 100 | 25 Base + 25 Long Context + 25 Missing Function + 25 Missing Parameter | `479f28404af7a878a637ae71b1f83911c614b38e00c8aa42217db975f88492cc` |
| Stage 2 | `checkpoint_gate_eval/val_base_100.parquet` | 100 | 100 Base | `5f1908c1f2766a5aea38d0a3cbc32f3fe39e132adc372b907c0e49301fa5f664` |

The Stage 1 directory retains validation outputs for updates 10, 15, 20, and 25. No update-5 Stage 1 validation JSON was retained, so no value is reconstructed or imputed.

The canonical 400-row held-in evaluation set is assembled from the upstream [AWorld-RL EnvTuning data](https://github.com/inclusionAI/AWorld-RL/tree/main/EnvTuning/data): 100 rows from [`bfcl_val.parquet`](https://github.com/inclusionAI/AWorld-RL/blob/main/EnvTuning/data/bfcl_val.parquet) and 300 rows from [`bfcl_test.parquet`](https://github.com/inclusionAI/AWorld-RL/blob/main/EnvTuning/data/bfcl_test.parquet). It contains 100 rows each of Base, Long Context, Missing Function, and Missing Parameter.

### Stage 1

The selected Stage 1 update-25 checkpoint and retained intermediate evaluations are summarized below. The cross-stage fixed-denominator comparison remains visible in [Stage 1 to Stage 2 Improvement](#stage-1-to-stage-2-improvement).

<details>
<summary><b>Stage 1 retained-checkpoint and eval-400 diagnostics</b></summary>

#### Stage 1 retained checkpoint validation

| Update | Epoch | Stage 1 score | Observed progress | Format | Tool-call execution | Tool-use rate | Mean rounds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 2 | 1.7122 | 0.3631 | 0.8646 | 0.8622 | 0.8700 | 10.95 |
| 15 | 3 | 1.6986 | 0.3495 | 0.8519 | 0.8629 | 0.8700 | 11.79 |
| 20 | 4 | 1.7463 | 0.3525 | 0.8719 | 0.8777 | 0.8900 | 11.11 |
| 25 | 5 | 1.7428 | 0.3489 | 0.8788 | 0.8740 | 0.8900 | 11.51 |

Because this validation split is balanced, its per-category Stage 1 score / observed-progress pairs are:

| Update | Base | Long Context | Missing Function | Missing Parameter |
|---:|---:|---:|---:|---:|
| 10 | 1.8046 / 0.4293 | 1.7299 / 0.3547 | 1.7235 / 0.4221 | 1.5908 / 0.2464 |
| 15 | 1.8702 / 0.3813 | 1.8785 / 0.3993 | 1.4750 / 0.2972 | 1.5706 / 0.3200 |
| 20 | 1.8163 / 0.4127 | 1.9040 / 0.3513 | 1.5721 / 0.3310 | 1.6930 / 0.3150 |
| 25 | 1.8649 / 0.4873 | 1.8092 / 0.3460 | 1.6457 / 0.2993 | 1.6512 / 0.2630 |

#### Update 25 on eval-400 under the Stage 1 reward

| Split | Rows | Stage 1 score | Format | Tool execution | Tool-use rate | Observed progress | Final-answer closure† | Rounds | Parser ≥2 actions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 100 | 1.7601 | 0.8633 | 0.9047 | 0.9200 | 0.4368 | 0.7300 | 11.61 | 0.9231 |
| Long Context | 100 | 1.7292 | 0.8511 | 0.8798 | 0.9000 | 0.4177 | 0.5400 | 10.63 | 0.9158 |
| Missing Function | 100 | 1.7584 | 0.8627 | 0.9057 | 0.9200 | 0.3889 | 0.7400 | 13.07 | 0.9140 |
| Missing Parameter | 100 | 1.5550 | 0.7646 | 0.7972 | 0.8100 | 0.3499 | 0.7300 | 10.36 | 0.9111 |
| **Overall** | **400** | **1.7007** | **0.8354** | **0.8718** | **0.8875** | **0.3983** | **0.6850** | **11.42** | **0.9161** |

`†` The legacy Stage 1 artifact named this field `terminal_coverage`, but its evaluator tests whether a transcript ends in a valid non-tool answer. It is documented as **final-answer closure**, not fixed expected-turn coverage.

The pooled Stage 1 action-parser rates are `0.9841` at action 1, `0.8939` at action 2, and `0.9182` at action 3 or later. With 2,000 sample-level bootstrap resamples (seed 42), overall 95% intervals are: score `[1.6398, 1.7611]`, format `[0.8042, 0.8650]`, tool execution `[0.8417, 0.9027]`, tool-use rate `[0.8575, 0.9175]`, and observed progress `[0.3628, 0.4330]`. Truncation and early-termination rates are both `0`.

</details>

### Stage 2

The selected Stage 2 update-25 checkpoint is initialized from Stage 1 update 25. Detailed retained-checkpoint and eval-400 diagnostics are preserved below.

<details>
<summary><b>Stage 2 retained-checkpoint and eval-400 diagnostics</b></summary>

#### Stage 2 retained checkpoint validation

The Stage 2 wrapper directly records $R_P$, coverage, incompleteness, expected turns, missing terminals, and event counts. Metrics marked `†` are deterministic post-derivations from saved codes using the Stage 1 formulas; they did not alter training reward.

| Update | $R_P$ | Terminal coverage | Incomplete rate | Missing turns | Format† | Tool execution† | Tool-use† | Rounds† |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.4463 | 0.9195 | 0.1100 | 0.25 | 0.8801 | 0.9364 | 0.9500 | 11.66 |
| 10 | 0.4668 | 0.8903 | 0.1200 | 0.35 | 0.8364 | 0.8910 | 0.9100 | 10.40 |
| 15 | 0.5267 | 0.8900 | 0.1200 | 0.33 | 0.8319 | 0.8850 | 0.9000 | 10.06 |
| 20 | 0.5590 | 0.9700 | 0.0300 | 0.10 | 0.8718 | 0.9466 | 0.9600 | 10.90 |
| 25 | 0.5565 | 0.9200 | 0.0800 | 0.27 | 0.8629 | 0.9018 | 0.9100 | 10.03 |

The mean expected-turn count is `3.68` throughout. Mean event counts per trajectory are:

| Update | `n(-3)` | `n(-2)` | `n(-1)` | `n(0)` | `n(1)` |
|---:|---:|---:|---:|---:|---:|
| 5 | 1.04 | 0.08 | 7.11 | 1.70 | 1.73 |
| 10 | 0.96 | 0.10 | 6.01 | 1.56 | 1.77 |
| 15 | 0.97 | 0.08 | 5.66 | 1.34 | 2.01 |
| 20 | 1.32 | 0.10 | 5.90 | 1.48 | 2.10 |
| 25 | 0.79 | 0.04 | 5.79 | 1.26 | 2.15 |

Update 20 has the highest retained Base-100 $R_P$; update 25 is the terminal selected checkpoint used for eval-400 and Stage 3 initialization. Intermediate validation JSON does not imply that every intermediate weight checkpoint was retained.

#### Update 25 detailed eval-400 diagnostics

| Split | $R_P$ | Coverage | Incomplete | Format† | Tool execution† | Tool-use† | Rounds | Parser ≥2 actions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 0.6027 | 0.9575 | 0.0500 | 0.8960 | 0.9547 | 0.9600 | 10.43 | 0.9374 |
| Long Context | 0.3952 | 0.7730 | 0.2600 | 0.8577 | 0.9170 | 0.9300 | 8.62 | 0.8737 |
| Missing Function | 0.4515 | 0.8868 | 0.1300 | 0.8582 | 0.9195 | 0.9300 | 10.52 | 0.9127 |
| Missing Parameter | 0.3774 | 0.8781 | 0.1400 | 0.8207 | 0.8786 | 0.8900 | 10.41 | 0.9279 |
| **Overall** | **0.4567** | **0.8739** | **0.1450** | **0.8582** | **0.9174** | **0.9275** | **10.00** | **0.9149** |

`†` Format/tool metrics are post-derived diagnostics and do not enter $R_P$. Complete mean event counts are:

| Split | Expected turns | `n(-3)` | `n(-2)` | `n(-1)` | `n(0)` | `n(1)` |
|---|---:|---:|---:|---:|---:|---:|
| Base | 3.68 | 0.86 | 0.02 | 6.02 | 1.24 | 2.29 |
| Long Context | 3.68 | 1.04 | 0.07 | 4.72 | 1.35 | 1.44 |
| Missing Function | 4.68 | 1.00 | 0.06 | 5.23 | 2.01 | 2.22 |
| Missing Parameter | 4.68 | 1.14 | 0.08 | 5.05 | 2.30 | 1.84 |
| **Overall** | **4.18** | **1.0100** | **0.0575** | **5.2550** | **1.7250** | **1.9475** |

The pooled Stage 2 action-parser rates are `0.9816` at action 1, `0.9030` at action 2, `0.9163` at action 3 or later, and `0.9149` over all actions at position 2 or later.

</details>

### Stage 1 to Stage 2 Improvement

To compare task progress on one scale, Stage 1 update 25 was re-evaluated with the exact Stage 2 fixed-denominator wrapper:

| Model | Overall $R_P$ | Base | Long Context | Missing Function | Missing Parameter |
|---|---:|---:|---:|---:|---:|
| Stage 1 update 25 | 0.3728 | 0.4457 | 0.3429 | 0.3725 | 0.3302 |
| **Stage 2 update 25** | **0.4567** | **0.6027** | **0.3952** | **0.4515** | **0.3774** |

The paired overall improvement is `+0.0839`, with a paired 95% bootstrap interval of `[+0.0536, +0.1146]` over the same 400 sample IDs.

### Stage 3 Model Evaluation

The final ToolWeave Stage 3 checkpoint was evaluated on the canonical balanced 400-row held-in set: 100 entries each from Base, Missing Function, Missing Parameter, and Long Context. These values are complete-entry BFCL Multi-Turn accuracies, not the training-time Progress Reward $R_P$.

| Model | Overall | Base | Missing Function | Missing Parameter | Long Context | Correct entries |
|---|---:|---:|---:|---:|---:|---:|
| **ToolWeave Stage 3** | **48.50** | **56.00** | **50.00** | **42.00** | **46.00** | **194 / 400** |

Because the four categories are balanced, the overall score is both their unweighted mean and the complete-entry accuracy over all 400 entries:

$$
\frac{56.00+50.00+42.00+46.00}{4}
=48.50.
$$

### Stage 3 Implementation Validation

The runtime-interaction credit implementation passes the complete public CPU suite, deterministic parser/provenance and trainer tensor-contract checks, and the real K=16 formal-training replay documented above. These checks validate implementation and integration.

<details>
<summary>Audited Stage 3 formal-training configuration</summary>

The current portable formal launch contract is the layered [`stage3_reference.yaml`](stage1_format_rl/configs/layers/profiles/stage3_reference.yaml) profile. The public [monolithic configuration](stage1_format_rl/configs/stage3_rods_matchtir_v1_training_branch.yaml) is retained only as a compatibility/reference artifact.

| Setting | Stage 3 |
|---|---|
| Starting model | Stage 2 update 25 |
| Original training pool | 400 BFCL rows (100 per type) |
| Prompt groups / update | 20 |
| Rollouts / prompt | 16 |
| Learning rate | `1e-6` |
| PPO epochs | 1 |
| Local discount $\gamma$ | `0.9` |
| Local weight $\lambda_{\mathrm{local}}$ | `1.0` |
| Matching | True maximum-weight Hungarian (`hard`) |
| Unmatched penalty | `0.0` |
| Minimum peer support | 2 |
| PPO clip low / high | `0.20 / 0.28` |
| Dual clip | `10` |
| Loss-side KL | `low_var_kl`, coefficient `0.01` |
| Boundary low / high | `0.20 / 0.85` |
| Selection budget $M$ | 16 |
| Per-type quota $M_{\tau}$ | 4 per BFCL type |
| Sample-identity cooldown $c$ | 13 steps |
| Next-epoch admission | Yes |
| Generated-pool cap | 400 |

</details>

<details>
<summary>Audited Stage 1/2 training configuration</summary>

| Setting | Stage 1 | Stage 2 |
|---|---|---|
| Starting model | `Qwen/Qwen3-4B` | merged Stage 1 update 25 |
| RL training rows | 100 Base rows | the same 100 Base rows |
| Prompt groups per update | 20 | 20 |
| Rollouts per prompt | 16 | 16 |
| Trajectories per optimizer update | 320 | 320 |
| Prompt / response limit | 8,192 / 10,000 tokens | 8,192 / 10,000 tokens |
| Optimizer | AdamW through veRL, learning rate `1e-6`, warmup ratio `0.03` | same |
| PPO epochs | 1 | 1 |
| PPO mini-batch | 20 pre-expansion prompt groups | same |
| PPO micro-batch | 2 per GPU with dynamic token batching | same |
| Parallelism | 2 GPUs, Ulysses sequence parallel size 2 | same |
| Gradient checkpointing | enabled | enabled |
| FSDP residency | actor resident; optimizer and reference offloaded | same |
| PPO clipping | low `0.20`, high `0.28`, dual-clip `10` | same |
| Entropy coefficient | `0.001` | `0.001` |
| KL path | reward-side adaptive KL: initial `0.1`, target `0.1`, horizon `10,000` | reward-side KL off; loss-side `low_var_kl = 0.01` |
| Selected checkpoint | update 25, epoch 5 | update 25, epoch 5 |

</details>

## Models

| Model | Stage | Description | Status | Link |
|---|---|---|---|---|
| `ToolWeave_stage1` | Stage 1 | Selected merged update-25 checkpoint after the Stage 1 gate | Public selected checkpoint | [Hugging Face](https://huggingface.co/muradil211/ToolWeave_stage1) |
| `ToolWeave_stage2` | Stage 2 | Selected merged update-25 checkpoint initialized from Stage 1 update 25 | Public selected checkpoint | [Hugging Face](https://huggingface.co/muradil211/ToolWeave_stage2) |
| `ToolWeave_stage3` | Stage 3 | Final checkpoint after formal boundary-guided interaction-aware online reinforcement learning | Final ToolWeave Stage 3 model | [Hugging Face](https://huggingface.co/muradil211/ToolWeave_stage3) |

The model links do not imply that the full reproducibility package has already been released. The Stage 3 link points to the completed final ToolWeave Stage 3 model.

## Data

ToolWeave does not rehost the upstream BFCL/EnvTuning datasets; canonical upstream sources are linked below.

**Trajectory anatomy.** See [Data & Trajectory Anatomy](docs/data-and-trajectories.md) for the BFCL sample → user turn → non-answer runtime interaction → parsed-call hierarchy and a real parser-recovery rollout.

| Resource | Composition and role | Source |
|---|---|---|
| Stage 1/2 training | 100 Base interaction rows | [AWorld-RL `bfcl_train_base.parquet`](https://github.com/inclusionAI/AWorld-RL/blob/main/EnvTuning/data/bfcl_train_base.parquet) |
| Stage 3 original pool | 400 rows: 100 per BFCL multi-turn category | [AWorld-RL `bfcl_train.parquet`](https://github.com/inclusionAI/AWorld-RL/blob/main/EnvTuning/data/bfcl_train.parquet) |
| Held-in evaluation | 400 rows: 100-row validation + 300-row test | [`bfcl_val.parquet`](https://github.com/inclusionAI/AWorld-RL/blob/main/EnvTuning/data/bfcl_val.parquet) + [`bfcl_test.parquet`](https://github.com/inclusionAI/AWorld-RL/blob/main/EnvTuning/data/bfcl_test.parquet) |
| Original benchmark source | BFCL V3 Multi-Turn | [BFCL dataset](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) and [repository data](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard/bfcl_eval/data) |
| Generated Stage 3 candidates | Execution- and semantics-validated online replay rows | Formal-training project artifact; separate data release not included here |

These parquet rows provide prompts, tools, environment metadata, and reward-side GT for executable RL interaction; ToolWeave does not describe them as supervised trajectory-imitation data.

RODS describes an 800-row BFCL V3 Multi-Turn in-distribution protocol: 400 training rows (100 per category) and 400 held-in evaluation rows (100 per category). In the public AWorld-RL processed layout, the held-in IDs are the union of the 100-row validation and 300-row test files above. The separate local RODS-style audit subset has 100 rows (25 per category); it is not the canonical Stage 1/2 eval-400 used in the reported comparison.

### Audited local dataset identities

| Stage | Dataset | Composition | SHA256 |
|---|---|---|---|
| Stage 1/2 | `bfcl_stage1_train_base_100_shuffled_seed42.parquet` | 100 Base rows | `d02122551606f616c5d9d6b2915113e8266872078906c58cdefbc97ea198bf5d` |
| Stage 3 | `bfcl_stage3_train_all_400_shuffled_seed42.parquet` | 400 rows, 100 per category | `fee03852fefed510e4022a7f44894518ef0af6790807e35655b0baf9979ef2d6` |

The local Stage 1/2 membership matches upstream `bfcl_train_base.parquet`; the prepared Stage 3 original-pool membership matches upstream `bfcl_train.parquet`. Online generated candidates remain separately provenance-gated.

## Repository Layout

```text
ToolWeave/
├── .env.example
├── README.md
├── ALGORITHM_REPORT_STAGE3_RUNTIME_INTERACTION_CREDIT_FINAL.md
├── assets/                         # Project marks and method figures
├── code/
│   └── AWorld-RL-stage1-worktree/
│       └── EnvTuning/              # Public interaction, credit, trainer, and generator source
├── configs/                        # Historical standalone configuration examples
├── docs/
│   ├── data-and-trajectories.md
│   └── infrastructure-decoupling.md
├── environment/                    # Machine-local configuration templates
├── scripts/                        # Data, evaluation, and audit utilities
└── stage1_format_rl/
    ├── configs/layers/
    │   └── profiles/stage3_reference.yaml
    ├── infrastructure/             # Layer resolver, preflight, and launch CLI
    ├── rewards/
    ├── schemas/
    └── tests/
```

Root [`configs/`](configs/) retains standalone historical examples. The formal portable launch contract lives under [`stage1_format_rl/configs/layers/`](stage1_format_rl/configs/layers/), where experiment, asset, hardware, runtime, and qualification concerns are resolved independently.

## License

ToolWeave currently has no root project-level `LICENSE` file. The repository is source-visible, but no project-wide reuse grant should be inferred from visibility alone. Adapted or vendored upstream components remain subject to their own terms, including [AWorld-RL's MIT license](https://github.com/inclusionAI/AWorld-RL/blob/main/LICENSE), the vendored [veRL Apache-2.0 license](code/AWorld-RL-stage1-worktree/EnvTuning/verl/LICENSE), and [BFCL/Gorilla's Apache-2.0 license](https://github.com/ShishirPatil/gorilla/blob/main/LICENSE). A root ToolWeave license should be added only after the project owner completes the upstream compatibility and notice review.

## Acknowledgements

ToolWeave builds on and adapts ideas or public infrastructure from the following projects. Their authors and teams are not implied to be contributors to ToolWeave.

- [AWorld-RL](https://github.com/inclusionAI/AWorld-RL)
- [Environment Tuning](https://github.com/inclusionAI/AWorld-RL/tree/main/EnvTuning) and its [paper](https://arxiv.org/abs/2510.10197)
- [RODS](https://github.com/inclusionAI/AWorld-RL/tree/main/RODS) and its [paper](https://arxiv.org/abs/2606.19047)
- [MatchTIR](https://github.com/quchangle1/MatchTIR) and its [paper](https://arxiv.org/abs/2601.10712)
- [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
- [veRL](https://github.com/volcengine/verl)
- [Qwen](https://huggingface.co/Qwen/Qwen3-4B)

## Citation

A formal ToolWeave citation will be added with the public technical report. Please cite the original upstream work when using its concepts or infrastructure.

### RODS

```bibtex
@article{fang2026rods,
  title={RODS: Reward-Driven Online Data Synthesis for Multi-Turn Tool-Use Agents},
  author={Fang, Ruishan and Lu, Siyuan and Zhuang, Chenyi and Lin, Tao},
  journal={arXiv preprint arXiv:2606.19047},
  year={2026}
}
```

### MatchTIR

```bibtex
@article{qu2026matchtir,
  title={MatchTIR: Fine-Grained Supervision for Tool-Integrated Reasoning via Bipartite Matching},
  author={Qu, Changle and Dai, Sunhao and Cai, Hengyi and Xu, Jun and Wang, Shuaiqiang and Yin, Dawei},
  journal={arXiv preprint arXiv:2601.10712},
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
