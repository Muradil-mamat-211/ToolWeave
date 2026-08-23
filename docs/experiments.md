# Experiments and Training Audit

This document preserves the complete Stage 1/2/3 evaluation protocol, retained-checkpoint diagnostics, implementation validation, and audited formal-training configuration from the root README. The expanded runtime-interaction evidence is in the [Credit-Assignment Audit](credit-assignment-audit.md).

[← Back to ToolWeave README](../README.md)

## Experiments and Results

Stage 1/2 results below come from deterministic one-rollout evaluation artifacts and report internal EnvTuning/RODS reward and protocol metrics. The final Stage 3 checkpoint is additionally reported with complete-entry BFCL Multi-Turn accuracy on the balanced 400-row held-in set. Stage 3 implementation validation and the real K=16 formal-training replay are documented separately. None of these results is presented as an official BFCL leaderboard submission.

### Evaluation Protocol

Evaluation uses the [diagnostic-event semantics defined in Stage 1](../README.md#diagnostic-event-semantics). Only codes `0` and `1` are terminal. Additional evaluation diagnostics are

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

#### Audited Local Training-Dataset Identities

| Stage | Dataset | Composition | SHA256 |
|---|---|---|---|
| Stage 1/2 | `bfcl_stage1_train_base_100_shuffled_seed42.parquet` | 100 Base rows | `d02122551606f616c5d9d6b2915113e8266872078906c58cdefbc97ea198bf5d` |
| Stage 3 | `bfcl_stage3_train_all_400_shuffled_seed42.parquet` | 400 rows, 100 per category | `fee03852fefed510e4022a7f44894518ef0af6790807e35655b0baf9979ef2d6` |

The local Stage 1/2 membership matches upstream `bfcl_train_base.parquet`; the prepared Stage 3 original-pool membership matches upstream `bfcl_train.parquet`. Online generated candidates remain separately provenance-gated.

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

The runtime-interaction credit implementation passes the complete public CPU suite, deterministic parser/provenance and trainer tensor-contract checks, and the real K=16 formal-training replay [documented in the credit-assignment audit](credit-assignment-audit.md). These checks validate implementation and integration.

<details>
<summary>Audited Stage 3 formal-training configuration</summary>

The current portable formal launch contract is the layered [`stage3_reference.yaml`](../stage1_format_rl/configs/layers/profiles/stage3_reference.yaml) profile. The public [monolithic configuration](../stage1_format_rl/configs/stage3_rods_matchtir_v1_training_branch.yaml) is retained only as a compatibility/reference artifact.

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
