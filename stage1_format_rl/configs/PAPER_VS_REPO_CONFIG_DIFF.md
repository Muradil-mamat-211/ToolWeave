# Paper Versus Repository Stage 1 Configuration

The user's authoritative setting is `K=16`. References to `K=8` and filenames
containing `k8` in the copied template are stale and were not used.

| Parameter | RODS paper v1 | GitHub Stage 1 YAML | GitHub Stage 1 shell | 2 x 48 GB repo-aligned | 2 x 48 GB paper-aligned |
|---|---|---|---|---|---|
| Model | Qwen3-4B-Instruct | unset | Qwen2.5-7B placeholder | local `Qwen/Qwen3-4B` | same |
| Actor LR | `1e-6` | `1e-6` | no override | `1e-6` | `1e-6` |
| Loss-side KL coefficient | `0.01` | `0.1` | no override | `0.1` | `0.01` |
| Rollouts K | `16` | inherited default `1` | `16` | `16` | `16` |
| Epochs | `5` per stage | inherited base default `30` | `20` | `20` | `5` |
| Prompt batch | not reported | inherited base `1024` | `16` | `4` | `4` |
| PPO mini-batch | `512` trajectories | inherited base `256` | `32` prompt-level setting | `4` prompts / 64 trajectories | `4` prompts / 64 trajectories (scale-down from paper 512) |
| Actor micro-batch/GPU | not reported | `null` | not set | `1` | `1` |
| Entropy coefficient | not reported | `0.001` | no override | `0.001` | `0.001` (`repo_only_unreported_in_paper`) |
| Clip low/high | not reported | `0.2 / 0.28` | no override | `0.2 / 0.28` | same (`repo_only_unreported_in_paper`) |
| Dual-clip C | not reported | `10` | no override | `10` | same (`repo_only_unreported_in_paper`) |
| Max prompt length | not reported | `8192` | no override | `8192` | `8192` (`repo_only_unreported_in_paper`) |
| Max response length | not reported | `10000` | no override | `10000` | `10000` (`repo_only_unreported_in_paper`) |
| Max turns | not reported | `max_turns=100` (unused key) | no override | executable user/assistant limits `100/100` | same |
| Rollout engine | not reported | SGLang | inherited | SGLang | SGLang (`repo_only_unreported_in_paper`) |
| Tensor parallel | not reported | inherited base `2` | `1` | `1` | `1` |
| Max model length | not reported | `32768` | no override | `32768` | `32768` |
| Max batched rollout tokens | not reported | inherited `8192` | `131072` | `131072` | `131072` |
| Loss aggregation | not reported | `seq-mean-token-mean` | no override | same | same (`repo_only_unreported_in_paper`) |
| GPU count | `8 x A100` | unset | `8` | `2` | `2` |
| Gradient checkpointing | not reported | base default true | true | true | true |
| Fused kernels | not reported | base default false | true | true | true |
| Actor parameter offload | not reported | false | no override | false | false |
| Actor optimizer offload | not reported | false | no override | false | false |
| Reference parameter offload | not reported | false | no override | false | false |
| Reward-side KL | not reported | disabled, coefficient 0 | no override | disabled | disabled |
| Reward manager | not reported | `bfcl` | no override | `bfcl` | `bfcl` |
| Multi-turn | required by method | enabled | no override | enabled | enabled |

## Batch Semantics

The adapted FSDP worker normalizes `ppo_mini_batch_size` by multiplying it by
`rollout.n` and dividing by data-parallel world size. With this static profile:

```text
prompt batch = 4
K = 16
global trajectories/update = 64
world size = 2
normalized trajectories/worker = 4 * 16 / 2 = 32
micro-batch/GPU = 1
```

The paper's 512-trajectory mini-batch cannot fit inside a 64-trajectory update.
The paper-aligned file therefore preserves the paper's LR/KL/K/epoch choices
but explicitly marks mini-batch as an intentional hardware scale-down. It is
not a complete 8 x A100 reproduction.

## Static Profile Qualification

Both files are `recommended_static_profile`, not `GPU-validated profile`.
Prompt/response limits are intentionally unchanged from the repository.
FlashAttention is installed, but kernel, memory, throughput, SGLang, and
distributed behavior were not exercised in no-card mode.

