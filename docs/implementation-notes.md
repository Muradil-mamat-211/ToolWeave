# ToolWeave Implementation Notes

[← Back to ToolWeave README](../README.md)

This document records implementation-facing compatibility and fail-closed contracts for the frozen ToolWeave Stage 3 runtime-interaction credit-assignment algorithm. The defining scientific equations remain in the root README.

## Runtime and Provenance Compatibility

The public method symbols map to runtime provenance as follows:

| Public method identity | Runtime field |
|---|---|
| Prompt $q$ | `uid` |
| BFCL user turn $u$ | `user_turn_id` |
| Non-answer runtime-interaction depth $j$ | `runtime_interaction_index` |

Implementation fields map to this notation as follows: prompt identity `uid` corresponds to $q$; `user_turn_id` to $u$; and `runtime_interaction_index` to $j$. The backward-compatible field `policy_step_id` remains a legacy alias for $j$, while `tool_attempt_index` is diagnostic only. The package path `rods_matchtir_v1` is likewise a legacy module name; the active configuration selects `runtime_interaction_final`. This is the frozen ToolWeave Stage-3 formal-training credit-assignment algorithm.

`policy_step_id` is interpreted as the legacy compatibility alias only where runtime provenance verifies that mapping. Formal temporal depth is always the real non-answer runtime interaction $j$.

A valid final answer/turn closure is excluded from $\mathcal{D}_{i,u}$ and receives global credit only. Every temporally reliable non-answer runtime interaction remains in the local sequence, including parser-rejected or unclassified malformed actions. Multiple calls inside one action are never converted into multiple temporal steps.

The historical `tool_attempt_index` field is retained only as diagnostic metadata; it does not control matching, discount distance, peer grouping, normalization, advantage alignment, or token broadcast. ToolWeave matches at the **individual-call** level, accumulates temporal credit at the **runtime-interaction** level, and applies policy gradients at the **trainable actor-token** level.

At every new BFCL user turn, `runtime_interaction_index` and the local discount accumulator restart from zero. `tool_attempt_index` may also reset for diagnostics, but it has no formal credit-assignment role.

## Advantage Field Compatibility

**Notation compatibility.** Public notation uses $A^g$, $A^{\ell}$, and $A^{TW}$. Existing implementation/artifact fields such as `A_RODS`, `rods_advantages`, `A_local`, and `A_TW` are retained only for backward compatibility.

These fields are internal compatibility identifiers, not alternative public mathematical notation.

## Local-Credit Implementation Invariants

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

## Clipped Policy-Update Implementation Contract

ToolWeave retains the existing GRPO training framework and its PPO-style clipped policy surrogate; the Stage 3 modification enters through the fused advantage $A^{TW}$.

- `epsilon_low = 0.20`, `epsilon_high = 0.28`, and dual-clip constant `10`.
- The log-ratio is clamped to `[-20,20]` before exponentiation.
- The reference term uses `kl_loss_type = low_var_kl` with coefficient `0.01`; reward-side KL is disabled.
- The masked actor loss uses the existing `seq-mean-token-mean` aggregation.
- For GRPO tensor-contract consistency, the implementation mirrors the same token-level local residual into `returns_new`; the actor-only path consumes `advantages`, not a critic return. This does not introduce critic learning.
