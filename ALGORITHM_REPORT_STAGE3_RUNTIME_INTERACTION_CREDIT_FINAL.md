# ToolWeave Stage-3 Runtime-Interaction Credit — Frozen Formal Algorithm

## 1. Scope and verdict

This report records the final source-of-truth Stage-3 credit-assignment correction. The implementation uses real non-answer runtime interactions as its local temporal axis. The fixed-denominator RODS Progress Reward, global GRPO normalization, call-similarity mathematics, maximum-weight solver, PPO/GRPO objective, KL semantics, boundary selection, data-generation lifecycle, Stage 1, and Stage 2 are unchanged.

No new formal training was started and no checkpoint was written. Static validation, the complete 319-test decoupled-source suite, deterministic trainer/token tensor checks, and the real formal-training K=16 replay all passed in the no-GPU environment.

The source package remains named `rods_matchtir_v1` for import compatibility. The active formal configuration selects `runtime_interaction_final`.

Public scientific notation uses global advantage $A_i^g$, local advantage $A_{i,u,j}^{\ell}$, and fused ToolWeave advantage $A_{i,u,j}^{TW}$. Backward-compatible implementation and artifact fields—including `A_RODS`, `rods_advantages`, `A_local`, and `A_TW`—retain their existing internal names.

## 2. Previous compressed temporal semantics

The previous implementation stored both `runtime_interaction_index` and `tool_attempt_index`, but constructed its local sequence only from reliably classified tool attempts. It then:

- discounted over that filtered list;
- set `depth=tool_attempt_index`;
- normalized peers by `(uid, user_turn_id, tool_attempt_index)`; and
- omitted malformed non-answer interactions classified as `unknown`.

That behavior was narrower than the frozen formal specification. In particular, removing a real parser-rejected interaction shortened discount distance and could change peer alignment.

## 3. Formal runtime-interaction definition

For rollout `i` and BFCL User Turn `u`, the local sequence is

```text
I[i,u] = (I[i,u,j]) for j in D[i,u]
D[i,u] = {0, ..., J[i,u]-1}.
```

`j` indexes every reliably owned non-answer assistant generation before valid answer/turn closure. Each runtime interaction consists of one assistant generation and its parser/environment handling. It may be:

- a successfully parsed tool action;
- a parser-rejected or malformed action;
- a parsed tool action whose environment execution later fails; or
- another non-answer generation that forms no structured call.

A valid final answer is excluded from `D[i,u]`. `policy_step_id` remains a backward-compatible alias for runtime order. `tool_attempt_index` remains diagnostic metadata only and does not control ordering, discounting, peer support, normalization, alignment, or token broadcast.

## 4. Matching scope

For each runtime interaction, `P[i,u,j]` contains only successfully parsed structured calls. If parsing yields no structured calls, `P[i,u,j]` is empty.

All parsed calls in one rollout/user-turn are duplicate-preserving concatenated once:

```text
P[i,u] = concat-with-multiplicity over j in D[i,u] of P[i,u,j].
```

`P[i,u]` is matched once against the complete BFCL user-turn ground truth `G[u]`. Matching is never repeated per interaction, parser failures do not create synthetic call nodes, and calls are not deduplicated.

## 5. Matching mathematics

The audited production similarity is unchanged. Tool names match case-insensitively; parameter names use the existing Counter/multiset Jaccard; ground-truth-key values use exact structured equality. A name mismatch gives zero. Otherwise:

```text
S(p,g) = S_tn * (S_tn + S_pn + S_pc) / (2 + |N_g|).
```

Unmatched reward remains `0.0`. The hard solver remains:

```python
scipy.optimize.linear_sum_assignment(score_matrix, maximize=True)
```

No greedy approximation was introduced.

## 6. Parser-error and execution-failure semantics

For an unparsed non-answer runtime interaction:

```text
P[i,u,j] = []
r[i,u,j] = 0
```

The interaction remains in the temporal chain and peer domain. In contrast, a successfully parsed structured call remains in semantic matching even if environment execution subsequently fails. Local credit measures call semantics; the global RODS branch measures complete stateful task success.

## 7. Interaction reward

Whole-user-turn assignment rewards are scattered back to each call's original runtime interaction. If one interaction contains parsed calls `p_1...p_m`:

```text
r[i,u,j] = mean(call rewards in P[i,u,j])  when m > 0
r[i,u,j] = 0                               when m = 0.
```

One JSON-array action containing multiple calls remains one temporal interaction. Calls are never expanded into independent timesteps.

## 8. Runtime-depth discounted return

With frozen `gamma=0.9`, the local return is computed on the complete non-answer runtime sequence:

```text
R_local[i,u,j]
  = sum from h=j to J[i,u]-1 of gamma^(h-j) * r[i,u,h]
  = r[i,u,j] + gamma * R_local[i,u,j+1].
```

Zero-reward parser failures consume real discount distance. The accumulator resets at every BFCL User Turn and never crosses into the next user turn.

Regression invariant:

```text
r = [1, 0, 1]  ->  R = [1.81, 0.9, 1]
```

The compressed result `[1.9, 1]` is explicitly rejected by tests.

## 9. Peer-set domain and ragged normalization

The peer set is defined only by actual interaction existence:

```text
S[q,u,j] = {i | j in D[i,u]}.
```

The production grouping key is:

```text
(uid, user_turn_id, runtime_interaction_index).
```

Missing late interactions are absent, not zero-valued samples. For support $n\ge 2$, ToolWeave computes the peer mean and unbiased sample standard deviation, then:

$$
A_{i,u,j}^{\ell}
=
\frac{R_{i,u,j}^{\ell}-\mu_{q,u,j}^{\ell}}
{s_{q,u,j}^{\ell}+10^{-6}}.
$$

Actor-span reliability does not alter peer membership. An unreliable actor span prevents a token write but leaves a temporally reliable interaction in return and normalization calculations.

## 10. Singleton abstention

If support is below two, sample standard deviation is zero, or the standard deviation is non-finite:

$$
A_{i,u,j}^{\ell}=0.
$$

No missing-depth zero padding and no MatchTIR singleton `mean=0, std=1` fallback are used.

## 11. Global RODS branch

The global score remains fixed-denominator Progress Reward:

```text
R_P = terminally successful BFCL user turns / expected BFCL user turns.
```

The unchanged veRL GRPO estimator groups the `K` rollouts by prompt UID and computes the global advantage $A_i^g$ with unbiased sample standard deviation. The local module neither recomputes nor mutates this branch. Boundary selection continues to observe `R_P` only.

## 12. Fusion and tensor contract

With frozen $\lambda_{\mathrm{local}}=1.0$:

$$
A_{i,u,j}^{TW}
=
A_i^g+A_{i,u,j}^{\ell}.
$$

There is no division by two, post-fusion normalization, centering, RMS rescaling, adaptive weighting, or additional clipping. Global-only tokens use $A_i^g$ without a local residual.

The actor-only veRL tensor contract mirrors the residual into returns:

The internal tensor assignments remain `advantages_new = advantages_global + A_local_token` and `returns_new = returns_global + A_local_token`.

This does not introduce critic learning or alter the PPO/GRPO surrogate objective.

## 13. Token broadcast and fail-closed behavior

Each local scalar is broadcast to that runtime interaction's exact trainable assistant span, intersected with the existing actor response/loss mask. Environment, user, observation, and tool tokens cannot receive the residual.

Thus, a parser-error interaction with reliable temporal identity and actor span can receive nonzero local credit on the actual malformed-generation tokens. If temporal identity is reliable but actor span is not, the interaction remains in the temporal/peer computation but receives no token write. If user-turn ownership or runtime ordering is unreliable, the user-turn local branch fails closed.

## 14. Answer and empty-GT semantics

- Valid final answer/turn closure is excluded from the local sequence.
- Answer tokens receive global advantage only.
- ToolWeave does not use MatchTIR's final-answer F1 local reward.
- Empty-GT Missing Function/Missing Parameter user turns receive $A_{i,u,j}^{\ell}=0$ and remain global-only.

## 15. MatchTIR comparison

The audited MatchTIR commit `975c4535fbb86a49f21ff7d291a1fa822f827684` initializes process reward to zero for every assistant interaction, matches only successfully parsed calls, scatters call rewards to the originating interaction, averages multiple calls within an interaction, discounts over real interaction turns, and normalizes by prompt plus turn index.

ToolWeave retains that structural backbone while adapting it to stateful BFCL:

- local scope resets at each BFCL User Turn;
- final answer is excluded from local credit;
- global task reward is fixed-denominator RODS `R_P`;
- true SciPy maximum-weight assignment replaces the public greedy helper;
- unmatched reward is zero;
- singleton/zero-variance groups abstain;
- fusion is additive with no `/2`.

This is not a literal MatchTIR implementation.

## 16. Real K=16 formal-training regression

The audited source contains 512 trajectories and the `multi_turn_base_156` group contains 16 unique rollout IDs. The special rollout is JSONL line 10, `trajectory_index=9`, `rollout_offset=9`, rollout ID `8516d0df-e6fb-4a67-969d-637bfd967e77`.

For User Turn 3, production replay gives:

| Quantity | Value |
|---|---|
| Interaction rewards $r$ | `[0, 0, 0, 0, 0, 1]` |
| Local returns $R^{\ell}$ | `[0.59049, 0.6561, 0.729, 0.81, 0.9, 1]` |
| Peer support | `[16, 16, 1, 1, 1, 1]` |
| Local advantage $A^{\ell}$ | `[-3.7438335935, -3.6416055642, 0, 0, 0, 0]` |
| Global advantage $A^g$ | `-0.4966976345` |
| Fused advantage $A^{TW}$ | `[-4.2405312279, -4.1383031986, -0.4966976345, -0.4966976345, -0.4966976345, -0.4966976345]` |

Offset 0 preserves efficient local/global agreement. Offset 14 reproduces the second-interaction sign flip. Offset 2 reproduces locally correct calls under a strongly negative global stateful outcome. Offset 9 reproduces early parser-error residuals and singleton abstention for late interactions. The offset 2 state checker again verifies the earlier TravelAPI mismatch rather than mislabeling User Turn 3's calls as locally wrong.

## 17. Validation, files changed, and limitations

Validation results:

- static compile/import: PASS;
- all 30 decoupled-source pytest files: 319 passed;
- real K=16 production replay: PASS;
- runtime-depth, ragged support, offsets 0/2/9/14: PASS;
- parser-error token residual: PASS;
- deterministic veRL trainer tensor contract: PASS;
- new training started: no;
- checkpoints written: zero.

Production source changes are limited to the local-credit/parser integration:

- `EnvTuning/env_tuning/interaction/data_models.py`;
- `EnvTuning/env_tuning/interaction/new_multi_turn_fc.py`;
- `EnvTuning/env_tuning/interaction/response_handler.py`;
- `EnvTuning/env_tuning/interaction/utils.py`;
- `EnvTuning/env_tuning/rods_matchtir_v1/advantage.py`;
- `EnvTuning/env_tuning/rods_matchtir_v1/provenance.py`;
- `EnvTuning/env_tuning/rods_matchtir_v1/__init__.py`;
- `EnvTuning/verl/verl/trainer/ppo/ray_trainer.py`; and
- `EnvTuning/verl/verl/workers/rollout/schemas.py`.

Configuration, deterministic replay, and tests were updated outside the source submodule. `matching.py` received provenance wording only; its mathematics is unchanged. Global reward code, veRL global GRPO, trainer objective, lifecycle, generator, and checkpoints were not changed.

Because the current machine has no GPU, no new GPU integration smoke was run. This session instead verified the real trainer tensor path deterministically on CPU and ran the complete pytest suite. The existing formal-training artifact remained read-only.
