# Credit-Assignment Audit

This document preserves the complete deterministic K=16 formal-training runtime-interaction audit and the associated Stage 3 implementation map from the root README.

[← Back to ToolWeave README](../README.md)

## Formal-Training Credit-Assignment Audit

### Real Rollout Case Study: Interaction-Level Credit Assignment

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

The replay below applies exactly the reward, runtime-depth return, ragged peer normalization, and additive fusion defined in [Sections 3.1–3.3](../README.md#31-reward-modeling).

#### Why Dual-Level Credit Assignment?

A trajectory-only global estimator assigns the same signal to every interaction in one rollout:

$$
A_{i,u,j}^{\mathrm{trajectory}}=A_i^g.
$$

It can distinguish which rollout was better overall, but it cannot distinguish a strong interaction from a weak interaction inside that same rollout. ToolWeave adds the interaction-level residual:

$$
A_{i,u,j}^{TW}
=A_i^g+A_{i,u,j}^{\ell}.
$$

This permits $A_{i,u,j_0}^{TW}\ne A_{i,u,j_1}^{TW}$ and can even produce opposite signs at two runtime depths within one globally successful trajectory.

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
| Runtime depth $j$ | Runtime outcome | Parsed calls | Call rewards | $r_j$ | $R_j$ | Peer support | Peer mean $R$ | Peer sample std $R$† | Local $A^{\ell}$ | Global $A^g$ | Fused $A^{TW}$ |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | Parse error | — | — | 0.000000 | 0.590490 | 16 | 1.813468 | 0.326664 | -3.7438 | -0.4967 | -4.2405 |
| 1 | Parse error | — | — | 0.000000 | 0.656100 | 16 | 0.973298 | 0.087103 | -3.6416 | -0.4967 | -4.1383 |
| 2 | Parse error | — | — | 0.000000 | 0.729000 | 1 | 0.729000 | 0.000000 | 0.0000 | -0.4967 | -0.4967 |
| 3 | Parse error | — | — | 0.000000 | 0.810000 | 1 | 0.810000 | 0.000000 | 0.0000 | -0.4967 | -0.4967 |
| 4 | Parse error | — | — | 0.000000 | 0.900000 | 1 | 0.900000 | 0.000000 | 0.0000 | -0.4967 | -0.4967 |
| 5 | Valid two-call action | `ticket_login`, `create_ticket` | `[1.000000, 1.000000]` | 1.000000 | 1.000000 | 1 | 1.000000 | 0.000000 | 0.0000 | -0.4967 | -0.4967 |
<!-- TOOLWEAVE_CASE_STUDY_CORE_TABLE_END -->

`†` For peer support below two, the unbiased sample standard deviation is mathematically undefined. The production diagnostic records `0.000000` as a sentinel, and the estimator abstains with $A^{\ell}=0$.

The special rollout closes four of five expected BFCL user turns, so the source-of-truth fixed-denominator wrapper gives $R_P=4/5=0.8$. Across the 16-rollout group, the recomputed Progress Rewards have mean `0.925000` and unbiased sample standard deviation `0.251661`, producing $A^g=-0.496698$ for this rollout.

ToolWeave abstains from local relative credit when no same-runtime-depth peer exists; the global advantage $A^g$ remains active. Thus the late self-correction is not assigned fabricated singleton credit, while the earlier inefficient/error interactions at $j=0$ and $j=1$ are sharply distinguished from their peers.

> **Scope of the peer summary.** Runtime patterns, immediate rewards, and discounted returns below describe **User Turn 3 only**. Full-rollout $R_P$ and $A^g$ are trajectory-level quantities computed over **all five BFCL user turns**. A rollout can therefore have locally perfect User Turn 3 calls yet have $R_P=0$ when none of its five user turns receives terminal success. Here $A^g$ is the same-prompt, $K=16$ normalized global advantage derived from full-rollout $R_P$, not a User Turn 3 local advantage.

<!-- TOOLWEAVE_CASE_STUDY_K16_TABLE_BEGIN -->
| Offset | Runtime-interaction pattern *(User Turn 3)* | Immediate rewards *(User Turn 3)* | Discounted returns *(User Turn 3)* | Full-rollout $R_P$ *(5 turns)* | Full-rollout global $A^g$ |
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

#### Full K=16 User Turn 3 Interaction Audit

The following table is generated directly by deterministic replay through the production implementation. It expands the 15 non-special peer rollouts into one row per actual User Turn 3 non-answer runtime interaction; the special offset 9 is intentionally not repeated here because its six interactions are shown above. The complete 36-row K=16 artifact, including offset 9, is available as [`user_turn3_k16_full_interaction_advantage.json`](https://huggingface.co/datasets/muradil211/ToolWeave-BFCL-Rollout-Case-Study/blob/main/analysis/user_turn3_k16_full_interaction_advantage.json) and [`CSV`](https://huggingface.co/datasets/muradil211/ToolWeave-BFCL-Rollout-Case-Study/blob/main/analysis/user_turn3_k16_full_interaction_advantage.csv).

<details>
<summary><b>Full K=16 interaction-level audit (30 peer rows)</b></summary>

<!-- TOOLWEAVE_CASE_STUDY_FULL_INTERACTION_TABLE_BEGIN -->
| Offset | $j$ | Runtime outcome | Parsed calls | Call rewards | $r_j$ | $R_j$ | Peer support | Peer mean $R$ | Peer sample std $R$† | Local $A^{\ell}$ | Global $A^g$ | Fused $A^{TW}$ |
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

#### Four Credit-Assignment Regimes in One Real K=16 Group

| Observed behavior | Global $A^g$ | Local $A^{\ell}$ | Fused $A^{TW}$ | Credit-assignment effect |
|---|:---:|:---:|:---:|---|
| Efficient + globally successful | + | + | stronger + | Strengthens efficient, correct interactions |
| Globally successful + locally weaker | + | − | can become − | Can suppress a weaker interaction despite positive trajectory credit |
| Locally correct + globally failed | − | + | still − here, but softened | Preserves local evidence while whole-trajectory consistency remains decisive |
| Repeated failure + delayed recovery | − | early strong − | early stronger negative | Suppresses supported early errors while preserving the final correction's immediate reward |

The four rows are all observed in this group:

- **Offset 0 — efficient and globally successful.** Both User Turn 3 calls match perfectly, and positive local advantages $A^{\ell}$ align with positive global advantage $A^g$. The fused advantage $A^{TW}$ is therefore stronger at both runtime depths.

- **Offset 14 — globally successful but locally weaker.** The second `create_ticket` call scores `0.916667`. Although $A^g=+0.298019$, its local advantage is $A^{\ell}=-0.650158$, flipping the fused interaction advantage to $A^{TW}=-0.352139$. Trajectory-only supervision would instead assign the same positive global value to both interactions.

- **Offset 2 — locally correct but globally failed.** `ticket_login` and `create_ticket` each match at `1.000000`, producing positive local advantages $A^{\ell}$. The full rollout nevertheless has $R_P=0$ and $A^g=-3.675562$, so both fused advantages $A^{TW}$ remain negative, but are softened by the correct local evidence.

  The runtime replay identifies the cause precisely. In User Turn 0, the model emitted only `get_flight_cost` with `travel_from='SAN'`, while the GT uses `travel_from='SFO'` and also requires `book_flight`. At the User Turn 3 terminal check, `state_checker` reports `multi_turn:instance_state_mismatch` in `booking_record` and `credit_card_list`: the model has an empty booking record and card balance `6000`, whereas the GT state contains booking `3426812` and balance `5000`. The TicketAPI state itself matches. Thus these User Turn 3 actions are locally correct, not locally bad; the negative fused value comes from the stateful global branch.

- **Offset 9 — repeated parser failure with delayed recovery.** Parser failures occupy five real runtime depths at $r=0$. The first two depths have supported negative local advantages $A^{\ell}$; later singleton depths abstain. The final valid two-call action retains $r=1$, and its negative fused advantage $A^{TW}$ comes only from $A^g=-0.496698$, not a fabricated local penalty.

This real K=16 formal-training group shows strictly finer interaction-level credit resolution than trajectory-only supervision: the estimator can strengthen efficient interactions, suppress a relatively weaker interaction inside a successful rollout, preserve positive local evidence inside a globally failed rollout, and distinguish direct execution from delayed recovery. This is evidence about credit-assignment behavior, not a controlled claim of superior final-policy performance. Full arguments, runtime messages, parser provenance, rollout identities, exact floating-point values, and the complete trajectory are available in the [dataset record](https://huggingface.co/datasets/muradil211/ToolWeave-BFCL-Rollout-Case-Study/blob/main/data/multi_turn_base_156_rollout_offset_9.json), [K=16 summary](https://huggingface.co/datasets/muradil211/ToolWeave-BFCL-Rollout-Case-Study/blob/main/analysis/user_turn3_k16_credit_summary.json), and [full interaction audit](https://huggingface.co/datasets/muradil211/ToolWeave-BFCL-Rollout-Case-Study/blob/main/analysis/user_turn3_k16_full_interaction_advantage.json).

> **Matching provenance.** The MatchTIR paper describes maximum-weight Hungarian/KM assignment. At audited public commit [`975c453`](https://github.com/quchangle1/MatchTIR/commit/975c4535fbb86a49f21ff7d291a1fa822f827684), the helper named `hungarian_assignment` performs greedy sorted-edge matching. ToolWeave's implemented solver uses the paper-style one-to-one objective with SciPy's true `linear_sum_assignment(..., maximize=True)`.

## ToolWeave Stage 3 at a Glance

```text
Input: current policy, active pool, K rollouts per prompt,
       per-user-turn executable GT call structures and environment contract

For each training update:
  1. Sample K stateful BFCL rollouts for each prompt.
  2. Compute fixed-denominator Progress Reward R_P.
  3. Group-normalize R_P to obtain global advantage A^g.
  4. Build each BFCL user turn's ordered non-answer runtime-interaction sequence.
  5. Match successfully parsed calls to GT once per BFCL user turn.
  6. Average call rewards inside each runtime interaction; unparsed interactions receive r=0.
  7. Compute discounted local returns over real runtime depth within each BFCL user turn.
  8. Normalize over ragged (prompt, user turn, runtime depth) peers to obtain local advantage A^ℓ.
  9. Form fused A^TW = A^g + A^ℓ on reliable non-answer actor spans.
 10. Optimize A^TW under the existing GRPO training framework using the inherited PPO-style clipped surrogate.
 11. Select up to M=16 boundary seeds from grouped R_P only (4/type, cooldown c=13).
 12. Synthesize and validate executable candidate trajectories.
 13. Admit eligible candidates from the next epoch onward.
```

<details>
<summary>Audited Stage 3 implementation map and local-credit constants</summary>

| Responsibility | Public implementation source |
|---|---|
| Global GRPO followed by residual fusion | [`ray_trainer.py`](../code/AWorld-RL-stage1-worktree/EnvTuning/verl/verl/trainer/ppo/ray_trainer.py) |
| Parser classification and runtime indexing | [`utils.py`](../code/AWorld-RL-stage1-worktree/EnvTuning/env_tuning/interaction/utils.py), [`response_handler.py`](../code/AWorld-RL-stage1-worktree/EnvTuning/env_tuning/interaction/response_handler.py), and [`new_multi_turn_fc.py`](../code/AWorld-RL-stage1-worktree/EnvTuning/env_tuning/interaction/new_multi_turn_fc.py) |
| Structured rollout provenance and actor-span binding | [`provenance.py`](../code/AWorld-RL-stage1-worktree/EnvTuning/env_tuning/rods_matchtir_v1/provenance.py) and [`schemas.py`](../code/AWorld-RL-stage1-worktree/EnvTuning/verl/verl/workers/rollout/schemas.py) |
| Call similarity and maximum-weight assignment | [`matching.py`](../code/AWorld-RL-stage1-worktree/EnvTuning/env_tuning/rods_matchtir_v1/matching.py) |
| Interaction reward, turn-local return, ragged normalization, token residual, and fusion | [`advantage.py`](../code/AWorld-RL-stage1-worktree/EnvTuning/env_tuning/rods_matchtir_v1/advantage.py) |
| Boundary selection and next-epoch lifecycle | [`lifecycle.py`](../code/AWorld-RL-stage1-worktree/EnvTuning/env_tuning/rods_matchtir_v1/lifecycle.py) |
| Verified synthesis | [`rods_data_generation_v1/`](../code/AWorld-RL-stage1-worktree/EnvTuning/env_tuning/rods_data_generation_v1/) |

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
