# Undecoupled Original Source Branch

This file marks the `undecoupled-original` branch as ToolWeave's historical, machine-bound source layout.

## Scope

- The topology/path layer is based on the isolated pre-decoupling snapshot at local source commit `b746aa7` (which restores the vendored veRL sources on top of snapshot `cbabf00`).
- The Stage-3 local-credit functions are synchronized with the final undeccoupled source commits `b309111` and vendored veRL commit `59dce3f`.
- The active local-credit mode is `runtime_interaction_final`: non-answer runtime interaction `j` is the temporal axis; `tool_attempt_index` is diagnostic only.
- The fixed-denominator global RODS reward, matching mathematics, PPO/GRPO objective, gamma, and local fusion remain unchanged.

## Important distinction

This branch intentionally retains legacy absolute workspace paths and direct machine/topology assumptions. It is published for source provenance and comparison, not as the default portable recipe. The supported infrastructure-decoupled source is the [`main` branch](https://github.com/Muradil-mamat-211/ToolWeave/tree/main).

No model weights, checkpoints, runtime artifacts, datasets, tokens, keys, or production credentials are included in this branch.
