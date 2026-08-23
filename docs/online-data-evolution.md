# Online Data Evolution

This document preserves the complete verified-synthesis pipeline, semantic-hardening contract, and durable lifecycle behavior from the root README.

[← Back to ToolWeave README](../README.md)

## Boundary-Guided Lifecycle

Validated candidates generated in epoch `n` are staged and become eligible only at epoch `n+1`. ToolWeave implements a reproducible adaptation of RODS-style lifecycle management, not a claim of a verbatim unpublished RODS lifecycle. It protects all original rows, limits new admission to at most `floor(0.20 × active_pool_before)`, caps the generated sub-pool at 400, supports trial eviction and drift retirement for generated rows, and persists deferred validated candidates for later epochs.

<details>
<summary>Audited lifecycle constants and edge behavior</summary>

- One observation is required before a generated row leaves its trial state.
- Trial rows below `0.20` can be evicted as too hard.
- Observed generated rows above `0.95` can retire as mastered; the lower retirement boundary is `0.20`.
- Stale retirement is an optional disabled hook because no reproducible paper-default stale window is available.
- If restored state is already above the 400-row generated cap, observed generated rows with the lowest available $\phi$ are pruned first. The implementation does not fabricate priorities for unobserved trial rows.

</details>

## Verified Online Data Synthesis

The Data-Generation Branch is a separate queue consumer. It does not redo boundary selection and does not block the current optimizer step.

### 1. Boundary Seed Selection

The Training Branch validates and emits the selected `rods_boundary_seed.v1` records after a successful update. Each seed retains its original question, GT, available functions, initial configuration, mean $R_P$, priority $\phi$, source epoch/step, and BFCL type.

### 2. Planning and Function Construction

A RODS-derived planner concept proposes an ordered executable structure and latent narrative. Function sampling is constrained to the audited active 128-function catalog. Parameter generation is grounded in the schema, current environment configuration, earlier successful results, and dependency context. Hallucinated or blocked functions fail closed.

### 3. Real BFCL VM Execution

Ground truth is generated and executed before the natural-language query. User turns share state and execution history. Failures enter the generator's [12-class structured taxonomy](../code/AWorld-RL-stage1-worktree/EnvTuning/env_tuning/rods_data_generation_v1/error_taxonomy.py), and only eligible failures trigger deterministic configuration patching, cumulative blocklisting, and feedback-conditioned replanning. A seed is dropped after at most three complete pipeline attempts.

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
