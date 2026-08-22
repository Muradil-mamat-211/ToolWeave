# Infrastructure Decoupling Audit

This document records the source-of-truth audit performed against the original
read-only tree before the infrastructure refactor. The isolated import base is
commit `cbabf00`; commit `b746aa7` restores vendored veRL Python sources that a
broad `models/` copy-ignore rule initially omitted. No training or generation
semantics are inferred from a proposed architecture.

## Baseline

- Original read-only CPU test suite: `265 passed`, `0 failed`, `3 warnings`.
- Stage 3 verification launcher: `57 passed`, `0 failed`, `3 warnings`.
- Stage 2 plain-environment preflight: passed for all eight BFCL classes.
- Training launch guards: `3 passed`.
- GPU smoke and formal training: not run during this refactor.

The algorithm-field and source-hash baseline is stored in
`stage1_format_rl/tests/fixtures/stage3_algorithm_golden.json`.

## Coupling map

| Layer | Audited responsibility | Pre-refactor finding | Required boundary |
|---|---|---|---|
| Algorithm | Progress Reward, GRPO advantage, MatchTIR-derived matching/local credit, lifecycle selection, PPO objective | The dedicated reward and `rods_matchtir_v1` modules do not read CUDA, Ray, host paths, or fixed topology. | Keep source and mathematical fields byte/semantic stable. |
| Experiment | Batch/rollout sampling, optimization, reward and lifecycle hyperparameters | Correct values exist, but share YAML files with paths, GPU counts, backend memory knobs, and output directories. | Retain values in an experiment layer and bind infrastructure separately. |
| Assets | Models, datasets, tokenizer-compatible checkpoint, interaction config, schemas | Paths are repeated as host-specific absolute strings; integrity data is scattered through historical manifests. | Use logical asset IDs, one asset manifest, and optional/required checksums. |
| Framework integration | veRL/Hydra field names and Ray resource-pool construction | Project launchers write veRL fields directly; Ray resources are independently reconstructed from `nnodes` and `n_gpus_per_node`. | One adapter translates a validated topology plan into veRL/Hydra fields. |
| Runtime | SGLang/vLLM, FSDP offload/checkpointing, dynamic token budgets, Ray scheduling | Runtime knobs are mixed with algorithms and repeated in shell scripts. | Runtime profile owns backend and performance behavior. |
| Distributed infrastructure | world size, Ray bundles, rollout TP/DP, FSDP rank topology | Each consumer derives a partial topology; placement strategy and CPUs per worker are effectively hard-wired. | `TopologyPlan` is the only project-level derivation source. |
| Hardware | nodes, physical GPUs, memory, CPU, RAM | GPU count is encoded in veRL config and GPU IDs in launchers; CPU counts are copied into shell exports. | Hardware inventory is declarative and generically validated. |
| Machine-local environment | source root, asset roots, Python/Conda, cache/temp/output locations, endpoint/port | Host-specific checkout and Conda paths were embedded in Python, YAML, tests, and shell. | Environment variables or an ignored machine-local file supply these values. |
| Launcher | guard, config resolution, preflight, launch, signals/logging | Launchers also define paths, GPU roles, world size, CPU threads, and experiment overrides. | Thin launcher: load local environment, resolve, preflight, execute. |

## Confirmed anti-patterns

1. `rods_data_generation_v1.config` defines a module-level absolute workspace
   and derives every default path from it.
2. public Stage 1/2/3 YAML profiles embed one server path and duplicate
   `nnodes`/`n_gpus_per_node` rather than consuming a topology plan.
3. shell launchers combine experiment selection with fixed GPU lists, fixed CPU
   thread counts, Conda installation paths, temporary directories, and output
   paths.
4. tests import implementation and data from the original absolute workspace,
   so a copied checkout can pass while not testing its own source.
5. veRL consumes topology in several places (`main_ppo`, trainer validation,
   resource pools), while no project-level object proves those values agree.
6. Ray placement and per-worker CPU bundles are framework defaults rather than
   explicit outputs of a validated project runtime plan.

## Explicit non-couplings

- `matching.py`, `advantage.py`, `provenance.py`, and the Stage 2 Progress
  Reward contain no machine/hardware/runtime inspection.
- Lifecycle paths are injected through `LifecycleConfig`; boundary math,
  quota/cooldown behavior, and admission semantics do not depend on hardware.
- Native veRL reads `RANK`, `WORLD_SIZE`, and `LOCAL_RANK` inside worker code.
  These are legitimate distributed-runtime inputs, not algorithm coupling.
- Paths retained inside historical manifests are provenance records, not active
  runtime defaults.  They are not rewritten because that would falsify audit
  evidence.
- vendored veRL examples and upstream recipes are not ToolWeave launch
  contracts and remain upstream examples.

## Refactor contract

The target dependency direction is:

```text
algorithm
  <- project/framework adapter
  <- runtime plan
  <- TopologyPlan
  <- hardware profile
  <- machine-local environment
```

Only the adapter may translate logical assets and topology into veRL, Ray,
FSDP, SGLang, and vLLM field names.  Reference qualification is an optional
strict check layered after generic validation; it is not part of portable
resource validation.

## Implemented boundary

The refactor implements the contract as these modules:

| Boundary | Implementation |
|---|---|
| Machine | `environment/machine.template.yaml`, ignored `environment/env.local.sh`, and `MachineConfig` |
| Assets | `configs/layers/assets/*.yaml`, `AssetSpec`, checksum/Parquet metadata validation |
| Hardware | `configs/layers/hardware/*.yaml` and `HardwareConfig` |
| Runtime | `configs/layers/runtime/*.yaml` and `RuntimeConfig` |
| Topology | `build_topology_plan()` returning immutable `TopologyPlan` |
| Framework adapters | `integrations/verl.py` and `integrations/generator.py` |
| Qualification | `configs/layers/qualification/*.yaml` and `qualify_reference()` |
| Launch | `infrastructure.cli` plus thin Stage 3 and Generator shell entrypoints |

`TopologyPlan` is the project-level source for learner/FSDP world size, node
count, GPUs per node, role visibility, Ray pool/bundles, rollout TP/DP and
Generator TP/DP. The veRL adapter writes those values into framework-native
Hydra fields. Runtime-configurable Ray placement and per-worker CPU bundle
size are consumed by `ResourcePoolManager` and `RayResourcePool`; they are not
decorative YAML fields.

## Validation modes

- **Portable validation** checks generic resource and topology invariants. It
  does not require a particular GPU model, GPU count, CPU count, RAM size or
  physical GPU numbering.
- **Reference qualification** additionally checks the declared reference GPU
  model/memory, CPU/RAM profile, topology and role mapping.
- **Observed preflight** optionally compares the declared single-node hardware
  against scheduler-visible GPUs, CPU affinity/quota and cgroup-limited RAM.
- In reference mode, observed preflight additionally checks GPU identity,
  nominal memory, CPU count, RAM, and topology against the qualification
  profile. Portable mode deliberately performs only generic capacity checks.

The alternate `8 x 80 GiB / 96 CPU / 512 GiB` profile resolves to learner
world size 8, rollout TP 2, rollout DP 4 and eight Ray GPU bundles without a
GPU-count branch in Python.

## Compatibility and limits

- The audited production adapter is single-node. Multi-node learner placement
  fails closed even if a profile attempts to add an enabling flag.
- `cluster_mode: existing` expresses attachment to an existing Ray runtime,
  but does not imply a multi-node launcher.
- Historical Stage 1/2 and smoke YAML files remain reproducibility snapshots.
  Their host paths are localized; new portable Stage 3 launches use layered
  profiles.
- Vendored veRL examples retain upstream example paths and topology constants.
  They are not ToolWeave launch contracts and were intentionally not rewritten.

## Verification levels

- Unit/config regression: executed (`292 passed`, no failures or skips).
- Layered profile resolution and alternate-topology launcher dry run: executed.
- Existing Stage 1/2/3 CPU tests and preflight paths: executed.
- GPU smoke: not run for this refactor.
- Formal training: not run for this refactor.
