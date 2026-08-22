# Layered runtime configuration

The portable launch contract is a profile under `profiles/`. A profile composes
five independent concerns:

| Layer | Owns | Must not own |
|---|---|---|
| `experiment/` | reward, rollout sampling, optimization, lifecycle and dataset roles | paths, GPU IDs, world size, backend placement |
| `assets/` | logical model/data/config IDs, paths, hashes and row counts | training or topology choices |
| `hardware/` | node capacity and physical GPU IDs | Ray, FSDP or algorithm values |
| `runtime/` | role placement, Ray, FSDP, SGLang/vLLM and batching memory knobs | reward, advantage or lifecycle math |
| `qualification/` | exact reference-machine requirements | portable capacity validation |

Machine-local roots and Python executables come from
`environment/machine.template.yaml` plus `TOOLWEAVE_*` environment variables.
Copy `environment/env.template.sh` to ignored `environment/env.local.sh` to
customize a host. Changing the checkout or asset location does not require a
Python or experiment edit.

Examples:

```bash
python -m stage1_format_rl.infrastructure.cli \
  --profile stage1_format_rl/configs/layers/profiles/stage3_reference.yaml \
  resolve

python -m stage1_format_rl.infrastructure.cli \
  --profile stage1_format_rl/configs/layers/profiles/stage3_portable_8gpu.yaml \
  launch
```

`launch` is a dry run unless `--execute` is supplied, and execution remains
protected by the existing explicit training guard. The 8-GPU profile is a
configuration/topology qualification fixture; it is not evidence of an 8-GPU
training run.

The current veRL adapter supports one learner node. `cluster_mode: existing`
can attach to an existing single-node Ray runtime. Any learner assignment that
spans nodes fails closed; no YAML flag can claim unimplemented multi-node
support.

Historical monolithic YAML files under `stage1_format_rl/configs/` are retained
as reference/qualification snapshots. Their machine paths are environment
resolved, but the portable Stage 3 interface is the layered profile system.
