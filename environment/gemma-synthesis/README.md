# Gemma synthesis environment

This directory is the public, machine-independent reconstruction contract for
the isolated Python environment used by ToolWeave's Gemma-4-31B Stage 3
synthesis service. The original local environment was named `rods-synth`;
reproductions use `toolweave-gemma-synthesis` by default.

This environment is deliberately separate from the learner/training
environment. `TOOLWEAVE_SYNTH_PYTHON` selects its absolute Python executable;
it does not download, remotely mount, or implicitly activate an environment.

## Audited contract

| Component | Audited value |
|---|---|
| Platform | Ubuntu 22.04.5, Linux x86_64 |
| Reference GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB |
| Reference driver | 595.58.03 |
| Host `nvcc` | 12.8.93 |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu129 |
| vLLM | 0.26.0 |
| Transformers | 5.14.1 |
| Triton | 3.6.0 |
| FlashInfer | 0.6.14 |
| Humming kernels | 0.1.10 |

`manifest.json` is the machine-readable contract. The 28-record Conda
explicit file fixes the Linux base packages and builds. Two source-specific
Pip locks capture all 202 visible Python distributions: 198 releases from PyPI
and four CUDA packages from the official PyTorch cu129 index. The manifest
records SHA-256 digests and record counts for all three locks.

## Rebuild

Conda on Linux x86_64 is required. The creation script refuses to modify an
existing environment:

```bash
bash environment/gemma-synthesis/create.sh
```

To choose another environment name:

```bash
bash environment/gemma-synthesis/create.sh my-gemma-synthesis
```

The script validates the committed locks and creates the exact Conda base. It
installs the 198-package PyPI snapshot and then the four PyTorch packages from
the official CUDA 12.9 wheel index, all without dependency re-resolution.
Finally, `pip check` validates the installed dependency closure and the verifier
checks the core versions. Expect roughly 11 GiB of installed files.

The equivalent manual sequence is:

```bash
conda create -y -n toolweave-gemma-synthesis \
  --file environment/gemma-synthesis/conda-linux-64.explicit
conda run -n toolweave-gemma-synthesis python -m ensurepip --upgrade
conda run -n toolweave-gemma-synthesis python -m pip install \
  pip==26.2.1 setuptools==80.10.2
conda run -n toolweave-gemma-synthesis python -m pip install --no-deps \
  -r environment/gemma-synthesis/requirements-pypi-linux-x86_64.lock
conda run -n toolweave-gemma-synthesis python -m pip install --no-deps \
  --index-url https://download.pytorch.org/whl/cu129 \
  -r environment/gemma-synthesis/requirements-pytorch-cu129-linux-x86_64.lock
conda run -n toolweave-gemma-synthesis python \
  environment/gemma-synthesis/verify.py
```

## Connect it to ToolWeave

Print the rebuilt environment's absolute interpreter path:

```bash
conda run -n toolweave-gemma-synthesis \
  python -c 'import sys; print(sys.executable)'
```

Copy that output into the ignored `environment/env.local.sh`:

```bash
export TOOLWEAVE_SYNTH_PYTHON=/absolute/path/to/toolweave-gemma-synthesis/bin/python
```

When direct interpreter paths are used, leave `TOOLWEAVE_CONDA_ENV` empty.
Then load the machine configuration and verify GPU visibility:

```bash
source environment/env.local.sh
"$TOOLWEAVE_SYNTH_PYTHON" environment/gemma-synthesis/verify.py --require-gpu
```

The online profile's generator commands are assembled by the project Python
environment but execute vLLM with `TOOLWEAVE_SYNTH_PYTHON`:

```bash
"$TOOLWEAVE_PYTHON" -m stage1_format_rl.infrastructure.cli \
  --profile stage1_format_rl/configs/layers/profiles/stage3_online_2gpu.yaml \
  generator-server

RODS_ALLOW_VLLM_SERVER=1 \
"$TOOLWEAVE_PYTHON" -m stage1_format_rl.infrastructure.cli \
  --profile stage1_format_rl/configs/layers/profiles/stage3_online_2gpu.yaml \
  generator-server --execute
```

The first command is a dry run. Inspect it before using the explicitly guarded
execution command.

## Scope and limits

- Model weights are not stored here. The profile expects the logical
  `gemma_4_31b` asset declared in
  `stage1_format_rl/configs/layers/assets/stage3_online_reference.yaml`.
- The committed Pip files pin versions, but not individual wheel hashes.
  Package indexes must continue to retain those releases.
- The NVIDIA driver is a host responsibility. CUDA libraries supplied by the
  Python packages do not replace a compatible host driver.
- Other GPU types may work, but only the reference Blackwell topology above is
  an audited ToolWeave configuration.
