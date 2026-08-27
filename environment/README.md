# Environment contracts

ToolWeave separates machine-local configuration from reproducible software
environments:

- `env.template.sh` and `machine.template.yaml` define host-specific paths,
  interpreter entrypoints, endpoints, and writable roots.
- `gemma-synthesis/` records and rebuilds the isolated Gemma-4-31B vLLM
  synthesis environment.

Copy `env.template.sh` to the ignored `env.local.sh` and set absolute paths for
the current host. `TOOLWEAVE_PYTHON` selects the learner/training interpreter;
`TOOLWEAVE_SYNTH_PYTHON` independently selects the Gemma synthesis
interpreter. An absolute interpreter path is sufficient—Conda activation is
not required for subprocesses launched through the infrastructure CLI.

The Gemma contract is version-pinned and has a creation and verification path.
The learner/training stack currently still requires a separately prepared
compatible Python environment; the Gemma synthesis environment must not be
used as a substitute for it.
