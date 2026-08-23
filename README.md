<div align="center">

<img src="assets/toolweave-mark.svg" alt="ToolWeave mark" width="130">

# ToolWeave

**🧵 Boundary-Guided Verified Tool-Use Synthesis and Agentic Reinforcement Learning for Multi-Turn Tool-Calling Agents.**

ToolWeave trains a multi-turn tool-use policy through a three-stage curriculum, then couples policy optimization with asynchronous, execution-verified data evolution around the current policy's capability boundary.

[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Stage1%20%7C%20Stage2%20%7C%20Stage3-yellow)](https://github.com/Muradil-mamat-211/ToolWeave/tree/main#models)
[![Code](https://img.shields.io/badge/GitHub-Code-181717?logo=github&logoColor=white)](https://github.com/Muradil-mamat-211/ToolWeave)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Agentic RL](https://img.shields.io/badge/Agentic-RL-6D28D9)](https://github.com/Muradil-mamat-211/ToolWeave/tree/main#stage-3-toolweave)
[![Tool Calling](https://img.shields.io/badge/Tool-Calling-0EA5E9)](https://github.com/Muradil-mamat-211/ToolWeave/tree/main#overview)
[![BFCL](https://img.shields.io/badge/BFCL-Multi--Turn-F59E0B)](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)

🤗 [ToolWeave Stage 1 Model](https://huggingface.co/muradil211/ToolWeave_stage1) |
🤗 [ToolWeave Stage 2 Model](https://huggingface.co/muradil211/ToolWeave_stage2) |
🤗 [ToolWeave Stage 3 Model](https://huggingface.co/muradil211/ToolWeave_stage3)

</div>

> [!IMPORTANT]
> This `undecoupled-original` branch preserves the original, infrastructure-undecoupled codebase used during ToolWeave training. It is provided for source provenance and inspection.

## Original training source

The code in this branch retains the training-time machine-bound layout, legacy entry points, absolute workspace paths, and hardware/topology assumptions. It may require environment-specific adaptation and is not the recommended portable setup.

For the supported infrastructure-decoupled codebase, complete documentation, configurations, evaluation results, and usage instructions, use the [`main` branch](https://github.com/Muradil-mamat-211/ToolWeave/tree/main).

Model weights, datasets, checkpoints, runtime artifacts, tokens, keys, and credentials are not stored in this branch.
