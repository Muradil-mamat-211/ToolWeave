# Custom veRL v0.4.1 Modifications

## Project Overview

This project is based on the official [verl v0.4.1](https://github.com/volcengine/verl/tree/v0.4.1) version with custom modifications to address several framework issues discovered during training. The following sections detail the specific modifications and their explanations. 

To use this package, you don't need to install other dependencies other than the official verl dependencies. You can install the dependencies by running:
```bash
pip install -e '.[sglang]'
```

## Related Research

This is the customized veRL version used in our paper:

**Don't Just Fine-tune the Agent, Tune the Environment**

*Siyuan Lu, Zechuan Wang, Hongxuan Zhang, Qintong Wu, Leilei Gan, Chenyi Zhuang, Jinjie Gu, Tao Lin*

[![arXiv](https://img.shields.io/badge/arXiv-2510.10197-b31b1b?logo=arXiv)](https://arxiv.org/abs/2510.10197) 
[![Paper](https://img.shields.io/badge/Hugging%20Face-Paper-yellow?logo=huggingface)](https://huggingface.co/papers/2510.10197)


### Citation

```bibtex
@article{lu2025dont,
  title={Don't Just Fine-tune the Agent, Tune the Environment},
  author={Lu, Siyuan and Wang, Zechuan and Zhang, Hongxuan and Wu, Qintong and Gan, Leilei and Zhuang, Chenyi and Gu, Jinjie and Lin, Tao},
  journal={arXiv preprint arXiv:2510.10197},
  year={2025}
}
```

## Modifications

### 1. Fix Out-of-Memory (OOM) Issue After First Training Step

**Problem Description:**
Training encounters GPU memory overflow after the first step when using FSDP + SGLang hybrid training mode.

**Modified File:** `verl/workers/sharding_manager/fsdp_sglang.py`

**Problem Analysis:**
- GPU memory was not effectively released during sharding manager enter/exit phases
- Overlapping memory occupation between model weight synchronization and inference engine management

**Solution:**
- Added `get_torch_device().empty_cache()` calls at critical checkpoints
- Optimized memory occupation timing management to ensure timely memory release before and after model weight updates
- Improved memory management strategy in multi-stage wake-up mechanism

**Reference:** https://github.com/volcengine/verl/issues/2445

### 2. Fix Training Timeout Issue

**Problem Description:**
During distributed training, timeout interruptions occur due to long-tail trajectory phenomena in the rollout phase, where some sequences require extended generation time.

**Modified File:** `verl/workers/fsdp_workers.py`

**Problem Analysis:**
- Default distributed training timeout setting is too short
- Long-tail trajectories in rollout phase require extended processing time
- Insufficient consideration for large model inference time overhead

**Solution:**
- Adjusted distributed initialization timeout from default to `timeout=datetime.timedelta(days=1)`
- Ensures training process won't be interrupted by normal long inference operations
- Provides sufficient time buffer for multi-turn conversations and complex reasoning tasks

**Modification Location:** Line 110
```python
torch.distributed.init_process_group(
    backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}", 
    rank=rank, 
    world_size=world_size, 
    timeout=datetime.timedelta(days=1),  # Extended timeout
    init_method=os.environ.get("DIST_INIT_METHOD", None)
)
```

### 3. Schema Code Adaptation for LLaMA 3.1 Series Models

**Problem Description:**
When using LLaMA 3.1 series models, tool calling related template parsing encounters issues, causing training data processing anomalies.

**Modified File:** `verl/workers/rollout/schemas.py`

**Problem Analysis:**
- LLaMA 3.1 series models have stricter JSON format requirements for tool calling
- Original data serialization methods may include None values, affecting template parsing
- Tool calling information in multi-turn conversations requires more precise handling

**Solution:**
- Use `exclude_none=True` parameter during message data serialization
- Ensure clean data structure passed to chat template without null values
- Improved compatibility with LLaMA 3.1 series models

**Modification Location:** Line 327
```python
# Before
messages = [msg.model_dump() for msg in self.messages]

# After  
messages = [msg.model_dump(exclude_none=True) for msg in self.messages]
```

### 4. Add BFCL Reward Manager
**Problem Description:**

**Added File:** `verl/workers/reward_manager/bfcl_reward_manager.py`

Used for our fine-grained reward computation.
