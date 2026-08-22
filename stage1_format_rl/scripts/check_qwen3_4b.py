#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from machine_paths import project_roots
from transformers import AutoConfig, AutoTokenizer


ROOTS = project_roots()


def estimate_qwen3_dense_parameters(config) -> int:
    hidden = config.hidden_size
    q_width = config.num_attention_heads * config.head_dim
    kv_width = config.num_key_value_heads * config.head_dim
    attention = hidden * q_width + 2 * hidden * kv_width + q_width * hidden
    mlp = 3 * hidden * config.intermediate_size
    norms = 2 * hidden
    per_layer = attention + mlp + norms
    embeddings = config.vocab_size * hidden
    lm_head = 0 if config.tie_word_embeddings else embeddings
    final_norm = hidden
    return embeddings + config.num_hidden_layers * per_layer + final_norm + lm_head


def inspect_model(model_path: Path) -> dict:
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)

    raw_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    raw_tokenizer = json.loads(
        (model_path / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    readme = (model_path / "README.md").read_text(encoding="utf-8")
    config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )

    rendered_default = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "test"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    rendered_no_thinking = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "test"},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    architecture_ok = (
        config.model_type == "qwen3"
        and "Qwen3ForCausalLM" in getattr(config, "architectures", [])
    )
    card_is_post_trained = "Training Stage: Pretraining & Post-training" in readme
    card_title_is_base = "# Qwen3-4B-Base" in readme
    path_is_base = "base" in model_path.name.lower()
    chat_template = tokenizer.chat_template or ""
    model_identity_ok = (
        architecture_ok
        and card_is_post_trained
        and not card_title_is_base
        and not path_is_base
        and bool(chat_template)
    )
    if not model_identity_ok:
        raise RuntimeError(
            "The local model is not verified as post-trained/instruction-capable Qwen3-4B"
        )

    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    missing_shards = [name for name in shards if not (model_path / name).is_file()]
    if missing_shards:
        raise RuntimeError(f"Missing model shards: {missing_shards}")

    parameter_estimate = estimate_qwen3_dense_parameters(config)
    return {
        "model_path": str(model_path.resolve()),
        "raw_config_name_or_path": raw_config.get("_name_or_path"),
        "auto_config_name_or_path": getattr(config, "_name_or_path", None),
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
        "model_card_identity": "Qwen/Qwen3-4B",
        "model_card_lineage": "Qwen/Qwen3-4B-Base",
        "training_stage": "Pretraining & Post-training",
        "is_base_checkpoint": False,
        "architecture": config.architectures,
        "model_type": config.model_type,
        "parameter_estimate": parameter_estimate,
        "parameter_estimate_billions": parameter_estimate / 1e9,
        "weight_bytes": index.get("metadata", {}).get("total_size"),
        "weight_shards": shards,
        "missing_shards": missing_shards,
        "tokenizer_class": tokenizer.__class__.__name__,
        "chat_template_present": bool(chat_template),
        "chat_template_mentions_enable_thinking": "enable_thinking" in chat_template,
        "default_thinking_mode": True,
        "default_render_tail": rendered_default[-200:],
        "thinking_disabled_render_tail": rendered_no_thinking[-200:],
        "envtuning_protocol": "<think>...</think> plus exactly one of <tool_call> or <answer>",
        "envtuning_should_disable_native_thinking": False,
        "tokenizer_config_has_chat_template": bool(raw_tokenizer.get("chat_template")),
        "identity_check": "PASS",
    }


def write_report(path: Path, report: dict) -> None:
    shards = "\n".join(f"- `{name}`" for name in report["weight_shards"])
    text = f"""# Qwen3-4B Model Report

## Identity

- Actual path: `{report['model_path']}`
- Official model identity: `{report['model_card_identity']}`
- Raw `config.json` `_name_or_path`: `{report['raw_config_name_or_path']}`
- `AutoConfig._name_or_path`: `{report['auto_config_name_or_path']}`
- Model-card lineage: `{report['model_card_lineage']}` (lineage only, not checkpoint identity)
- Training stage: `{report['training_stage']}`
- Base checkpoint: `{report['is_base_checkpoint']}`
- Identity check: **{report['identity_check']}**

The local model card identifies this checkpoint as `Qwen/Qwen3-4B`, with both
pretraining and post-training. Its `base_model` metadata points to
`Qwen/Qwen3-4B-Base`; that field describes ancestry and does not make this local
checkpoint the Base checkpoint.

## Architecture

- Architecture: `{report['architecture']}`
- Model type: `{report['model_type']}`
- Config-only parameter estimate: `{report['parameter_estimate']:,}` ({report['parameter_estimate_billions']:.3f}B)
- Indexed weight bytes: `{report['weight_bytes']:,}`
- Missing shards: `{report['missing_shards']}`

Weight shards:

{shards}

## Tokenizer And Thinking

- Tokenizer: `{report['tokenizer_class']}`
- Chat template present: `{report['chat_template_present']}`
- `tokenizer_config.json` contains chat template: `{report['tokenizer_config_has_chat_template']}`
- Chat template supports `enable_thinking`: `{report['chat_template_mentions_enable_thinking']}`
- Model-card default thinking mode: `{report['default_thinking_mode']}`
- EnvTuning executable protocol: `{report['envtuning_protocol']}`
- Disable Qwen native thinking for Stage 1: `{report['envtuning_should_disable_native_thinking']}`

The current EnvTuning parser requires the generated assistant text itself to
contain `<think>...</think>`. Passing `enable_thinking=False` prepends an empty
thinking block in the input template; that block is not part of generated text
and therefore does not satisfy the parser. The recommended static configuration
keeps the tokenizer's default thinking path enabled.

No model weights were loaded and no GPU was used.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOTS.models_root / "Qwen3-4B",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOTS.reports_root / "qwen3_4b_model_report.md",
    )
    args = parser.parse_args()
    report = inspect_model(args.model_path)
    write_report(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
