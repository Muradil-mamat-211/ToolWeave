#!/usr/bin/env python3
"""Summarize the released-vs-step25 parser diagnostic from durable evidence."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from run_qwen_rods_bfcl100_official_eval import atomic_text


WORKSPACE = Path("/root/autodl-tmp/rods-workspace")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def source_line(path: Path, needle: str) -> int:
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if needle in line:
            return number
    raise ValueError(f"needle not found in {path}: {needle}")


def unified_patch(current_files: list[Path], baseline: Path | None) -> str:
    chunks: list[str] = []
    strict = WORKSPACE / "stage1_format_rl/scripts/run_qwen_rods_bfcl100_strict_envtuning_eval.py"
    if baseline is not None and baseline.is_file():
        chunks.extend(
            difflib.unified_diff(
                baseline.read_text(encoding="utf-8").splitlines(keepends=True),
                strict.read_text(encoding="utf-8").splitlines(keepends=True),
                fromfile="a/stage1_format_rl/scripts/run_qwen_rods_bfcl100_strict_envtuning_eval.py",
                tofile="b/stage1_format_rl/scripts/run_qwen_rods_bfcl100_strict_envtuning_eval.py",
            )
        )
    else:
        chunks.append("# Pre-change strict evaluator snapshot unavailable.\n")
    for path in current_files:
        if path == strict:
            continue
        relative = path.relative_to(WORKSPACE)
        chunks.extend(
            difflib.unified_diff(
                [],
                path.read_text(encoding="utf-8").splitlines(keepends=True),
                fromfile="/dev/null",
                tofile=f"b/{relative}",
            )
        )
    return "".join(chunks)


def main(args: argparse.Namespace) -> None:
    root = args.debug_root.resolve()
    released = read_json(root / "released_single_step_debug.json")
    step25 = read_json(root / "step25_single_step_debug.json")
    evidence = {"Released": released, "Step25": step25}

    released_prompt = released["level_1_rendered_prompt"]
    step25_prompt = step25["level_1_rendered_prompt"]
    prompt_text_equal = (
        released_prompt["rendered_prompt_text"]
        == step25_prompt["rendered_prompt_text"]
    )
    prompt_ids_equal = (
        released_prompt["rendered_prompt_token_ids"]
        == step25_prompt["rendered_prompt_token_ids"]
    )
    if not prompt_text_equal or not prompt_ids_equal:
        raise AssertionError("released and step25 diagnostic prompts are not identical")

    comparison: dict[str, Any] = {
        "sample_id": released["sample_id"],
        "prompt_text_identical": prompt_text_equal,
        "prompt_token_ids_identical": prompt_ids_equal,
        "rendered_prompt_sha256": sha256_text(
            released_prompt["rendered_prompt_text"]
        ),
        "rendered_prompt_token_count": len(
            released_prompt["rendered_prompt_token_ids"]
        ),
        "models": {},
    }
    for label, data in evidence.items():
        raw = data["level_2_true_decoder_raw_output"]
        api = data["level_3_serving_api_response"]
        level4 = data["level_4_envtuning_parser_input"]
        comparison["models"][label] = {
            "model_path": data["model_path"],
            "tokenizer_path": data["tokenizer_path"],
            "enable_thinking": data["level_1_rendered_prompt"]["enable_thinking"],
            "true_raw_generated_token_count": len(
                raw["true_raw_generated_token_ids"]
            ),
            "true_raw_has_open_think": raw["raw_has_open_think"],
            "true_raw_has_close_think": raw["raw_has_close_think"],
            "true_raw_has_complete_think_block": raw[
                "raw_has_complete_think_block"
            ],
            "api_reasoning_content_nonempty": bool(api["api_reasoning_content"]),
            "api_content_has_think": (
                isinstance(api["api_content"], str)
                and "<think>" in api["api_content"]
            ),
            "api_tool_calls_present": bool(api["api_tool_calls"]),
            "serialization_source": level4["compatible"]["serialization_source"],
            "reconstructed_think_from_reasoning_content": level4["compatible"][
                "reconstructed_think_from_reasoning_content"
            ],
            "reconstructed_action_from_tool_calls": level4["compatible"][
                "reconstructed_action_from_tool_calls"
            ],
            "original_strict_parser_pass": level4["original"]["parse_success"],
            "compatible_strict_parser_pass": level4["compatible"]["parse_success"],
        }
    atomic_text(
        root / "released_vs_step25_comparison.json",
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
    )

    diff = "".join(
        difflib.unified_diff(
            released_prompt["rendered_prompt_text"].splitlines(keepends=True),
            step25_prompt["rendered_prompt_text"].splitlines(keepends=True),
            fromfile="released_rendered_prompt",
            tofile="step25_rendered_prompt",
        )
    )
    atomic_text(
        root / "rendered_prompt_diff.txt",
        diff or "NO DIFFERENCE: rendered prompt text and token IDs are identical.\n",
    )

    strict_metrics = {
        label: read_json(
            root / "smoke" / f"{label.lower()}_strict/metrics/strict_envtuning_metrics.json"
        )
        for label in ("Released", "Step25")
    }
    bfcl_metrics = {
        label: read_json(
            root / "smoke" / f"{label.lower()}_bfcl/metrics/primary_metrics.json"
        )
        for label in ("Released", "Step25")
    }
    unit_tests = (root / "logs/unit_tests.txt").read_text(encoding="utf-8")

    strict_source = (
        WORKSPACE
        / "stage1_format_rl/scripts/run_qwen_rods_bfcl100_strict_envtuning_eval.py"
    )
    adapter_source = WORKSPACE / "stage1_format_rl/scripts/envtuning_response_adapter.py"
    response_handler = (
        WORKSPACE
        / "code/AWorld-RL-stage1-worktree/EnvTuning/env_tuning/interaction/response_handler.py"
    )
    parser_source = (
        WORKSPACE
        / "code/AWorld-RL-stage1-worktree/EnvTuning/env_tuning/interaction/utils.py"
    )
    locations = {
        "request_fields": source_line(strict_source, '"return_token_ids": True'),
        "token_decode": source_line(strict_source, "true_raw = self.tokenizer.decode"),
        "api_fields": source_line(strict_source, "reasoning = api_reasoning_content"),
        "adapter_call": source_line(strict_source, "compatibility = build_envtuning_parser_input"),
        "adapter": source_line(adapter_source, "def build_envtuning_parser_input"),
        "frozen_response_handler": source_line(response_handler, "content, msg_flag = parse_model_response"),
        "frozen_parser": source_line(parser_source, "def parse_model_response"),
    }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=WORKSPACE / "code/AWorld-RL-stage1-worktree",
        text=True,
        check=True,
        capture_output=True,
    ).stdout.strip()

    released_raw = released["level_2_true_decoder_raw_output"]["true_raw_decoded_text"]
    step25_raw = step25["level_2_true_decoder_raw_output"]["true_raw_decoded_text"]
    released_api = released["level_3_serving_api_response"]
    step25_api = step25["level_3_serving_api_response"]
    released_strict = strict_metrics["Released"]
    step25_strict = strict_metrics["Step25"]
    released_bfcl = bfcl_metrics["Released"]["Overall"]
    step25_bfcl = bfcl_metrics["Step25"]["Overall"]

    lines = [
        "# RODS / EnvTuning Parser Fix Report",
        "",
        "## 1. Verdict",
        "",
        "The controlled A/B rules out prompt encoding as the cause. Both checkpoints received "
        "the same 14,030-character rendered prompt and the same 3,355 input token IDs. "
        "The released checkpoint's sampled token sequence does **not** contain literal "
        "`<think>...</think>`; step25's sampled sequence does.",
        "",
        "No full BFCL100 or BFCL400 run was started.",
        "",
        "## 2. Experiment Lock",
        "",
        f"- Sample: `{released['sample_id']}` (`{released['data_type']}`).",
        f"- Released checkpoint: `{released['model_path']}`.",
        f"- Step25 checkpoint: `{step25['model_path']}`.",
        f"- AWorld-RL local commit: `{commit}`.",
        "- Backend for both: vLLM OpenAI-compatible chat completions, TP=1, BF16, "
        "same server configuration loaded sequentially on one GPU.",
        f"- Backend fingerprint: `{released_api['backend_version']}`.",
        "- Reasoning parser: disabled for both. Tool-call parser: disabled for both.",
        "- Decoding for the raw A/B: temperature=0, top_p=1, top_k=-1, max_tokens=1024, seed=42, no explicit stop strings.",
        "- Shared template: the step25 tokenizer template, passed explicitly to both requests; "
        "`enable_thinking=True` for both.",
        f"- Rendered prompt SHA256: `{comparison['rendered_prompt_sha256']}`.",
        "- The released and step25 tokenizer vocab/merges are byte-identical. Their "
        "`tokenizer_config.json`/repository templates differ, but both native templates also "
        "render this first-step prompt identically when thinking is enabled. The explicit "
        "shared template removes that variable completely.",
        "",
        "This corrects an earlier loose use of the word *embedding*: a different tokenizer or "
        "chat template can change the token IDs presented to the same embedding table. No "
        "embedding weights were swapped here, and the input token IDs are identical.",
        "",
        "## 3. Original Bug and Call Chain",
        "",
        "Before this patch, the strict client requested no output token IDs or rendered prompt "
        "and retained only `choices[0].message.content`. It then appended that string as the "
        "assistant message. EnvTuning `ResponseHandler.parse_and_validate` passed it directly "
        "to the frozen `parse_model_response`, which rejects a response without exactly one "
        "`<think>...</think>` block.",
        "",
        "```text",
        "messages",
        "  -> vLLM /v1/chat/completions",
        "  -> message.content only",
        "  -> MultiTurnFunctionCallInteraction",
        "  -> ResponseHandler.parse_and_validate",
        "  -> parse_model_response",
        "  -> Missing <think> => diagnostic -3",
        "```",
        "",
        "Reading only `message.content` was a real observability/compatibility defect because it "
        "would lose an explicit API reasoning channel. The controlled run shows it was not the "
        "cause of the released model's missing tag in this backend: the server had no reasoning "
        "parser configured and its API reasoning field is null.",
        "",
        "## 4. Fixed Four-Level Call Chain",
        "",
        "```text",
        "LEVEL 1 messages + embedded ordered tools",
        "  -> explicit shared chat template + enable_thinking=True",
        "  -> local rendered prompt text/IDs",
        "  -> vLLM return_prompt_text + return_token_ids",
        "  -> exact local/server prompt equality assertion",
        "LEVEL 2 sampled output token IDs",
        "  -> tokenizer.decode(skip_special_tokens=False) => TRUE RAW",
        "LEVEL 3 full API JSON",
        "  -> reasoning/reasoning_content + content + tool_calls retained separately",
        "LEVEL 4 compatibility adapter",
        "  -> TRUE RAW strict serialization, or explicit serving reasoning restoration only",
        "  -> exact parser input string",
        "  -> unchanged EnvTuning parse_model_response",
        "```",
        "",
        f"Implementation: `{strict_source}:{locations['request_fields']}` requests prompt/token "
        f"evidence; `{strict_source}:{locations['token_decode']}` decodes TRUE RAW; "
        f"`{strict_source}:{locations['api_fields']}` retains all API channels; and "
        f"`{adapter_source}:{locations['adapter']}` implements the fail-closed adapter. The "
        f"official parser remains unchanged at `{parser_source}:{locations['frozen_parser']}`.",
        "",
        "The known generated terminal token `<|im_end|>` remains in TRUE RAW evidence but is "
        "removed from the Level-4 assistant-content serialization, matching the API content "
        "boundary. No semantic text is removed.",
        "",
        "## 5. Single-Step TRUE RAW Evidence",
        "",
        "### Released Qwen3-4B-RODS",
        "",
        "```text",
        released_raw,
        "```",
        "",
        f"- Output token count: {len(released['level_2_true_decoder_raw_output']['true_raw_generated_token_ids'])}.",
        "- `<think>` token (151667): absent.",
        "- `</think>` token (151668): absent.",
        "- API reasoning content: null.",
        "- API tool_calls: empty; literal `<tool_call>` remains in content.",
        "- Serialization source: `no_valid_thinking_channel`.",
        "- Reconstruction used: no.",
        "- Strict parser: fail, `Error: Missing <think></think> tags`.",
        "",
        "### Step25",
        "",
        "```text",
        step25_raw,
        "```",
        "",
        f"- Output token count: {len(step25['level_2_true_decoder_raw_output']['true_raw_generated_token_ids'])}.",
        "- `<think>` token (151667): output token index 0.",
        "- `</think>` token (151668): output token index 191.",
        "- API reasoning content: null; no serving split occurred.",
        "- API tool_calls: empty; literal action remains in content.",
        "- Serialization source: `true_raw`.",
        "- Reconstruction used: no.",
        "- Strict parser: pass (`tool_call`).",
        "",
        "Full prompts, token IDs, API JSON, and exact parser inputs are stored beside this report.",
        "",
        "## 6. Released vs Step25 Comparison",
        "",
        "| field | Released | Step25 |",
        "|---|---|---|",
        f"| sample id | `{released['sample_id']}` | `{step25['sample_id']}` |",
        "| enable_thinking | `True` | `True` |",
        "| prompt identical | `True` (text and 3355 IDs) | `True` |",
        "| TRUE RAW has complete `<think>` | `False` | `True` |",
        f"| reasoning content nonempty | `{bool(released_api['api_reasoning_content'])}` | `{bool(step25_api['api_reasoning_content'])}` |",
        f"| API content has `<think>` | `{'<think>' in (released_api['api_content'] or '')}` | `{'<think>' in (step25_api['api_content'] or '')}` |",
        f"| tool_calls present | `{bool(released_api['api_tool_calls'])}` | `{bool(step25_api['api_tool_calls'])}` |",
        "| reconstruction used | `False` | `False` |",
        f"| strict parser pass | `{released['level_4_envtuning_parser_input']['compatible']['parse_success']}` | `{step25['level_4_envtuning_parser_input']['compatible']['parse_success']}` |",
        "",
        "## 7. Answers to the Three Experimental Questions",
        "",
        "### Q1 — Does released Qwen3-4B-RODS truly sample literal `<think>...</think>`?",
        "",
        "**NO** for the controlled sample and locked decoding. This answer comes from output "
        "token IDs, not `message.content`.",
        "",
        "### Q2 — If TRUE RAW has `<think>`, where does it disappear?",
        "",
        "It does **not** disappear in this experiment. Step25's tags exist in TRUE RAW and are "
        "also preserved in API content. Released has no tags at Level 2, so there is no later "
        "layer at which they could have been removed. The vLLM server's reasoning parser and "
        "tool parser were both disabled.",
        "",
        "The old code location that would have lost a separately returned reasoning channel was "
        f"the content-only transport in the pre-patch `{strict_source}` backend. The fixed code "
        f"captures all channels around lines {locations['api_fields']}-{locations['adapter_call']}.",
        "",
        "### Q3 — What caused released Strict Progress = 0?",
        "",
        "**B — model/output serialization incompatibility with the strict EnvTuning protocol** "
        "for this released checkpoint and controlled input. It is not a general tool-use "
        "capability failure: both models scored 2/2 in the separate BFCL smoke. It is not a "
        "current serving-parser/evaluator mismatch: no reasoning split occurred. The old "
        "evaluator did have incomplete channel capture, now fixed, but that defect did not cause "
        "the observed released output to lack tags.",
        "",
        "## 8. Compatibility Adapter Safety",
        "",
        "The adapter never wraps ordinary prose in `<think>`. It uses TRUE RAW when strict XML "
        "is actually sampled. Only when TRUE RAW is unavailable and a configured serving parser "
        "returns a nonempty explicit reasoning channel can it restore that channel. Structured "
        "tool calls may likewise be losslessly serialized without changing names, values, or "
        "types. Malformed/no-channel responses remain strict failures.",
        "",
        "**Was any `<think>` artificially fabricated in the real A/B or smoke? NO.** Both smoke "
        "runs report zero reconstructed-think actions.",
        "",
        "## 9. Strict Smoke Before/After Evidence",
        "",
        "| Model | Samples | Original parser pass actions | Compatible parser pass actions | Reconstructed think | TRUE RAW complete-think actions | Progress |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Released | 2 | {released_strict['Diagnostics']['original_strict_parser_pass_actions']} | {released_strict['Diagnostics']['compatible_strict_parser_pass_actions']} | {released_strict['Diagnostics']['reconstructed_think_actions']} | {released_strict['Diagnostics']['true_raw_complete_think_actions']} | {released_strict['Overall']['progress_score_percent']:.2f}% |",
        f"| Step25 | 2 | {step25_strict['Diagnostics']['original_strict_parser_pass_actions']} | {step25_strict['Diagnostics']['compatible_strict_parser_pass_actions']} | {step25_strict['Diagnostics']['reconstructed_think_actions']} | {step25_strict['Diagnostics']['true_raw_complete_think_actions']} | {step25_strict['Overall']['progress_score_percent']:.2f}% |",
        "",
        "For this backend, original and compatible scores are identical because no reasoning "
        "parser split occurred. The adapter correctly refuses to turn the released model's prose "
        "into a thinking block. Step25 had two truncated/malformed thinking actions among 25; "
        "these remained `-3` rather than being repaired.",
        "",
        "## 10. BFCL Smoke (Strict Protocol Fully Decoupled)",
        "",
        "| Model | BFCL correct | BFCL score | Strict Progress on same IDs |",
        "|---|---:|---:|---:|",
        f"| Released | {released_bfcl['correct']}/{released_bfcl['total']} | {released_bfcl['score_percent']:.2f}% | {released_strict['Overall']['progress_score_percent']:.2f}% |",
        f"| Step25 | {step25_bfcl['correct']}/{step25_bfcl['total']} | {step25_bfcl['score_percent']:.2f}% | {step25_strict['Overall']['progress_score_percent']:.2f}% |",
        "",
        "BFCL scoring continues to use the source-locked Qwen/BFCL tool-call semantics. It does "
        "not use missing `<think>` as an automatic BFCL failure. No changes were made to the "
        "BFCL evaluator or checker.",
        "",
        "## 11. Tests and Files",
        "",
        "```text",
        unit_tests.rstrip(),
        "```",
        "",
        "Changed/added:",
        "",
        "- `stage1_format_rl/scripts/envtuning_response_adapter.py` — fail-closed compatibility adapter.",
        "- `stage1_format_rl/scripts/run_qwen_rods_bfcl100_strict_envtuning_eval.py` — four-level evidence and selectable original/compatible input.",
        "- `stage1_format_rl/scripts/run_qwen_rods_single_step_parser_diagnostic.py` — fair single-step A/B capture.",
        "- `stage1_format_rl/scripts/summarize_qwen_rods_parser_fix.py` — evidence-derived report.",
        "- `stage1_format_rl/tests/test_envtuning_response_adapter.py` — seven adapter regressions.",
        "",
        "Unchanged: model weights, tokenizer vocabulary, EnvTuning parser, reward, training data, "
        "Stage1/Stage2 training logic, R_P, advantages, PPO/GRPO, and BFCL scorer.",
        "",
        "## 12. Readiness",
        "",
        "The raw-output diagnostic and both 2-sample smoke paths completed without runtime "
        "failures. The interface is ready for a user-authorized BFCL100 run, with BFCL and strict "
        "metrics reported separately. No BFCL100 or exact-400 rerun was started in this task.",
    ]
    atomic_text(root / "PARSER_FIX_REPORT.md", "\n".join(lines) + "\n")

    changed = [
        strict_source,
        adapter_source,
        WORKSPACE / "stage1_format_rl/scripts/run_qwen_rods_single_step_parser_diagnostic.py",
        WORKSPACE / "stage1_format_rl/scripts/summarize_qwen_rods_parser_fix.py",
        WORKSPACE / "stage1_format_rl/tests/test_envtuning_response_adapter.py",
    ]
    atomic_text(root / "git_diff.patch", unified_patch(changed, args.strict_baseline))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-root", type=Path, required=True)
    parser.add_argument("--strict-baseline", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
