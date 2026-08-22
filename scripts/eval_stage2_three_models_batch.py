#!/usr/bin/env python3
"""Batched vLLM frontend for the official BFCL interaction evaluator."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_stage2_three_models as base


class BatchTrajectory:
    def __init__(self, evaluator, sample, seed, stochastic, max_actions, rollout_index):
        self.evaluator = evaluator
        self.sample = sample
        self.seed = seed
        self.rollout_index = rollout_index
        self.stochastic = stochastic
        self.max_actions = max_actions
        self.instance = f"{evaluator.name}_{sample['id']}_{seed}"
        self.interaction = base.MultiTurnFunctionCallInteraction({"name": "stage2_three_model_eval"})
        self.state = None
        self.rewards = []
        self.actions = []
        self.conversation = [dict(x) for x in sample["messages"]]
        self.action_index = 0
        self.terminated = False
        self.error = None

    async def start(self):
        try:
            await self.interaction.start_interaction(
                instance_id=self.instance,
                id=self.sample["id"],
                initial_config=self.sample["initial_config"],
                involved_classes=self.sample["involved_classes"],
                ground_truth=self.sample["ground_truth"],
                processed_question=list(self.sample["processed_question"]),
                question=self.sample["question_turns"],
            )
            self.state = self.interaction._instance_dict[self.instance]
        except Exception as exc:
            self.error = repr(exc)

    async def step(self, raw):
        if self.error or self.terminated:
            return
        turn = int(self.state.current_turn_index)
        fmt, calls, parser_text, adapted = base.action_format(raw, self.sample["tools"])
        self.conversation.append({"role": "assistant", "content": raw})
        try:
            should_end, response, score, _ = await self.interaction.generate_response(
                self.instance,
                [{"role": "assistant", "content": parser_text}],
                id=self.sample["id"],
            )
            score = float(score)
            self.rewards.append(score)
            self.state = self.interaction._instance_dict.get(self.instance, self.state)
            self.actions.append({
                "turn_index": turn,
                "action_index": self.action_index,
                "raw_output": raw,
                "parser_input": parser_text,
                "parser_adapter_applied": adapted,
                "parsed_tool_calls": calls,
                "parse_status": fmt["parse_message"],
                "execution_status": base.score_status(score),
                "step_score": score,
                "tool_response": response,
                "environment_state_after": {
                    "turn": self.state.current_turn_index,
                    "attempt": self.state.current_turn_attempt_counts,
                    "instances": base.bounded_json(self.state.involved_instances),
                    "execution_results": base.bounded_json(self.state.all_turn_model_execution_results + self.state.single_turn_model_execution_results),
                },
                "format_metrics": fmt,
            })
            self.action_index += 1
            if should_end:
                self.terminated = True
            else:
                self.conversation.append({"role": "user", "content": response})
        except Exception as exc:
            self.error = repr(exc)

    async def finish(self):
        if self.error is None and not self.terminated:
            self.error = "max_actions_reached"
        try:
            await self.interaction.finalize_interaction(instance_id=self.instance)
        except Exception:
            pass
        reward = base.official_reward({"user_turn_rewards": self.rewards}, self.sample["ground_truth"], extra_info={"id": self.sample["id"]})
        counts = {}
        for action in self.actions:
            for key, value in action["format_metrics"].items():
                if key != "parse_message": counts[key] = counts.get(key, 0) + int(value)
        return {
            "model_name": self.evaluator.name,
            "model_path": str(self.evaluator.path),
            "dataset_name": self.sample["dataset_name"],
            "category": self.sample.get("category", "base"),
            "sample_id": self.sample["id"],
            "rollout_index": self.rollout_index,
            "decoding_mode": "stochastic" if self.stochastic else "deterministic",
            "seed": self.seed,
            "rollout_index": self.seed % 100,
            "initial_prompt": json.dumps(self.sample["messages"], ensure_ascii=False),
            "tools": self.sample["tools"],
            "conversation": self.conversation,
            "assistant_actions": self.actions,
            "user_turn_rewards": self.rewards,
            "progress_score": float(reward.get("score", 0.0)),
            "format_reward": float(reward.get("format_reward", 0.0)),
            "tool_call_reward": float(reward.get("tool_call_reward", 0.0)),
            "is_tool_call": float(reward.get("is_tool_call", 0.0)),
            "num_tool_calls": sum(len(x["parsed_tool_calls"]) for x in self.actions),
            "num_turns": sum(x in (0, 1) for x in self.rewards),
            "terminated_reason": "environment_terminated" if self.terminated else self.error,
            "environment_error": self.error,
            "parse_error_count": self.rewards.count(-3),
            "execution_error_count": self.rewards.count(-2),
            "successful_execution_count": self.rewards.count(-1),
            "incorrect_user_turn_count": self.rewards.count(0),
            "correct_user_turn_count": self.rewards.count(1),
            "over_max_turn": int(self.error == "max_actions_reached"),
            "format_counts": {**counts, "assistant_actions": len(self.actions), "tool_attempts": sum(x["valid_tool_call_block"] for x in [a["format_metrics"] for a in self.actions])},
            "eval_config": {"official_environment": True, "official_reward": "env_tuning.bfcl_reward.compute_score", "official_max_attempts_per_user_turn": self.interaction.max_step_limit, "max_actions": self.max_actions, "max_new_tokens": self.evaluator.max_new_tokens, "temperature": 0.7 if self.stochastic else 0.0, "top_p": 0.9 if self.stochastic else None, "batched_vllm": True},
        }


async def run_batch(items, batch_size, evaluator, output):
    for start in range(0, len(items), batch_size):
        group = items[start : start + batch_size]
        await asyncio.gather(*(x.start() for x in group))
        active = [x for x in group if not x.error]
        while active:
            messages = [x.conversation for x in active]
            seeds = [x.seed + x.action_index for x in active]
            raws = evaluator.generate_batch(messages, seeds, active[0].stochastic)
            await asyncio.gather(*(x.step(raw) for x, raw in zip(active, raws)))
            active = [x for x in active if not x.terminated and not x.error and x.action_index < x.max_actions]
        for item in group:
            row = await item.finish()
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()


async def main(args):
    data = base.datasets()
    root = Path(args.eval_root) / ("smoke" if args.mode == "smoke" else "")
    root.mkdir(parents=True, exist_ok=True)
    order = ["train_base_100", "heldout_base_100", "heldout_mixed_150"]
    for model_name, model_path in base.MODEL_PATHS.items():
        out_dir = root / model_name; out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{model_name}_eval_outputs.jsonl"
        if out_path.exists(): out_path.unlink()
        print(f"[model-start] {model_name}", flush=True)
        evaluator = base.Evaluator(model_name, model_path, args.max_new_tokens, args.max_actions, "vllm")
        with out_path.open("w", encoding="utf-8") as output:
            for di, dataset in enumerate(order):
                samples = data[dataset][: args.max_per_dataset]
                for mode, n in (("deterministic", 1), ("stochastic", args.stochastic_n)):
                    stochastic = mode == "stochastic"
                    items = []
                    for si, sample in enumerate(samples):
                        for ri in range(n):
                            seed = 42 + di * 100000 + si * 100 + ri
                            items.append(BatchTrajectory(evaluator, sample, seed, stochastic, args.max_actions, ri))
                    await run_batch(items, args.batch_size, evaluator, output)
                    print(f"[done] {model_name} {dataset} {mode} trajectories={len(items)}", flush=True)
        evaluator.unload(); print(f"[model-done] {out_path}", flush=True)
    if args.mode == "formal":
        by_model, comparison = base.summarize(root)
        base.cases(root, by_model)
        base.report(root, by_model, comparison, args)
        print(f"[summary-done] {root}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "formal"], required=True)
    parser.add_argument("--eval_root", required=True)
    parser.add_argument("--max_per_dataset", type=int, default=5)
    parser.add_argument("--stochastic_n", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_actions", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    asyncio.run(main(parser.parse_args()))
