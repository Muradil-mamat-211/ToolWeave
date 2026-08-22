#!/usr/bin/env python3
"""Compute comprehensive Stage-1 gate metrics + bootstrap CI from gate-eval rollouts.

Inputs: a rollout dir produced by eval_stage1_gate_dual_gpu.sh (rollouts/0.jsonl),
plus the val manifest for per-sample data_source/sample_id join (by order).
For each sample: score/format/tool/is_tool/progress/rounds (official values from
the reward), interaction-code counts and per-action parser success recomputed by
re-running the OFFICIAL EnvTuning parser on the stored multi-turn output.

Outputs: per-overall + per-category metrics with 95% bootstrap CIs, written to
--out-json, and a printed compact summary.
"""
from __future__ import annotations

import argparse, json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

AWORLD = "/root/autodl-tmp/rods-workspace/code/AWorld-RL-stage1-worktree/EnvTuning"
sys.path.insert(0, AWORLD)
from env_tuning.interaction.utils import parse_model_response  # official parser

ACTION_RE = re.compile(
    r"<think>.*?</think>\s*(?:<tool_call>.*?</tool_call>|<answer>.*?</answer>)",
    re.DOTALL,
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def enrich_with_manifest(rows, manifest_path):
    m = json.loads(Path(manifest_path).read_text())
    if "records" in m and m["records"]:
        records = m["records"]
    else:
        # legacy format: sample_ids list (assume all one category -> infer)
        sample_ids = m.get("sample_ids", [])
        records = [{"sample_id": sid, "data_source": m.get("category") or "multi_turn_base"} for sid in sample_ids]
    if len(rows) != len(records):
        raise RuntimeError(f"rows {len(rows)} != manifest {len(records)}")
    out = []
    for r, rec in zip(rows, records, strict=True):
        r = dict(r)
        r["sample_id"] = rec["sample_id"]
        r["data_source"] = rec.get("data_source", "multi_turn_base")
        out.append(r)
    return out


def parse_actions(output: str) -> list[dict]:
    """Extract assistant actions from the stored trajectory and classify each.

    Positional parsing: each action starts at a <think> and ends at the close of
    the following <tool_call> or <answer> block. This avoids regex backtracking
    across multiple actions in the concatenated transcript.
    """
    results = []
    i = 0
    n = len(output)
    while True:
        start = output.find("<think>", i)
        if start < 0:
            break
        think_end = output.find("</think>", start)
        if think_end < 0:
            break
        # after </think>, skip whitespace, look for action tag
        j = think_end + len("</think>")
        k = j
        while k < n and output[k].isspace():
            k += 1
        if output.startswith("<tool_call>", k):
            close = output.find("</tool_call>", k + len("<tool_call>"))
            if close < 0:
                break
            block_end = close + len("</tool_call>")
        elif output.startswith("<answer>", k):
            close = output.find("</answer>", k + len("<answer>"))
            if close < 0:
                break
            block_end = close + len("</answer>")
        else:
            # <think> not followed by a real action: this is the environment's
            # instruction text that literally mentions <think>/<answer> tags.
            # Skip it and keep scanning for real model actions.
            i = think_end + len("</think>")
            continue
        block = output[start:block_end]
        content, msg = parse_model_response(block)
        is_tool = "<tool_call>" in block
        results.append({
            "ok": msg in ("tool_call", "tool_all", "answer"),
            "msg": msg,
            "is_tool": is_tool,
        })
        i = block_end
    return results


def bootstrap_ci(vals, n_boot=2000, seed=42, alpha=0.95):
    if not vals:
        return (0.0, 0.0)
    rng = random.Random(seed)
    m = len(vals)
    means = [sum(rng.choice(vals) for _ in range(m)) / m for _ in range(n_boot)]
    means.sort()
    return (means[int((1 - alpha) / 2 * n_boot)], means[int((1 + alpha) / 2 * n_boot) - 1])


def analyze(rows):
    n = len(rows)
    def mean(key):
        v = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return sum(v) / len(v) if v else 0.0

    # per-action parser success by action index
    idx_ok = defaultdict(lambda: [0, 0])  # index -> [ok, total]
    # approximate interaction codes: -3 parse fail / -2 exec fail / -1 success / 0/1 turn
    approx_codes = Counter()
    terminal_covered = 0  # trajectory ends with a successfully-closed turn
    for r in rows:
        acts = parse_actions(r.get("output", ""))
        for i, a in enumerate(acts):
            bucket = 1 if i == 0 else (2 if i == 1 else 3)  # action index 1, 2, 3+
            idx_ok[bucket][1] += 1
            if a["ok"]:
                idx_ok[bucket][0] += 1
        if acts:
            last = acts[-1]
            if last["ok"] and not last["is_tool"]:
                terminal_covered += 1  # ends with a valid <answer> -> turn closed
            if not last["ok"]:
                approx_codes[-3] += 1
            elif last["is_tool"]:
                approx_codes[-1] += 1
            else:
                approx_codes[1] += 1
        else:
            approx_codes[-3] += 1

    action_parser = {}
    for idx in (1, 2, 3):
        ok, tot = idx_ok[idx]
        action_parser[f"action_index_{idx}_parser_success"] = ok / tot if tot else None
    # pooled parser success for ALL actions at index >= 2
    ok_ge2 = idx_ok[2][0] + idx_ok[3][0]
    tot_ge2 = idx_ok[2][1] + idx_ok[3][1]
    action_parser["action_index_ge2_parser_success"] = ok_ge2 / tot_ge2 if tot_ge2 else None

    score = [r["score"] for r in rows if isinstance(r.get("score"), (int, float))]
    fmt = [r["format_reward"] for r in rows if isinstance(r.get("format_reward"), (int, float))]
    tool = [r["tool_call_reward"] for r in rows if isinstance(r.get("tool_call_reward"), (int, float))]
    ist = [r["is_tool_call"] for r in rows if isinstance(r.get("is_tool_call"), (int, float))]
    prog = [r["progress"] for r in rows if isinstance(r.get("progress"), (int, float))]

    total_codes = sum(approx_codes.values()) or 1
    return {
        "n": n,
        "score": mean("score"),
        "format_reward": mean("format_reward"),
        "tool_call_reward": mean("tool_call_reward"),
        "is_tool_call": mean("is_tool_call"),
        "progress": mean("progress"),
        "rounds": mean("total_interaction_rounds"),
        "approx_code_counts": dict(approx_codes),
        "approx_code_ratios": {str(k): v / total_codes for k, v in approx_codes.items()},
        "action_parser_success": action_parser,
        "terminal_coverage": terminal_covered / n if n else 0.0,
        "truncation_rate": mean("truncation_rate"),
        "early_termination_rate": mean("early_termination_rate"),
        "zero_score_rate": mean("zero_score_rate") if "zero_score_rate" in rows[0] else None,
        "bootstrap_ci": {
            "score": bootstrap_ci(score),
            "format_reward": bootstrap_ci(fmt),
            "tool_call_reward": bootstrap_ci(tool),
            "is_tool_call": bootstrap_ci(ist),
            "progress": bootstrap_ci(prog),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    rd = Path(args.rollout_dir)
    all_rows = []
    for f in sorted(rd.glob("*.jsonl")):
        all_rows.extend(load_jsonl(f))
    if not all_rows:
        print(f"NO ROWS in {rd}"); raise SystemExit(1)
    all_rows = enrich_with_manifest(all_rows, args.manifest)

    result = {
        "model": args.model,
        "dataset": args.dataset,
        "overall": analyze(all_rows),
        "per_category": {},
    }
    cats = defaultdict(list)
    for r in all_rows:
        cats[r["data_source"]].append(r)
    result["per_category"] = {c: analyze(rs) for c, rs in sorted(cats.items())}

    Path(args.out_json).write_text(json.dumps(result, indent=2) + "\n")
    o = result["overall"]
    print(json.dumps({
        "model": args.model, "dataset": args.dataset,
        "score": round(o["score"], 4), "format": round(o["format_reward"], 4),
        "tool": round(o["tool_call_reward"], 4), "is_tool": round(o["is_tool_call"], 4),
        "progress": round(o["progress"], 4), "rounds": round(o["rounds"], 2),
        "act_parser": {k: (round(v,3) if v is not None else None) for k,v in o["action_parser_success"].items()},
        "codes": o["approx_code_counts"],
    }))


if __name__ == "__main__":
    main()
