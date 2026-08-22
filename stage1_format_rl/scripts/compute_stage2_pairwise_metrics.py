#!/usr/bin/env python3
"""Stage-2 final comparison: per-model metrics + paired bootstrap CIs (eval_400).

For each model run (rollouts/*.jsonl produced by eval_stage2_gate_dual_gpu.sh under
the Stage-2 config), joins rows to the val_400 manifest by order (sample_id), then:

  - true wrapper fields from the rollout row itself: progress, score,
    terminal_coverage, incomplete_trajectory, expected_user_turns, count_*;
  - derived with the OFFICIAL BFCL formulas (format_reward.py), exactly
    reconstructing user_turn_rewards from the stored counts:
      rounds = len(user_turn_rewards)
      format_reward, tool_call_reward, is_tool_call
  - per-action parser success by re-running the OFFICIAL EnvTuning parser on the
    stored trajectory (same as Stage-1 gate metrics).

Paired bootstrap (fixed seed, 2000 resamples) of the per-sample progress DIFFERENCE
between each pair of models, using identical sample IDs: overall + per category.

Usage:
  compute_stage2_pairwise_metrics.py \
      --run LABEL=ROLLOUT_DIR --run LABEL2=ROLLOUT_DIR2 ... \
      --manifest MANIFEST.json --out-json OUT.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from machine_paths import project_roots

AWORLD = project_roots().source_root / "code/AWorld-RL-stage1-worktree/EnvTuning"
sys.path.insert(0, str(AWORLD))
from env_tuning.interaction.utils import parse_model_response  # official parser

ACTION_RE = re.compile(
    r"<think>.*?</think>\s*(?:<tool_call>.*?</tool_call>|<answer>.*?</answer>)",
    re.DOTALL,
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def enrich_with_manifest(rows, manifest_path):
    m = json.loads(Path(manifest_path).read_text())
    records = m.get("records", [])
    if not records:
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
    """Same positional action parser as Stage-1 gate metrics (official parser)."""
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
            i = think_end + len("</think>")
            continue
        block = output[start:block_end]
        _, msg = parse_model_response(block)
        results.append({"ok": msg in ("tool_call", "tool_all", "answer")})
        i = block_end
    return results


def reconstruct_user_turn_rewards(r) -> list[int]:
    """Rebuild the exact user_turn_rewards list from stored per-code counts."""
    return (
        [-3] * int(r.get("count_-3", 0))
        + [-2] * int(r.get("count_-2", 0))
        + [-1] * int(r.get("count_-1", 0))
        + [0] * int(r.get("count_0", 0))
        + [1] * int(r.get("count_1", 0))
    )


def derive_bfcl_metrics(r) -> dict:
    """Official BFCL formulas (env_tuning/format_reward.py::compute_score body)."""
    u = reconstruct_user_turn_rewards(r)
    rounds = len(u)
    correct_tool_call = u.count(-1)
    error_tool_call = u.count(-2)
    error_format = u.count(-3)
    is_tool_call = 1.0 if (correct_tool_call + error_tool_call) > 0 else 0.0
    format_reward = ((rounds - error_format) / rounds) if rounds > 0 else 0.0
    tool_call_reward = (
        correct_tool_call / (correct_tool_call + error_tool_call)
        if is_tool_call > 0
        else 0.0
    )
    return {
        "rounds": rounds,
        "format_reward": format_reward,
        "tool_call_reward": tool_call_reward,
        "is_tool_call": is_tool_call,
    }


def per_sample_metrics(rows) -> list[dict]:
    out = []
    for r in rows:
        b = derive_bfcl_metrics(r)
        acts = parse_actions(r.get("output", ""))
        idx_ok = [0, 0, 0]  # action index 1, 2, 3+
        for i, a in enumerate(acts):
            bkt = 0 if i == 0 else (1 if i == 1 else 2)
            idx_ok[bkt] += 1 if a["ok"] else 0
        a1_ok, a1_tot = idx_ok[0], 1 if len(acts) >= 1 else 0
        a2_ok, a2_tot = idx_ok[1], 1 if len(acts) >= 2 else 0
        a3_ok, a3_tot = idx_ok[2], max(0, len(acts) - 2)
        out.append({
            "sample_id": r["sample_id"],
            "data_source": r["data_source"],
            "progress": float(r.get("progress", 0.0)),
            "score": float(r.get("score", 0.0)),
            "terminal_coverage": float(r.get("terminal_coverage", 0.0)),
            "incomplete_trajectory": float(bool(r.get("incomplete_trajectory", False))),
            "expected_user_turns": float(r.get("expected_user_turns", 0.0)),
            "count_-3": float(r.get("count_-3", 0.0)),
            "count_-2": float(r.get("count_-2", 0.0)),
            "count_-1": float(r.get("count_-1", 0.0)),
            "count_0": float(r.get("count_0", 0.0)),
            "count_1": float(r.get("count_1", 0.0)),
            "rounds": b["rounds"],
            "format_reward": b["format_reward"],
            "tool_call_reward": b["tool_call_reward"],
            "is_tool_call": b["is_tool_call"],
            "act1_ok": a1_ok, "act1_tot": a1_tot,
            "act2_ok": a2_ok, "act2_tot": a2_tot,
            "act3_ok": a3_ok, "act3_tot": a3_tot,
            "actge2_ok": a2_ok + a3_ok, "actge2_tot": a2_tot + a3_tot,
        })
    return out


def summarize(samples: list[dict]) -> dict:
    n = len(samples)
    def mean(key):
        return sum(s[key] for s in samples) / n if n else 0.0
    def rate(key):
        return mean(key)
    act = {
        "action_index_1_parser_success": (sum(s["act1_ok"] for s in samples) / sum(s["act1_tot"] for s in samples)) if sum(s["act1_tot"] for s in samples) else None,
        "action_index_2_parser_success": (sum(s["act2_ok"] for s in samples) / sum(s["act2_tot"] for s in samples)) if sum(s["act2_tot"] for s in samples) else None,
        "action_index_3plus_parser_success": (sum(s["act3_ok"] for s in samples) / sum(s["act3_tot"] for s in samples)) if sum(s["act3_tot"] for s in samples) else None,
        "action_index_ge2_parser_success": (sum(s["actge2_ok"] for s in samples) / sum(s["actge2_tot"] for s in samples)) if sum(s["actge2_tot"] for s in samples) else None,
    }
    return {
        "n": n,
        "progress": mean("progress"),
        "score": mean("score"),
        "terminal_coverage": mean("terminal_coverage"),
        "incomplete_trajectory_rate": rate("incomplete_trajectory"),
        "expected_user_turns": mean("expected_user_turns"),
        "count_-3": mean("count_-3"),
        "count_-2": mean("count_-2"),
        "count_-1": mean("count_-1"),
        "count_0": mean("count_0"),
        "count_1": mean("count_1"),
        "rounds": mean("rounds"),
        "format_reward": mean("format_reward"),
        "tool_call_reward": mean("tool_call_reward"),
        "is_tool_call": rate("is_tool_call"),
        "action_parser_success": act,
    }


def paired_bootstrap_diff(samples_a: list[dict], samples_b: list[dict], n_boot=2000, seed=42):
    """Paired bootstrap 95% CI of mean(progress_B - progress_A) on identical sample IDs."""
    a = {s["sample_id"]: s["progress"] for s in samples_a}
    b = {s["sample_id"]: s["progress"] for s in samples_b}
    common = sorted(set(a) & set(b))
    if not common:
        return None
    pa = [a[sid] for sid in common]
    pb = [b[sid] for sid in common]
    diffs = [x - y for x, y in zip(pb, pa)]
    rng = random.Random(seed)
    m = len(diffs)
    means = [sum(rng.choice(diffs) for _ in range(m)) / m for _ in range(n_boot)]
    means.sort()
    lo, hi = means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]
    return {"paired_n": m, "mean_diff": sum(diffs) / m, "ci95": [lo, hi], "pct_positive": sum(1 for x in diffs if x > 0) / m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=ROLLOUT_DIR")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    runs = {}
    for spec in args.run:
        label, rd = spec.split("=", 1)
        rows = []
        for f in sorted(Path(rd).glob("*.jsonl")):
            rows.extend(load_jsonl(f))
        if not rows:
            print(f"NO ROWS in {rd}")
            sys.exit(1)
        runs[label] = per_sample_metrics(enrich_with_manifest(rows, args.manifest))

    result = {"models": list(runs), "per_model": {}, "paired_bootstrap": {}}
    for label, samples in runs.items():
        by_cat = defaultdict(list)
        for s in samples:
            by_cat[s["data_source"]].append(s)
        result["per_model"][label] = {
            "overall": summarize(samples),
            "per_category": {c: summarize(ss) for c, ss in sorted(by_cat.items())},
        }

    # paired bootstrap: later-vs-earlier model progress difference per grouping
    labels = list(runs)
    cats = ["overall", "multi_turn_base", "multi_turn_long_context", "multi_turn_miss_func", "multi_turn_miss_param"]
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            la, lb = labels[i], labels[j]
            key = f"{lb}_minus_{la}"
            result["paired_bootstrap"][key] = {}
            for cat in cats:
                sa = runs[la] if cat == "overall" else [s for s in runs[la] if s["data_source"] == cat]
                sb = runs[lb] if cat == "overall" else [s for s in runs[lb] if s["data_source"] == cat]
                result["paired_bootstrap"][key][cat] = paired_bootstrap_diff(sa, sb)

    Path(args.out_json).write_text(json.dumps(result, indent=2) + "\n")
    print(f"WROTE {args.out_json}")


if __name__ == "__main__":
    main()
