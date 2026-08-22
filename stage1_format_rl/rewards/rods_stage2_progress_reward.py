"""RODS Stage 2 Progress Reward (fixed-denominator wrapper).

Replaces the plain `bfcl_reward.py` denominator (len(relevant)) with the RODS
definition: progress = (# of successfully solved user turns) / expected_user_turns.

- expected_user_turns is taken from the sample's ground truth metadata
  (len(ground_truth) == number of expected user turns).
- Only terminal code 1 counts as success; 0 counts as failure.
- -3 / -2 / -1 are intermediate diagnostics and never enter the numerator or
  the denominator.
- Missing 0/1 turns (truncated / early-terminated / abnormal trajectories) are
  counted as failures 0 by keeping the FIXED denominator expected_user_turns.
- If terminal_count > expected_user_turns, we raise (must not silently train).

The official `env_tuning/bfcl_reward.py` is intentionally left unmodified.
"""

from __future__ import annotations


def compute_score(
    reward_scores: dict | None = None,
    ground_truth: list | None = None,
    extra_info: dict | None = None,
    **kwargs,
) -> dict:
    user_turn_rewards = list((reward_scores or {}).get("user_turn_rewards", []))
    try:
        expected_user_turns = len(ground_truth or [])
    except TypeError:
        expected_user_turns = len(list(ground_truth)) if ground_truth is not None else 0

    terminal = [c for c in user_turn_rewards if c in (0, 1)]
    terminal_count = len(terminal)
    success_count = sum(1 for c in terminal if c == 1)

    if terminal_count > expected_user_turns:
        # Abnormal: more closed turns than the task expects. Must not silently train.
        raise ValueError(
            "RODS Stage 2 progress: terminal_count "
            f"({terminal_count}) > expected_user_turns ({expected_user_turns}); "
            f"sample extra_info={extra_info!r}; user_turn_rewards={user_turn_rewards!r}"
        )

    missing_terminal_turns = max(0, expected_user_turns - terminal_count)
    terminal_coverage = (
        terminal_count / expected_user_turns if expected_user_turns > 0 else 0.0
    )
    # Missing turns count as failures 0 against the FIXED denominator.
    progress = success_count / expected_user_turns if expected_user_turns > 0 else 0.0

    counts = {
        "count_-3": user_turn_rewards.count(-3),
        "count_-2": user_turn_rewards.count(-2),
        "count_-1": user_turn_rewards.count(-1),
        "count_0": user_turn_rewards.count(0),
        "count_1": user_turn_rewards.count(1),
    }
    # Purity: progress must be a clean [0,1] reward (no KL penalty added here).
    progress = max(0.0, min(1.0, progress))

    return {
        "score": progress,  # GRPO training reward is strictly progress.
        "progress": progress,
        "expected_user_turns": expected_user_turns,
        "terminal_count": terminal_count,
        "terminal_coverage": terminal_coverage,
        "missing_terminal_turns": missing_terminal_turns,
        "incomplete_trajectory": terminal_count < expected_user_turns,
        **counts,
    }
