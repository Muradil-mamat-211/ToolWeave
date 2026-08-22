#!/usr/bin/env python3
"""Unit tests for RODS Stage 2 progress reward wrapper."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rewards"))
from rods_stage2_progress_reward import compute_score

def call(utr, gt_len):
    return compute_score(
        reward_scores={"user_turn_rewards": utr},
        ground_truth=[["x"]] * gt_len,  # one expected call per user turn
    )

def test_case1():
    r = call([1, 0, 1, 1], 4)
    assert r["score"] == 0.75, r
    assert r["progress"] == 0.75
    assert r["terminal_count"] == 4
    assert r["missing_terminal_turns"] == 0
    assert r["terminal_coverage"] == 1.0
    print("PASS case1: expected=4, terminal=[1,0,1,1] -> score=0.75")

def test_case2():
    r = call([1], 4)
    assert r["score"] == 0.25, r
    assert r["missing_terminal_turns"] == 3
    assert r["incomplete_trajectory"] is True
    print("PASS case2: expected=4, terminal=[1] -> score=0.25 (3 missing as 0)")

def test_case3():
    r = call([-1, -2, -3], 4)
    assert r["score"] == 0.0, r
    assert r["terminal_count"] == 0
    assert r["missing_terminal_turns"] == 4
    print("PASS case3: expected=4, no 0/1 -> score=0")

def test_case4():
    try:
        call([0, 1, 1], 2)  # 3 terminal codes but expected 2
        raise AssertionError("case4 should have raised ValueError")
    except ValueError as e:
        print("PASS case4: expected=2, terminal=[0,1,1] -> raises ValueError")

def test_case5():
    # -3/-2/-1 must not enter numerator/denominator
    r = call([-3, -2, -1, 1, 0], 2)  # expected 2 turns, terminal=[1,0]
    assert r["score"] == 0.5, r
    assert r["count_-3"] == 1 and r["count_-2"] == 1 and r["count_-1"] == 1
    # denominator is 2 (expected), not 5 (all codes)
    print("PASS case5: -3/-2/-1 excluded from progress; denominator fixed at expected")

if __name__ == "__main__":
    test_case1(); test_case2(); test_case3(); test_case4(); test_case5()
    print("ALL REWARD UNIT TESTS PASSED")
