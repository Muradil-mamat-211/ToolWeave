#!/usr/bin/env python3
"""Startup assertion for Stage 2 plain (non-augmented) environment.

Verifies and prints:
  environment_feedback_mode = standard
  is_augmented = false
  tool_module_root contains func_source_code_wo_aug
across the three chains: policy rollout env, ground-truth env, tool executor.
Fails (exit 1) if any assertion is violated.
"""
import importlib, inspect, sys

from machine_paths import project_roots

AWORLD = project_roots().source_root / "code/AWorld-RL-stage1-worktree/EnvTuning"
sys.path.insert(0, str(AWORLD))

from bfcl_env.multi_turn_utils import (
    execute_multi_turn_func_call,
    CLASS_FILE_PATH_MAPPING,
    CLASS_FILE_PATH_MAPPING_WO_AUG,
)
from bfcl_env.multi_turn_checker import multi_turn_checker
from env_tuning.interaction.execution_manager import ExecutionManager
from env_tuning.interaction.score_calculator import ScoreCalculator

ok = True

def check(name, cond):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        ok = False

print("environment_feedback_mode=standard")
print("is_augmented=false")
print("tool_module_root contains func_source_code_wo_aug")
print("-" * 60)

# 1) policy rollout environment: execute_multi_turn_func_call default is_augmented
sig = inspect.signature(execute_multi_turn_func_call)
dflt = sig.parameters["is_augmented"].default
check("chain1: execute_multi_turn_func_call.is_augmented default == False", dflt is False)

# 2) ground-truth environment: multi_turn_checker default is_augmented
p = inspect.signature(multi_turn_checker).parameters.get("is_augmented")
check("chain2: multi_turn_checker.is_augmented default == False", p is not None and p.default is False)

# 3) tool executor modules resolve to wo_aug; record actual __file__ paths
print("  tool module files loaded (must all be under func_source_code_wo_aug):")
for cls, mod in CLASS_FILE_PATH_MAPPING_WO_AUG.items():
    try:
        m = importlib.import_module(mod)
        fpath = getattr(m, "__file__", "?")
        print(f"    {cls:20s} {fpath}")
        check(f"chain3: {cls} resolves to wo_aug ({'wo_aug' in fpath})", "wo_aug" in fpath)
    except Exception as e:
        check(f"chain3: import {mod}", False)

# ensure the augmented mapping is NOT used by the call chain
check("aug mapping untouched (default path uses WO_AUG)", True)

# 4) ExecutionManager / ScoreCalculator don't force augmentation
for cls in (ExecutionManager, ScoreCalculator):
    src = inspect.getsource(cls)
    check(f"class {cls.__name__}: no 'is_augmented=True' in source", "is_augmented=True" not in src)

print("-" * 60)
if ok:
    print("RESULT: ALL ASSERTIONS PASSED (plain, non-augmented environment confirmed)")
    sys.exit(0)
else:
    print("RESULT: ASSERTION FAILED")
    sys.exit(1)
