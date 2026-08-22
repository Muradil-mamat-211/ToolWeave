#!/usr/bin/env python3
"""Evaluate the 6 RODS Stage-1 gate conditions from gate-eval metrics + training log."""
import json, re, statistics, sys
from pathlib import Path

METRIC_ROOT = Path("/root/autodl-tmp/rods-workspace/stage1_format_rl/artifacts/gate_vs_base/metrics")
TRAIN_LOG = Path("/root/autodl-tmp/rods-workspace/stage1_format_rl/logs/retrain_train.log")

def load(name):
    p = METRIC_ROOT / f"{name}.json"
    if not p.exists():
        print(f"  [MISSING] metrics {name}"); return None
    return json.loads(p.read_text())

def base100(d):
    if not d: return None
    cat = d.get("per_category", {}).get("multi_turn_base")
    return cat if cat else d.get("overall")

def main():
    base = load("base_qwen3_4b_valbase")
    b400 = load("base_qwen3_4b_val400")
    s20 = load("step20_val400")
    s25 = load("step25_val400")

    baseB = base100(base) if base else None
    s20B = base100(s20)
    s25B = base100(s25)

    results = {}
    def cond(name, ok, detail):
        results[name] = (ok, detail)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    # ---- cond 1: step25 score >= base + 0.02 (Base-100) ----
    if baseB and s25B:
        ok = s25B["score"] >= baseB["score"] + 0.02
        cond("C1 step25 score>=base+0.02", ok,
             f"s25={s25B['score']:.4f} base={baseB['score']:.4f} (need >= {baseB['score']+0.02:.4f})")
    else:
        cond("C1", False, "missing data")

    # ---- cond 2: format/tool/is_tool not below base (Base-100) ----
    if baseB and s25B:
        ok = (s25B["format_reward"] >= baseB["format_reward"] - 1e-9 and
              s25B["tool_call_reward"] >= baseB["tool_call_reward"] - 1e-9 and
              s25B["is_tool_call"] >= baseB["is_tool_call"] - 1e-9)
        cond("C2 format/tool/is_tool>=base", ok,
             f"s25 fmt={s25B['format_reward']:.3f}/tool={s25B['tool_call_reward']:.3f}/is_tool={s25B['is_tool_call']:.3f} vs base fmt={baseB['format_reward']:.3f}/tool={baseB['tool_call_reward']:.3f}/is_tool={baseB['is_tool_call']:.3f}")
    else:
        cond("C2", False, "missing data")

    # ---- cond 3: plateau |s25-s20|/|s20| < 0.01 (Base-100) ----
    if s20B and s25B:
        rel = abs(s25B["score"] - s20B["score"]) / max(abs(s20B["score"]), 1e-8)
        ok = rel < 0.01
        cond("C3 plateau <1%", ok, f"|{s25B['score']:.4f}-{s20B['score']:.4f}|/{abs(s20B['score']):.4f}={rel:.5f}")
    else:
        cond("C3", False, "missing step20/25")

    # ---- cond 4: action>=2 parser >=0.90 or within 5pp of idx1; terminal coverage >=0.95 ----
    if s25B:
        ap = s25B.get("action_parser_success", {})
        a1 = ap.get("action_index_1_parser_success")
        ge2 = ap.get("action_index_ge2_parser_success")
        tc = s25B.get("terminal_coverage", 0.0)
        parser_ok = False
        if a1 is not None and ge2 is not None:
            parser_ok = (ge2 >= 0.90) or (a1 - ge2 <= 0.05)
        def fmt(v): return f"{v:.3f}" if v is not None else "None"
        cond("C4a action>=2 pooled parser>=0.90 or <=5pp of idx1", parser_ok,
             f"act1={fmt(a1)} pooled_ge2={fmt(ge2)}")
        # NOTE: terminal coverage here is a PROXY (trajectory ends with valid
        # <answer>); official 0/1 coverage is not persisted by the gate eval and
        # is validated properly by the Stage 2 smoke (user_turn_rewards logged).
        cond("C4b terminal coverage (PROXY, official checked in smoke)", tc >= 0.95,
             f"proxy_term_cov={tc:.3f} [proxy: ends-with-valid-answer]")
    else:
        cond("C4", False, "missing step25")

    # ---- cond 5: training health from log ----
    gns, ents, kls, clips = [], [], [], []
    for line in TRAIN_LOG.read_text().splitlines():
        if "global_step" not in line: continue
        def get(pat):
            m = re.search(pat, line)
            return float(m.group(1)) if m else None
        g = get(r"actor/grad_norm:([\d.eE+-]+)")
        e = get(r"actor/entropy:([\d.eE+-]+)")
        k = get(r"actor/kl_loss:([\d.eE+-]+)")
        c = get(r"response_length/clip_ratio:([\d.eE+-]+)")
        if g is not None: gns.append(g)
        if e is not None: ents.append(e)
        if k is not None: kls.append(k)
        if c is not None: clips.append(c)
    # last two epochs = last 10 steps
    recent_gn = gns[-10:]
    ok5 = (len(recent_gn) >= 2 and all(g < 2.0 for g in recent_gn))
    if len(recent_gn) >= 6:
        half = len(recent_gn)//2
        med_a, med_b = statistics.median(recent_gn[:half]), statistics.median(recent_gn[half:])
        ok5 = ok5 and (abs(med_a-med_b)/max(abs(med_a),1e-8) < 0.20)
    cond("C5 grad_norm health (last 10)", ok5,
         f"recent={[round(g,2) for g in recent_gn]}")
    def finite(lst): return all(v == v for v in lst[-10:]) if lst else True
    ent_l = f"entropy~{ents[-1]:.3f}" if ents else "entropy:no-data"
    kl_l = f"kl~{kls[-1]:.4f}" if kls else "kl:no-data"
    clip_l = f"clip~{clips[-1]:.3f}" if clips else "clip:no-data"
    cond("C5b entropy/KL/clip finite", finite(ents) and finite(kls) and finite(clips),
         f"{ent_l} {kl_l} {clip_l}")

    # ---- cond 6: val-400 category format drop <=5pp vs base ----

    if b400 and s25:
        cats = ["multi_turn_base","multi_turn_long_context","multi_turn_miss_func","multi_turn_miss_param"]
        fails = []
        for cat in cats:
            bf = b400.get("per_category",{}).get(cat,{}).get("format_reward")
            sf = s25.get("per_category",{}).get(cat,{}).get("format_reward")
            if bf is not None and sf is not None and sf < bf - 0.05:
                fails.append(f"{cat}: base={bf:.3f}->s25={sf:.3f}")
        cond("C6 val400 no category format drop>5pp", len(fails)==0, "; ".join(fails) if fails else "all categories ok")

    allpass = all(ok for ok,_ in results.values())
    print("-"*60)
    print("GATE_RESULT:", "ALL_PASS -> READY_FOR_STAGE2" if allpass else "FAILED")
    return 0 if allpass else 1

if __name__ == "__main__":
    sys.exit(main())
