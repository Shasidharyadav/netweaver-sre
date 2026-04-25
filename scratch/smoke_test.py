"""End-to-end smoke test for the new unified architecture.

Verifies:
  1. Each /reset randomizes targets (no fixed entity per task)
  2. /step validates against the randomized targets
  3. Multi-step tasks T15, T21, T22 progress through stages
  4. T20 grader gives full marks for correct command + target
  5. /grader returns the rubric breakdown
  6. All exposed scores live strictly inside (0.001, 0.999)
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from server.app import app  # noqa: E402

client = TestClient(app)


def jget(path):
    r = client.get(path)
    r.raise_for_status()
    return r.json()


def jpost(path, payload=None):
    r = client.post(path, json=payload or {})
    r.raise_for_status()
    return r.json()


def assert_in_range(score: float, label: str):
    assert 0.001 <= float(score) <= 0.999, f"{label}={score} out of (0.001, 0.999)"


def alert_for_task(task_level: str) -> str:
    jpost("/set_level", {"task_level": task_level})
    obs = jpost("/reset")["observation"]
    return obs["alert"]


def extract_target_from_alert(alert: str, prefixes):
    """Return first token in `alert` starting with any prefix (strips punctuation)."""
    cleaned = alert
    for ch in [",", ".", "(", ")", "[", "]", "{", "}", ":", ";", "!", "?"]:
        cleaned = cleaned.replace(ch, " ")
    for tok in cleaned.split():
        for p in prefixes:
            if tok.startswith(p):
                return tok
    return None


# ── 1. Randomization across resets ───────────────────────────────────────────
print("\n[1] Randomization sanity")
for task in ["t01", "t16", "t18", "t19", "t20"]:
    seen = set()
    for _ in range(8):
        a = alert_for_task(task)
        seen.add(a)
    assert len(seen) > 1, f"{task}: expected randomized alerts, got 1 unique alert"
    print(f"  {task}: {len(seen)} unique alerts across 8 resets - OK")


# ── 2. Easy task: pick the exact randomized target from the alert ────────────
print("\n[2] Easy task end-to-end (T01)")
jpost("/set_level", {"task_level": "t01"})
reset = jpost("/reset")
alert = reset["observation"]["alert"]
node = extract_target_from_alert(alert, ["node_"])
assert node, f"could not parse node from alert: {alert}"
step = jpost("/step", {"action": {"command": "DRAIN_TRAFFIC", "target": node, "value": None}})
assert step["done"], f"T01 should resolve in one step, got: {step}"
assert step["observation"]["resolved"], "T01 should be resolved"
assert_in_range(step["reward"], "T01 reward")
print(f"  T01 resolved on {node}, reward={step['reward']:.3f}, breakdown={step['observation']['grader_breakdown']}")


# ── 3. Medium task: T11 must require value=9000 (BUG 3 fix) ──────────────────
print("\n[3] T11 MTU value must be 9000")
jpost("/set_level", {"task_level": "t11"})
reset = jpost("/reset")
alert = reset["observation"]["alert"]
sw = extract_target_from_alert(alert, ["sw_"])
# Wrong value first
step1 = jpost("/step", {"action": {"command": "INCREASE_MTU", "target": sw, "value": 1500}})
assert not step1["done"], "wrong value should not finish"
# Right value
step2 = jpost("/step", {"action": {"command": "INCREASE_MTU", "target": sw, "value": 9000}})
assert step2["done"] and step2["observation"]["resolved"], f"T11 with 9000 should resolve: {step2}"
print(f"  T11 resolved on {sw} with value=9000, reward={step2['reward']:.3f}")


# ── 4. T18 cluster keyword (BUG 2 fix) ──────────────────────────────────────
print("\n[4] T18 cluster keyword")
jpost("/set_level", {"task_level": "t18"})
reset = jpost("/reset")
alert = reset["observation"]["alert"]
cluster = extract_target_from_alert(alert, ["cluster_"])
step = jpost("/step", {"action": {"command": "ISSUE_GLOBAL_ROLLBACK", "target": cluster, "value": None}})
assert step["done"] and step["observation"]["resolved"], f"T18 should resolve: {step}"
print(f"  T18 resolved on {cluster}, reward={step['reward']:.3f}")


# ── 5. T20 db keyword + db_cluster_X target (NEW BUG fix) ───────────────────
print("\n[5] T20 db_cluster target works with grader 'db' keyword")
jpost("/set_level", {"task_level": "t20"})
reset = jpost("/reset")
alert = reset["observation"]["alert"]
db = extract_target_from_alert(alert, ["db_cluster_"])
assert db is not None, f"T20 alert should mention db_cluster_X, got: {alert}"
step = jpost("/step", {"action": {"command": "PURGE_CORRUPT_BLOCK", "target": db, "value": None}})
assert step["done"] and step["observation"]["resolved"], f"T20 should resolve: {step}"
br = step["observation"]["grader_breakdown"]
assert br["resolution"] >= 0.20, f"T20 resolution should be high, got {br}"
print(f"  T20 resolved on {db}, reward={step['reward']:.3f}, breakdown={br}")


# ── 6. T15 multi-step with order enforcement ────────────────────────────────
print("\n[6] T15 multi-step (RUN_MINI_ITERATION -> DRAIN_TRAFFIC) with order")
jpost("/set_level", {"task_level": "t15"})
reset = jpost("/reset")
alert = reset["observation"]["alert"]
cluster = extract_target_from_alert(alert, ["cluster_"])
# Wrong order first
s_bad = jpost("/step", {"action": {"command": "DRAIN_TRAFFIC", "target": cluster, "value": None}})
assert not s_bad["done"], "DRAIN before triage should not resolve"
# Correct order
s1 = jpost("/step", {"action": {"command": "RUN_MINI_ITERATION", "target": cluster, "value": None}})
s2 = jpost("/step", {"action": {"command": "DRAIN_TRAFFIC", "target": cluster, "value": None}})
assert s2["done"], f"T15 should resolve after both stages: {s2}"
print(f"  T15 resolved in 3 steps on {cluster}, reward={s2['reward']:.3f}")


# ── 7. T21 cascading multi-step (3 stages, order matters) ────────────────────
print("\n[7] T21 cascading multi-step")
jpost("/set_level", {"task_level": "t21"})
reset = jpost("/reset")
alert = reset["observation"]["alert"]
node = extract_target_from_alert(alert, ["node_"])
db = extract_target_from_alert(alert, ["db_cluster_"])
assert node and db, f"T21 alert must contain node and db: {alert}"
# Wrong order first
bad = jpost("/step", {"action": {"command": "PIN_CPU_THREADS", "target": node, "value": 64}})
assert not bad["done"], "T21 wrong order should not resolve"
# Correct order
s1 = jpost("/step", {"action": {"command": "ADJUST_POWER_CAP", "target": node, "value": 350}})
s2 = jpost("/step", {"action": {"command": "PIN_CPU_THREADS", "target": node, "value": 64}})
s3 = jpost("/step", {"action": {"command": "SCALE_CONN_POOL", "target": db, "value": 800}})
assert s3["done"], f"T21 must resolve after all 3 stages: {s3}"
print(f"  T21 resolved on {node}+{db}, reward={s3['reward']:.3f}, breakdown={s3['observation']['grader_breakdown']}")


# ── 8. T22 gradient poisoning + amplification ───────────────────────────────
print("\n[8] T22 gradient poisoning multi-step")
jpost("/set_level", {"task_level": "t22"})
reset = jpost("/reset")
alert = reset["observation"]["alert"]
cluster = extract_target_from_alert(alert, ["cluster_"])
sw = extract_target_from_alert(alert, ["sw_"])
assert cluster and sw, f"T22 alert must have cluster and switch: {alert}"
s1 = jpost("/step", {"action": {"command": "RUN_MINI_ITERATION", "target": cluster, "value": None}})
s2 = jpost("/step", {"action": {"command": "ISOLATE_BROADCAST_STORM", "target": sw, "value": None}})
s3 = jpost("/step", {"action": {"command": "DRAIN_TRAFFIC", "target": cluster, "value": None}})
assert s3["done"], f"T22 should resolve: {s3}"
print(f"  T22 resolved on {cluster}/{sw}, reward={s3['reward']:.3f}, breakdown={s3['observation']['grader_breakdown']}")


# ── 9. Score clamping discipline ────────────────────────────────────────────
print("\n[9] Score clamping (no 0.0 or 1.0 leaks)")
import random
for _ in range(30):
    level = random.choice([f"t{i:02d}" for i in range(1, 23)])
    jpost("/set_level", {"task_level": level})
    obs = jpost("/reset")
    assert_in_range(obs["reward"], f"{level} reset reward")
    for _ in range(random.randint(1, 4)):
        st = jpost("/step", {"action": {"command": "DRAIN_TRAFFIC", "target": "node_99", "value": None}})
        assert_in_range(st["reward"], f"{level} step reward")
        assert_in_range(st["shaped_reward"], f"{level} shaped_reward")
        assert_in_range(st["env_reward"], f"{level} env_reward")
        if st["done"]:
            break
print("  All exposed rewards within (0.001, 0.999) - OK")


# ── 10. /grader returns proper structure ────────────────────────────────────
print("\n[10] /grader endpoint structure")
g = jget("/grader")
assert set(g.keys()) >= {"resolved", "total", "breakdown"}, g
assert set(g["breakdown"].keys()) >= {"diagnosis", "resolution", "best_practice"}, g
assert_in_range(g["total"], "/grader total")
print(f"  /grader OK: {g}")

print("\n=== ALL SMOKE TESTS PASSED ===")
