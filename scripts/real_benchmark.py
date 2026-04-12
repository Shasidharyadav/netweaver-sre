"""
Real Model Benchmarking for Netweaver SRE Environment.
Runs available HF models against all 20 tasks in our actual local environment.
"""

import os
import sys
import json
import time
import re
import requests
import traceback
from datetime import datetime
from openai import OpenAI

# --- Config ---
HF_TOKEN = os.environ.get("HF_TOKEN")
API_BASE  = "https://router.huggingface.co/v1"
ENV_URL   = "http://127.0.0.1:8000"

# Only models confirmed available on HF router
MODELS_TO_TEST = [
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "Qwen/Qwen2.5-7B-Instruct",
]

TASKS = [
    ("t01", "easy"),  ("t02", "easy"),  ("t03", "easy"),
    ("t04", "easy"),  ("t05", "easy"),  ("t06", "easy"),
    ("t07", "easy"),  ("t08", "medium"),("t09", "medium"),
    ("t10", "medium"),("t11", "medium"),("t12", "medium"),
    ("t13", "medium"),("t14", "medium"),("t15", "hard"),
    ("t16", "hard"),  ("t17", "hard"),  ("t18", "hard"),
    ("t19", "hard"),  ("t20", "hard"),
]

SRE_PROMPT = """You are an Autonomous Site Reliability Engineer for a 100-node GPU cluster.
System Logs: {logs}
Observation Queue Depths: {q}
Observation Gradient Variances: {v}
Observation GPU Memory Usage: {m}

INSTRUCTIONS:
- Read the hardware logs carefully. They tell you exactly what command to issue.
- If logs say "Isolate" a node using DRAIN_TRAFFIC -> command=DRAIN_TRAFFIC, target=the_node
- If logs say "CLEAR_DNS_CACHE" -> command=CLEAR_DNS_CACHE, target=the_node
- If logs say "RESTART_SERVICE" -> command=RESTART_SERVICE, target=the_node
- If logs say "RENEW_CERTIFICATE" -> command=RENEW_CERTIFICATE, target=the_node
- If logs say "CLEAR_TEMP_FILES" -> command=CLEAR_TEMP_FILES, target=the_node
- If logs say "RESTART_POD" -> command=RESTART_POD, target=the_node
- If logs say "KILL_ZOMBIE_PROCESS" -> command=KILL_ZOMBIE_PROCESS, target=the_node
- If logs mention a threshold/cap/limit with a number (Target: N), use that as value.
- If logs say "TUNE_PFC_THRESHOLD" -> command=TUNE_PFC_THRESHOLD, target=switch, value=N
- If logs say "ADJUST_POWER_CAP" with N W -> command=ADJUST_POWER_CAP, target=node, value=N
- If logs say "MITIGATE_ROUTE_FLAP" with AS N -> command=MITIGATE_ROUTE_FLAP, target=node, value=N
- If logs say "INCREASE_MTU" to N -> command=INCREASE_MTU, target=node, value=N
- If logs say "SET_RATE_LIMIT" to N -> command=SET_RATE_LIMIT, target=node, value=N
- If logs say "SCALE_CONN_POOL" to N -> command=SCALE_CONN_POOL, target=node, value=N
- If logs say "PIN_CPU_THREADS" to N -> command=PIN_CPU_THREADS, target=node, value=N
- If gradient_variances has a 999.9 spike, RUN_MINI_ITERATION with target="0-9" first to find it
- If queue_depths has 99.9 on sw_core_02, ISOLATE_BROADCAST_STORM on sw_core_02
- If gpu_memory_usage has 99.9 spike at index X, RESTART_GPU_DAEMON on sub_X
- If all gradient_variances are 0.0, ISSUE_GLOBAL_ROLLBACK on cluster_0
- If queue_depths is split (some 0, some 99.9), REBOOT_LEAF_SWITCHES on pod_x
- If logs say "PURGE_CORRUPT_BLOCK", issue that on db_cluster

Return ONLY valid JSON (no markdown, no explanation):
{{"command": "COMMAND_NAME", "target": "target_id", "value": null}}
"""


def env_call(endpoint, json_data=None):
    try:
        if json_data is not None:
            resp = requests.post(f"{ENV_URL}/{endpoint}", json=json_data, timeout=30)
        else:
            resp = requests.post(f"{ENV_URL}/{endpoint}", timeout=30)
        return resp.json()
    except Exception as e:
        print(f"  [ENV ERROR] {endpoint}: {e}")
        return {}


def run_single_task(client, model_name, task_id, difficulty):
    """Run one model on one task. Returns (score, steps, success)."""
    try:
        env_call("set_level", {"task_level": task_id})
        resp = env_call("reset", {})
        done = resp.get("done", False)
        reward = 0.001
        steps = 0

        for step in range(1, 16):
            if done:
                break
            steps = step

            obs = resp.get("observation", resp)
            logs = obs.get("hardware_logs", [])
            q = obs.get("queue_depths", {})
            v = obs.get("gradient_variances", [])
            m = obs.get("gpu_memory_usage", [])

            prompt = SRE_PROMPT.format(logs=logs, q=q, v=v, m=m)

            try:
                ans = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.0,
                )
                ai_reply = ans.choices[0].message.content.strip()

                # Extract JSON from response
                match = re.search(r"\{[\s\S]*?\}", ai_reply)
                if match:
                    payload = json.loads(match.group(0))
                else:
                    payload = json.loads(ai_reply)

                # Normalize value
                if "value" in payload and payload["value"] is not None:
                    try:
                        payload["value"] = int(payload["value"])
                    except:
                        payload["value"] = None

            except json.JSONDecodeError as je:
                print(f"    [JSON ERR] step {step}: {ai_reply[:80]}")
                payload = {"command": "UNKNOWN", "target": "NULL", "value": None}
            except Exception as e:
                print(f"    [API ERR] step {step}: {str(e)[:60]}")
                payload = {"command": "UNKNOWN", "target": "NULL", "value": None}

            resp = env_call("step", {"action": payload})
            done = resp.get("done", False)
            reward = float(resp.get("reward", 0.001))

        final_score = max(0.001, min(0.999, reward))
        success = final_score > 0.1
        return (final_score, steps, success)

    except Exception as e:
        print(f"    [TASK ERR] {task_id}: {str(e)[:60]}")
        return (0.001, 0, False)


def run_benchmark():
    print("=" * 70)
    print("  NETWEAVER SRE -- REAL MODEL BENCHMARK")
    print("  Started:", datetime.now().isoformat())
    print("  Environment:", ENV_URL)
    print("  API:", API_BASE)
    print("=" * 70)

    # Check server
    try:
        resp = requests.get(f"{ENV_URL}/state", timeout=5)
        assert resp.status_code == 200
        print("\n[OK] Local server is running")
    except:
        print("\n[ERROR] Local server not running!")
        sys.exit(1)

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE)

    all_results = {}

    for mi, model_name in enumerate(MODELS_TO_TEST):
        short = model_name.split("/")[-1]
        print(f"\n{'='*70}")
        print(f"  [{mi+1}/{len(MODELS_TO_TEST)}] BENCHMARKING: {short}")
        print(f"{'='*70}")

        model_scores = []

        for ti, (task_id, difficulty) in enumerate(TASKS):
            score, steps, success = run_single_task(
                client, model_name, task_id, difficulty
            )
            model_scores.append(round(score, 3))
            marker = "[OK]" if success else "[  ]"
            tag = f"[{difficulty.upper():6s}]"
            print(
                f"  {tag} {task_id.upper()}: {score:.3f} "
                f"(steps={steps}) {marker}"
            )
            time.sleep(0.3)  # light rate limit

        total = sum(model_scores)
        avg = total / 20
        resolved = sum(1 for s in model_scores if s >= 0.5)

        all_results[model_name] = {
            "short_name": short,
            "scores": model_scores,
            "total": round(total, 3),
            "average": round(avg, 4),
            "resolved": resolved,
        }

        print(f"\n  SUMMARY: {short}")
        print(f"     Total: {total:.3f} / 20")
        print(f"     Average: {avg:.4f}")
        print(f"     Resolved: {resolved} / 20")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        f"benchmark_results_{timestamp}.json",
    )
    with open(results_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "environment": "Netweaver SRE (local)",
                "api_base": API_BASE,
                "tasks": [t[0] for t in TASKS],
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\n[SAVED] Results: {results_file}")

    # Final leaderboard
    print("\n" + "=" * 70)
    print("  FINAL LEADERBOARD")
    print("=" * 70)
    sorted_models = sorted(
        all_results.items(), key=lambda x: x[1]["total"], reverse=True
    )
    for rank, (model, data) in enumerate(sorted_models, 1):
        print(
            f"  #{rank:2d}  {data['short_name']:40s} "
            f"Total: {data['total']:6.3f}  "
            f"Avg: {data['average']:.4f}  "
            f"Resolved: {data['resolved']:2d}/20"
        )

    print("\n" + "=" * 70)
    print("  Completed:", datetime.now().isoformat())
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    run_benchmark()
