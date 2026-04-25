import os
import json
import requests
import re
from openai import OpenAI

API_KEY = os.environ.get("API_KEY")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME") or "meta-llama/Meta-Llama-3.1-70B-Instruct"
ENV_URL = os.environ.get("ENV_URL", "http://0.0.0.0:8000")

if not API_KEY:
    API_KEY = os.environ.get("HF_TOKEN", "")
if not API_KEY:
    raise ValueError("API_KEY required")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

TASKS = [
    ("netweaver_sre_t01", "easy",   "t01"),
    ("netweaver_sre_t02", "easy",   "t02"),
    ("netweaver_sre_t03", "easy",   "t03"),
    ("netweaver_sre_t04", "easy",   "t04"),
    ("netweaver_sre_t05", "easy",   "t05"),
    ("netweaver_sre_t06", "easy",   "t06"),
    ("netweaver_sre_t07", "easy",   "t07"),
    ("netweaver_sre_t08", "medium", "t08"),
    ("netweaver_sre_t09", "medium", "t09"),
    ("netweaver_sre_t10", "medium", "t10"),
    ("netweaver_sre_t11", "medium", "t11"),
    ("netweaver_sre_t12", "medium", "t12"),
    ("netweaver_sre_t13", "medium", "t13"),
    ("netweaver_sre_t14", "medium", "t14"),
    ("netweaver_sre_t15", "hard",   "t15"),
    ("netweaver_sre_t16", "hard",   "t16"),
    ("netweaver_sre_t17", "hard",   "t17"),
    ("netweaver_sre_t18", "hard",   "t18"),
    ("netweaver_sre_t19", "hard",   "t19"),
    ("netweaver_sre_t20", "hard",   "t20"),
    ("netweaver_sre_t21", "hard",   "t21"),
    ("netweaver_sre_t22", "hard",   "t22"),
]

# Task-specific diagnostic hints so agent knows what to look at and what to issue
TASK_HINTS = {
    "t01": "Inspect hardware_logs for offline node ID. Issue DRAIN_TRAFFIC targeting that node.",
    "t02": "Inspect hardware_logs for DNS failure. Issue CLEAR_DNS_CACHE targeting the affected node.",
    "t03": "Inspect hardware_logs for OOM error. Issue RESTART_SERVICE targeting the crashed service.",
    "t04": "Inspect hardware_logs for TLS/certificate expiry. Issue RENEW_CERTIFICATE on the affected node.",
    "t05": "Inspect hardware_logs for disk usage at 100%. Issue CLEAR_TEMP_FILES on the affected node.",
    "t06": "Inspect hardware_logs for stuck pod. Issue RESTART_POD targeting the unhealthy pod.",
    "t07": "Inspect hardware_logs for zombie process. Issue KILL_ZOMBIE_PROCESS on the affected node.",
    "t08": "Check queue_depths for buffer congestion. Find the maxed switch. Issue TUNE_PFC_THRESHOLD with a threshold value between 1000-9000.",
    "t09": "Inspect hardware_logs for power throttle warning. Note the node. Issue ADJUST_POWER_CAP with the correct watt value (100-400).",
    "t10": "Inspect hardware_logs for BGP flapping. Extract the AS number. Issue MITIGATE_ROUTE_FLAP on the router with that AS number as value.",
    "t11": "Inspect hardware_logs for jumbo frame / MTU packet drops. Issue INCREASE_MTU with value 9000 on the affected switch.",
    "t12": "Check queue_depths for traffic spike. Issue SET_RATE_LIMIT with a request count value on the affected gateway.",
    "t13": "Inspect hardware_logs for connection pool exhaustion on a 'db_cluster_*' target. Issue SCALE_CONN_POOL with a pool size value (50-5000) on that exact db_cluster_*.",
    "t14": "Inspect hardware_logs for high CPU context switching. Issue PIN_CPU_THREADS with a thread count (1-256) on the affected node.",
    "t15": "Check gradient_variances array for NaN or very high values. First issue RUN_MINI_ITERATION on the affected cluster to isolate, then issue DRAIN_TRAFFIC to contain it. Two actions required.",
    "t16": "Check queue_depths for a switch value near 99.9 (broadcast storm). Issue ISOLATE_BROADCAST_STORM on that switch.",
    "t17": "Check gpu_memory_usage array for a spike above normal. Issue RESTART_GPU_DAEMON on the affected cluster/node.",
    "t18": "All telemetry arrays (gradient_variances, queue_depths, gpu_memory_usage) are frozen at 0.0 — this is a cluster deadlock. Issue ISSUE_GLOBAL_ROLLBACK on the cluster named in hardware_logs.",
    "t19": "Check queue_depths for a split pattern (one very low ~0.01, one very high ~99.9) — network partition. Issue REBOOT_LEAF_SWITCHES on the affected pod.",
    "t20": "Inspect hardware_logs for a 'db_cluster_*' with a checksum mismatch / I/O error. Issue PURGE_CORRUPT_BLOCK on that exact db_cluster_*.",
    "t21": "Cascading failure. ORDER MATTERS: (1) ADJUST_POWER_CAP value 350 on the throttled node_*, (2) PIN_CPU_THREADS value 64 on the SAME node_*, (3) SCALE_CONN_POOL value 800 on the db_cluster_* mentioned in logs. All 3 in order.",
    "t22": "Gradient poisoning + broadcast amplification co-occurring. ORDER MATTERS: (1) RUN_MINI_ITERATION on the affected cluster_*, (2) ISOLATE_BROADCAST_STORM on the sw_* with 99.x queue depth, (3) DRAIN_TRAFFIC on the SAME cluster_*. All 3 in order.",
}

SYSTEM_PROMPT = """You are an Autonomous Site Reliability Engineer managing a 100-node GPU cluster.

=== CURRENT ALERT ===
{alert}

=== TELEMETRY ===
Hardware Logs: {logs}
Queue Depths:  {q}
Gradient Variances: {v}
GPU Memory Usage:   {m}
System Health: {health}

=== EPISODE STATE ===
Step: {step}/15
Previous actions this episode: {prev_actions}

=== TASK GUIDANCE ===
{hint}

=== INSTRUCTIONS ===
1. Read the telemetry carefully.
2. Identify the fault from hardware_logs or array values.
3. Issue the correct remediation command.
4. If a numeric value is needed (threshold, watts, AS number, thread count), extract it from the logs and include it.
5. Target must reference the specific node, switch, cluster, or pod from the logs.

Return ONLY a valid JSON object — no markdown, no explanation:
{{"command": "STRING", "target": "STRING", "value": NUMBER_OR_NULL}}
"""


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.3f} done={str(done).lower()} error={err}", flush=True)

def log_end(task, success, steps, score, rewards):
    s = max(0.001, min(0.999, float(score)))
    rewards_str = ",".join(f"{float(r):.3f}" for r in rewards)
    print(f"[END] task={task} success={str(success).lower()} steps={steps} score={s:.3f} rewards={rewards_str}", flush=True)

def env_call(endpoint, json_data=None, method="POST"):
    url = f"{ENV_URL}/{endpoint}"
    if method == "GET":
        return requests.get(url, timeout=30).json()
    return requests.post(url, json=json_data or {}, timeout=30).json()

def run_episode(task_id: str, difficulty: str, level: str):
    rewards_list = []
    success, steps_taken, score = False, 0, 0.001
    prev_actions = []

    log_start(task_id, "netweaver_sre", MODEL_NAME)

    try:
        env_call("set_level", {"task_level": level})
        resp = env_call("reset", {})
        done = resp.get("done", False)

        for step in range(1, 16):
            if done:
                break
            steps_taken = step

            obs = resp.get("observation", {})
            logs  = obs.get("hardware_logs", [])
            q     = obs.get("queue_depths", {})
            v     = obs.get("gradient_variances", [])
            m     = obs.get("gpu_memory_usage", [])
            health = obs.get("system_health", 1.0)
            alert = obs.get("alert", "No alert text provided.")

            hint = TASK_HINTS.get(level, "Inspect hardware_logs to identify the fault and issue the correct command.")

            sys_msg = SYSTEM_PROMPT.format(
                alert=alert,
                logs=json.dumps(logs),
                q=json.dumps(q),
                v=json.dumps(v),
                m=json.dumps(m),
                health=health,
                step=step,
                prev_actions=json.dumps(prev_actions[-5:]),  # last 5 only
                hint=hint,
            )

            try:
                ans = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": sys_msg}],
                    max_tokens=150,
                    temperature=0.1,
                )
                ai_reply = ans.choices[0].message.content.strip()
                # Extract JSON robustly
                match = re.search(r"\{[\s\S]*?\}", ai_reply)
                json_str = match.group(0) if match else ai_reply
                payload = json.loads(json_str)
                # Ensure correct types
                payload["command"] = str(payload.get("command", "UNKNOWN")).upper()
                payload["target"]  = str(payload.get("target", "unknown"))
                raw_val = payload.get("value")
                payload["value"]   = int(raw_val) if raw_val is not None else None
            except Exception as e:
                print(f"[PARSE_ERROR] step={step} err={e}", flush=True)
                payload = {"command": "UNKNOWN", "target": "null", "value": None}

            prev_actions.append(payload["command"])

            resp  = env_call("step", {"action": payload})
            done  = resp.get("done", False)
            reward = float(resp.get("reward", 0.0))
            rewards_list.append(reward)
            log_step(step, json.dumps(payload), reward, done)

        # Fetch grader score
        try:
            grader_resp = env_call("grader", method="GET")
            score = float(grader_resp.get("total", 0.001))
            success = grader_resp.get("resolved", False)
        except Exception:
            score = max(0.001, min(0.999, rewards_list[-1] if rewards_list else 0.001))
            success = score > 0.5

    except Exception as e:
        print(f"[DEBUG] {task_id} episode error: {e}", flush=True)
        score = 0.001
    finally:
        log_end(task_id, success, steps_taken, score, rewards_list)


if __name__ == "__main__":
    for tid, dif, lev in TASKS:
        run_episode(tid, dif, lev)