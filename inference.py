import os
import json
import requests
import re
from openai import OpenAI

API_KEY      = os.environ.get("API_KEY")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME") or "meta-llama/Meta-Llama-3.1-70B-Instruct"
ENV_URL      = os.environ.get("ENV_URL", "http://0.0.0.0:8000")

if not API_KEY:
    API_KEY = os.environ.get("HF_TOKEN", "")
    if not API_KEY:
        raise ValueError("API_KEY required")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

TASKS = [
    ("netweaver_sre_t01", "easy", "t01"),
    ("netweaver_sre_t02", "easy", "t02"),
    ("netweaver_sre_t03", "easy", "t03"),
    ("netweaver_sre_t04", "easy", "t04"),
    ("netweaver_sre_t05", "easy", "t05"),
    ("netweaver_sre_t06", "easy", "t06"),
    ("netweaver_sre_t07", "easy", "t07"),
    ("netweaver_sre_t08", "medium", "t08"),
    ("netweaver_sre_t09", "medium", "t09"),
    ("netweaver_sre_t10", "medium", "t10"),
    ("netweaver_sre_t11", "medium", "t11"),
    ("netweaver_sre_t12", "medium", "t12"),
    ("netweaver_sre_t13", "medium", "t13"),
    ("netweaver_sre_t14", "medium", "t14"),
    ("netweaver_sre_t15", "hard", "t15"),
    ("netweaver_sre_t16", "hard", "t16"),
    ("netweaver_sre_t17", "hard", "t17"),
    ("netweaver_sre_t18", "hard", "t18"),
    ("netweaver_sre_t19", "hard", "t19"),
    ("netweaver_sre_t20", "hard", "t20"),
]

PROMPTS = {
    "GENERIC": """You are an Autonomous Site Reliability Engineer.
System Logs: {logs}
Observation Queue: {q}
Observation Gradients: {v}
Observation GPU Memory: {m}

Look at the hardware logs. Determine the required Command, Target, and Value.
If logs are inconclusive, look at the queues or GPU arrays for 99.9 spikes and clear the corresponding subsystem.
If everything is 0.0, ISSUE_GLOBAL_ROLLBACK on cluster_0.
Return your answer in the following JSON format ONLY:
{{
  "command": "STRING",
  "target": "STRING",
  "value": NUMBER_OR_NULL
}}
Ensure the JSON is strictly valid. No markdown, no explanations.
"""
}

def log_start(task, env, model): print(f"[START] task={task} env={env} model={model}", flush=True)
def log_step(step, action, reward, done, error=None):
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)
def log_end(task, success, steps, score, rewards):
    s = max(0.001, min(0.999, float(score)))
    rewards_str = ",".join(f"{float(r):.2f}" for r in rewards)
    print(f"[END] task={task} success={str(success).lower()} steps={steps} score={s:.3f} rewards={rewards_str}", flush=True)

def env_call(endpoint, json_data):
    return requests.post(f"{ENV_URL}/{endpoint}", json=json_data, timeout=30).json()

def run_episode(task_id, difficulty, level):
    rewards_list = []
    success, steps_taken, score = False, 0, 0.001
    log_start(task_id, "netweaver_sre", MODEL_NAME)

    try:
        env_call("set_level", {"task_level": level})
        resp = env_call("reset", {})
        done = resp.get("done", False)

        for step in range(1, 16):
            if done: break
            steps_taken = step
            obs = resp.get("observation", {})
            logs = obs.get("hardware_logs", [])
            q = obs.get("queue_depths", {})
            v = obs.get("gradient_variances", [])
            m = obs.get("gpu_memory_usage", [])

            sys_msg = PROMPTS["GENERIC"].format(logs=logs, q=q, v=v, m=m)
            chat = [{"role": "user", "content": sys_msg}]

            try:
                ans = client.chat.completions.create(model=MODEL_NAME, messages=chat, max_tokens=100)
                ai_reply = ans.choices[0].message.content.strip()
                match = re.search(r"\{[\s\S]*\}", ai_reply)
                if match:
                    json_str = match.group(0)
                else:
                    json_str = ai_reply
                payload = json.loads(json_str)
            except Exception as e:
                print(f"JSON Parse Error: {e}")
                payload = {"command": "UNKNOWN", "target": "NULL"}

            resp = env_call("step", {"action": payload})
            done = resp.get("done", False)
            reward = float(resp.get("reward", 0.0))
            rewards_list.append(reward)
            log_step(step, json.dumps(payload), reward, done)

        score = max(0.001, min(0.999, rewards_list[-1] if rewards_list else 0.0))
        success = score > 0.1

    except Exception as e:
        print(f"[DEBUG] {task_id} error: {e}", flush=True)
        score = 0.001
    finally:
        log_end(task_id, success, steps_taken, score, rewards_list)

if __name__ == "__main__":
    for tid, dif, lev in TASKS:
        run_episode(tid, dif, lev)