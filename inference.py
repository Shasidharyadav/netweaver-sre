"""
inference.py — OpenEnv-compliant inference script for Netweaver SRE.

Runs ALL 3 tasks in a single invocation using HTTP requests to the HF Space.
Uses API_KEY and API_BASE_URL env vars injected by the validator.
Emits [START], [STEP], [END] lines with strict score clamping.
"""

import os
import json
import requests
from openai import OpenAI

# ── Environment variables (validator injects these) ──────────────────────────
API_KEY      = os.environ.get("API_KEY")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_URL      = os.environ.get("ENV_URL", "https://Shasidharyadavr-netweaver-sre.hf.space")

# ── Validate API_KEY ─────────────────────────────────────────────────────────
if not API_KEY:
    # Fallback for local testing only
    API_KEY = os.environ.get("HF_TOKEN", "")
    if not API_KEY:
        raise ValueError("API_KEY environment variable is required (set API_KEY or HF_TOKEN for local testing)")

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL,
)

# ── Constants ────────────────────────────────────────────────────────────────
BENCHMARK = "netweaver_sre"
MAX_STEPS = 15

# Task definitions: (task_id, difficulty, level_for_server)
TASKS = [
    ("netweaver_sre_easy",   "easy",   "easy"),
    ("netweaver_sre_medium", "medium", "medium"),
    ("netweaver_sre_hard",   "hard",   "hard"),
]

# ── Task prompts ─────────────────────────────────────────────────────────────
PROMPTS = {
    "EASY": """You are an Autonomous Site Reliability Engineer (SRE).
System Logs: {logs}

Task: Identify the exact node name that is offline from the logs (e.g., 'node_54').
Respond ONLY with a valid JSON object (no markdown) containing precisely:
{{"command": "DRAIN_TRAFFIC", "target": "the_identified_node_name"}}""",

    "MEDIUM": """You are an Autonomous Site Reliability Engineer (SRE).
System Logs: {logs}

Task: There is a buffer congestion warning. Identify the target threshold number.
Respond ONLY with a valid JSON object (no markdown) containing precisely:
{{"command": "TUNE_PFC_THRESHOLD", "target": "", "value": threshold_integer}}""",

    "HARD": """You are an Autonomous Site Reliability Engineer.
System Logs: {logs}
Observation: {variances}

Task: Find the NaN source and isolate it.
1. The Observation explicitly tells you the Index X.
2. Output JSON: {{"command": "RUN_MINI_ITERATION", "target": "X-X"}} where X is the given index.
3. Once the logs say "TRIAGE CONFIRMED: NaN source is node_XX", output: {{"command": "DRAIN_TRAFFIC", "target": "node_XX"}}.

Respond ONLY with a valid JSON object.""",
}

# ── Logging helpers ──────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    action_str = action.replace("\n", "").replace("\r", "")[:200]
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(task: str, success: bool, steps: int, score: float, rewards: list) -> None:
    """Emit [END] line with task= and score= fields per spec."""
    # CRITICAL: Clamp score strictly between 0 and 1 (exclusive)
    score = float(score)
    score = max(0.001, min(0.999, score))
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] task={task} success={str(success).lower()} "
        f"steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ── HTTP Environment Client ─────────────────────────────────────────────────
def env_set_level(level: str) -> dict:
    """Pin the task level on the server."""
    try:
        resp = requests.post(
            f"{ENV_URL}/set_level",
            json={"task_level": level},
            timeout=30,
        )
        return resp.json()
    except Exception as e:
        print(f"[DEBUG] set_level failed: {e}", flush=True)
        return {}

def env_reset() -> dict:
    """Reset the environment and return the response."""
    resp = requests.post(
        f"{ENV_URL}/reset",
        json={},
        timeout=30,
    )
    return resp.json()

def env_step(action: dict) -> dict:
    """Execute an action and return the response.
    
    The OpenEnv HTTP server expects: {"action": {command, target, ...}}
    """
    resp = requests.post(
        f"{ENV_URL}/step",
        json={"action": action},
        timeout=30,
    )
    return resp.json()

# ── Main agent loop ──────────────────────────────────────────────────────────
def run_episode(task_id: str, difficulty: str, level: str) -> None:
    """Run a single episode for the given task and emit START/STEP/END logs."""
    rewards_list: list = []
    steps_taken  = 0
    score        = 0.001  # Safe default — never exactly 0
    success      = False

    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        # Pin level on the server
        env_set_level(level)

        # Reset environment
        reset_resp = env_reset()
        obs_data = reset_resp.get("observation", {})
        done     = reset_resp.get("done", False)

        hardware_logs       = obs_data.get("hardware_logs", [])
        gradient_variances  = obs_data.get("gradient_variances", [0.01] * 16)
        current_mode        = difficulty.upper()

        chat_history: list = []

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            steps_taken = step

            # Detect current mode from logs
            for log in hardware_logs:
                if "Running mode:" in log:
                    current_mode = log.split("Running mode:")[1].strip()

            # Build prompt
            if not chat_history:
                raw_prompt = PROMPTS.get(current_mode, PROMPTS["EASY"])
                clean_logs = [log.replace("(e.g., target='0-5')", "") for log in hardware_logs]

                if current_mode == "HARD":
                    spike_i = max(range(len(gradient_variances)), key=lambda x: gradient_variances[x])
                    var_str = f"Spike detected at Index {spike_i}. You MUST use target '{spike_i}-{spike_i}'."
                else:
                    var_str = ", ".join(f"Index {i}: {v}" for i, v in enumerate(gradient_variances))

                system_msg = raw_prompt.format(logs=str(clean_logs), variances=var_str)
                chat_history.append({"role": "user", "content": system_msg})
            else:
                clean_logs = [log.replace("(e.g., target='0-5')", "") for log in hardware_logs]
                msg = f"New system logs after your last action: {clean_logs}."
                if current_mode == "HARD":
                    spike_i = max(range(len(gradient_variances)), key=lambda x: gradient_variances[x])
                    var_str = f"Spike detected at Index {spike_i}. If you haven't yet, use target '{spike_i}-{spike_i}'."
                    msg += f" Observation: {var_str}."
                msg += " What is your next action JSON?"
                chat_history.append({"role": "user", "content": msg})

            # LLM call — goes through validator's proxy via API_KEY + API_BASE_URL
            error_msg = None
            try:
                ans = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=chat_history,
                    max_tokens=500,
                    temperature=0.0,
                )
                ai_reply = ans.choices[0].message.content.strip()
            except Exception as e:
                ai_reply  = '{"command": "UNKNOWN", "target": ""}'
                error_msg = str(e)

            chat_history.append({"role": "assistant", "content": ai_reply})

            # Parse action
            try:
                payload = json.loads(ai_reply)
            except json.JSONDecodeError:
                payload   = {"command": "UNKNOWN", "target": ""}
                error_msg = "JSONDecodeError"

            # Build action for environment
            action_payload = {
                "command": payload.get("command", "UNKNOWN"),
                "target":  str(payload.get("target", "")),
            }
            if payload.get("value") is not None:
                action_payload["value"] = int(payload["value"])

            # Step environment via HTTP
            step_resp = env_step(action_payload)
            obs_data  = step_resp.get("observation", {})
            done      = step_resp.get("done", False)

            hardware_logs      = obs_data.get("hardware_logs", [])
            gradient_variances = obs_data.get("gradient_variances", [0.01] * 16)

            reward = step_resp.get("reward", 0.0)
            if reward is None:
                reward = 0.0
            reward = float(reward)

            rewards_list.append(reward)

            log_step(
                step=step,
                action=json.dumps(payload),
                reward=reward,
                done=done,
                error=error_msg,
            )

            if done:
                break

        # Compute final score — clamp strictly to (0, 1) exclusive
        if rewards_list:
            raw_score = rewards_list[-1]
        else:
            raw_score = 0.0
        raw_score = float(raw_score) if raw_score is not None else 0.0
        score   = max(0.001, min(0.999, raw_score))
        success = score > 0.1

    except Exception as e:
        print(f"[DEBUG] Task {task_id} error: {e}", flush=True)
        score   = 0.001  # Never exactly 0
        success = False

    finally:
        # ALWAYS emit [END], even on crash
        log_end(task=task_id, success=success, steps=steps_taken, score=score, rewards=rewards_list)


def main():
    """Run all 3 task levels in sequence so the validator sees 3 graded tasks."""
    print(f"[DEBUG] Starting inference with ENV_URL={ENV_URL}", flush=True)
    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)

    for task_id, difficulty, level in TASKS:
        print(f"\n[DEBUG] === Running task: {task_id} (difficulty={difficulty}) ===", flush=True)
        run_episode(task_id, difficulty, level)

    print("\n[DEBUG] All tasks complete.", flush=True)


if __name__ == "__main__":
    main()