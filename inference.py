import os
import json
from openai import OpenAI
from client import NetweaverSreEnv, NetweaverSreAction

# ── Environment variables ────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

# HF_TOKEN is MANDATORY — no default, raise immediately if absent
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
)

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

# ── Config ────────────────────────────────────────────────────────────────────
ENV_URL      = os.getenv("ENV_URL", "https://Shasidharyadavr-netweaver-sre.hf.space")
FORCE_LEVEL  = os.getenv("FORCE_TASK_LEVEL", "")
TASK_NAME    = "netweaver_sre"
BENCHMARK    = "netweaver_sre"

# ── Logging helpers ───────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    action_str = action.replace("\n", "").replace("\r", "")
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, rewards: list) -> None:
    """Emit [END] line exactly matching the spec (no extra fields)."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )

# ── Main agent loop ───────────────────────────────────────────────────────────
def run_episode(env, level: str) -> None:
    """Run a single episode for the given level and emit START/STEP/END logs."""
    import requests as _req

    # Pin level on the server
    try:
        _req.post(f"{ENV_URL}/set_level", json={"task_level": level}, timeout=5)
    except Exception:
        pass

    rewards_list: list[float] = []
    steps_taken  = 0
    success      = False

    obs_res      = env.reset()
    obs          = obs_res.observation
    current_mode = level.upper()
    chat_history: list[dict] = []

    log_start(task=f"{TASK_NAME}_{level}", env=BENCHMARK, model=MODEL_NAME)

    try:
        while not obs.done:
            steps_taken += 1

            # Detect task difficulty from cluster init log
            for log in obs.hardware_logs:
                if "Running mode:" in log:
                    current_mode = log.split("Running mode:")[1].strip()

            if not chat_history:
                raw_prompt = PROMPTS.get(current_mode, PROMPTS["EASY"])
                clean_logs = [log.replace("(e.g., target='0-5')", "") for log in obs.hardware_logs]
                
                if current_mode == "HARD":
                    spike_i = max(range(len(obs.gradient_variances)), key=lambda x: obs.gradient_variances[x])
                    var_str = f"Spike detected at Index {spike_i}. You MUST use target '{spike_i}-{spike_i}'."
                else:
                    var_str = ", ".join(f"Index {i}: {v}" for i, v in enumerate(obs.gradient_variances))
                    
                system_msg = raw_prompt.format(
                    logs=str(clean_logs),
                    variances=var_str,
                )
                chat_history.append({"role": "user", "content": system_msg})
            else:
                clean_logs = [log.replace("(e.g., target='0-5')", "") for log in obs.hardware_logs]
                msg = f"New system logs after your last action: {clean_logs}."
                if current_mode == "HARD":
                    spike_i = max(range(len(obs.gradient_variances)), key=lambda x: obs.gradient_variances[x])
                    var_str = f"Spike detected at Index {spike_i}. If you haven't yet, use target '{spike_i}-{spike_i}'."
                    msg += f" Observation: {var_str}."
                msg += " What is your next action JSON?"
                chat_history.append({"role": "user", "content": msg})

            error_msg = None
            try:
                ans      = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=chat_history,
                    response_format={"type": "json_object"},
                )
                ai_reply = ans.choices[0].message.content
            except Exception as e:
                ai_reply  = '{"command": "UNKNOWN", "target": ""}'
                error_msg = str(e)

            chat_history.append({"role": "assistant", "content": ai_reply})

            try:
                payload = json.loads(ai_reply)
            except json.JSONDecodeError:
                payload   = {"command": "UNKNOWN", "target": ""}
                error_msg = "JSONDecodeError"

            action = NetweaverSreAction(
                command=payload.get("command", "UNKNOWN"),
                target=str(payload.get("target", "")),
                value=int(payload["value"]) if payload.get("value") is not None else None,
            )

            obs_res = env.step(action)
            obs     = obs_res.observation

            reward = getattr(obs, "reward", None) or getattr(obs_res, "reward", 0.01) or 0.01
            rewards_list.append(reward)
            log_step(step=steps_taken, action=json.dumps(payload), reward=reward, done=obs.done, error=error_msg)

        final_reward = rewards_list[-1] if rewards_list else 0.01
        success      = final_reward > 0.1
        log_end(success=success, steps=steps_taken, rewards=rewards_list)

        # Save transcript
        import datetime, pathlib
        pathlib.Path("logs").mkdir(exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"logs/transcript_{current_mode}_{ts}.json", "w") as f:
            json.dump(
                {"mode": current_mode, "final_reward": final_reward, "chat_history": chat_history},
                f, indent=2,
            )

    except Exception:
        log_end(success=False, steps=steps_taken, rewards=rewards_list)
        raise


def run_agent():
    """Run all 3 task levels in sequence so the validator sees 3 graded tasks."""
    levels = ["easy", "medium", "hard"] if not FORCE_LEVEL else [FORCE_LEVEL]

    with NetweaverSreEnv(base_url=ENV_URL).sync() as env:
        for level in levels:
            run_episode(env, level)


if __name__ == "__main__":
    run_agent()