"""Live Hard mode test — uses /set_level to pin difficulty, then runs inference."""
import os
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
ENV_URL = os.getenv("ENV_URL", "https://Shasidharyadavr-netweaver-sre.hf.space")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

HARD_PROMPT = """You are an Autonomous Site Reliability Engineer (SRE).
System Logs: {logs}

Task: A NaN contagion has infected the network. Use binary search to locate it:

RULES (follow strictly):
1. Use RUN_MINI_ITERATION to narrow the sub-cluster range: {{"command": "RUN_MINI_ITERATION", "target": "start-end"}}
   - Start broad: "0-9", then split the HIT range in half each time
   - e.g. if 0-9 hits try "0-4" next; if miss try "5-9"
2. NEVER issue DRAIN_TRAFFIC until you see a log starting with "TRIAGE CONFIRMED"
3. When you see "TRIAGE CONFIRMED: NaN source is node_XX" -> issue: {{"command": "DRAIN_TRAFFIC", "target": "node_XX"}}
   - Copy the EXACT node name from TRIAGE CONFIRMED (e.g. node_54, not node_5)

Respond ONLY with a valid JSON object (no markdown)."""

def parse_obs(data: dict) -> tuple[list, bool, float]:
    """Handle both flat and nested observation structures from OpenEnv."""
    # Try nested structure first
    obs = data.get("observation", data)
    hardware_logs = obs.get("hardware_logs", [])
    done = data.get("done", obs.get("done", False))
    reward = data.get("reward", obs.get("reward", 0.0))
    return hardware_logs, done, reward

def run_hard_test():
    print("[START] HARD MODE Live Test on HF Space")
    print(f"[INFO] {ENV_URL}")
    print("-" * 50)

    # Step 1: Pin the level and immediately reset (must happen in sequence)
    lvl_resp = requests.post(f"{ENV_URL}/set_level", json={"task_level": "hard"})
    print(f"[SET_LEVEL] {lvl_resp.json()}")

    # Step 2: Reset AFTER setting level, also pass task_level in reset body as extra safety
    reset_resp = requests.post(f"{ENV_URL}/reset", json={"task_level": "hard"})
    reset_data = reset_resp.json()
    hardware_logs, done, _ = parse_obs(reset_data)
    print(f"[RESET] logs={hardware_logs}")

    chat_history = []
    step = 0

    while not done and step < 15:
        step += 1
        last_log = hardware_logs[-1] if hardware_logs else "No log"
        print(f"[STEP {step}] {last_log}")

        if not chat_history:
            chat_history.append({"role": "user", "content": HARD_PROMPT.format(logs=hardware_logs)})
        else:
            chat_history.append({"role": "user", "content": f"New logs: {hardware_logs}. What is your next JSON action?"})

        ans = client.chat.completions.create(
            model=MODEL_NAME,
            messages=chat_history,
            response_format={"type": "json_object"}
        )
        ai_reply = ans.choices[0].message.content
        chat_history.append({"role": "assistant", "content": ai_reply})

        payload = json.loads(ai_reply)
        print(f"[AI]   -> {payload}")

        step_resp = requests.post(f"{ENV_URL}/step", json={
            "action": {
                "command": payload.get("command", "UNKNOWN"),
                "target": str(payload.get("target", "")),
                "value": payload.get("value")
            }
        })
        step_data = step_resp.json()
        hardware_logs, done, reward = parse_obs(step_data)

    print(f"[END] Final Reward: {reward}")

if __name__ == "__main__":
    run_hard_test()
