import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from client import NetweaverSreEnv, NetweaverSreAction

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(
    api_key=HF_TOKEN or "dummy-key",
    base_url=API_BASE_URL
)

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

    "HARD": """You are an Autonomous Site Reliability Engineer (SRE).
System Logs: {logs}
Gradient Variances (Indexes 0-9): {variances}

Task: A NaN contagion has infected the network. Do NOT blindly guess. Use the observations to instantly find it:

STRATEGY:
1. Look at 'Gradient Variances'. Find the array index (0-9) that has an extremely high variance (e.g., 999.9). This index 'X' is the faulty sub-cluster.
2. Immediately run a mini iteration exactly on that index to confirm the node: {{"command": "RUN_MINI_ITERATION", "target": "X-X"}}
3. The system will then log "TRIAGE CONFIRMED: NaN source is node_XX". Only then issue {{"command": "DRAIN_TRAFFIC", "target": "node_XX"}}.

Respond ONLY with a valid JSON object (no markdown)."""
}

ENV_URL = os.getenv("ENV_URL", "https://Shasidharyadavr-netweaver-sre.hf.space")
FORCE_LEVEL = os.getenv("FORCE_TASK_LEVEL", "")  # e.g. "hard", "medium", "easy"

def run_agent():
    print("[START] Initializing NetWeaver-SRE Evaluation")
    print(f"[INFO]  Environment: {ENV_URL}")
    print(f"[INFO]  Model: {MODEL_NAME}")
    print(f"[INFO]  API Base: {API_BASE_URL}")
    if FORCE_LEVEL:
        print(f"[INFO]  Forcing level: {FORCE_LEVEL.upper()}")
    print("-" * 50)

    # Pin task level via HTTP before opening WebSocket session
    if FORCE_LEVEL:
        try:
            import requests
            requests.post(f"{ENV_URL}/set_level", json={"task_level": FORCE_LEVEL}, timeout=5)
        except Exception:
            pass  # non-fatal, server will pick randomly

    try:
        with NetweaverSreEnv(base_url=ENV_URL).sync() as env:
            obs_res = env.reset()
            obs = obs_res.observation
            
            current_mode = "EASY"
            # Growing conversation history so AI remembers all triage results
            chat_history = []
            
            while not obs.done:
                last_log = obs.hardware_logs[-1] if obs.hardware_logs else 'No Logs'
                print(f"[STEP] Logs: {last_log}")
                
                # Dynamically detect task difficulty from the CLUSTER INIT log
                for log in obs.hardware_logs:
                    if "Running mode:" in log:
                        current_mode = log.split("Running mode:")[1].strip()
                
                # Build the system prompt once (first step), then append observations
                if not chat_history:
                    raw_prompt = PROMPTS.get(current_mode, PROMPTS["EASY"])
                    system_msg = raw_prompt.format(logs=str(obs.hardware_logs), variances=str(obs.gradient_variances))
                    chat_history.append({"role": "user", "content": system_msg})
                else:
                    # Append the latest triage result as a new user turn so AI adapts
                    msg = f"New system logs after your last action: {obs.hardware_logs}."
                    if current_mode == "HARD":
                        msg += f" Gradient Variances: {obs.gradient_variances}."
                    msg += " What is your next action JSON?"
                    
                    chat_history.append({
                        "role": "user",
                        "content": msg
                    })
                
                ans = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=chat_history,
                    response_format={"type": "json_object"}
                )
                
                ai_reply = ans.choices[0].message.content
                # Add AI reply to history so it remembers its own prior decisions
                chat_history.append({"role": "assistant", "content": ai_reply})
                
                try:
                    payload = json.loads(ai_reply)
                except json.JSONDecodeError:
                    print("[STEP] Warning: AI returned malformed JSON.")
                    payload = {"command": "UNKNOWN", "target": ""}
                
                action = NetweaverSreAction(
                    command=payload.get("command", "UNKNOWN"),
                    target=str(payload.get("target", "")),
                    value=int(payload["value"]) if payload.get("value") is not None else None
                )
                
                obs_res = env.step(action)
                obs = obs_res.observation

            print(f"[END] Evaluation Completed. Final Reward: {obs.reward}")
            
            # ---- Save Transcript ----
            os.makedirs("logs", exist_ok=True)
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"logs/transcript_{current_mode}_{timestamp}.json"
            transcript_data = {
                "mode": current_mode,
                "final_reward": obs.reward,
                "chat_history": chat_history
            }
            with open(log_filename, "w") as f:
                json.dump(transcript_data, f, indent=4)
            print(f"[INFO] Transcript saved to {log_filename}")

    except Exception as e:
        print(f"[END] Error: {e}")

if __name__ == "__main__":
    run_agent()
