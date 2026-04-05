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

Task: A NaN contagion has infected the network. Use binary search to locate it:

RULES (follow strictly):
1. Use RUN_MINI_ITERATION to narrow the sub-cluster range: {{"command": "RUN_MINI_ITERATION", "target": "start-end"}}
   - Start broad: "0-9", then split the HIT range in half each time
   - e.g. if 0-9 hits → try "0-4" and "5-9" next
2. NEVER issue DRAIN_TRAFFIC until you see a log that starts with "TRIAGE CONFIRMED"
3. Only when you see "TRIAGE CONFIRMED: NaN source is node_XX" → issue: {{"command": "DRAIN_TRAFFIC", "target": "node_XX"}}
   - Copy the EXACT node name from the TRIAGE CONFIRMED message (e.g., node_54)
   - Never guess a node name

Respond ONLY with a valid JSON object (no markdown)."""
}

ENV_URL = os.getenv("ENV_URL", "https://Shasidharyadavr-netweaver-sre.hf.space")

def run_agent():
    print("[START] Initializing NetWeaver-SRE Evaluation")
    print(f"[INFO]  Environment: {ENV_URL}")
    print(f"[INFO]  Model: {MODEL_NAME}")
    print(f"[INFO]  API Base: {API_BASE_URL}")
    print("-" * 50)
    
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
                    system_msg = raw_prompt.format(logs=str(obs.hardware_logs))
                    chat_history.append({"role": "user", "content": system_msg})
                else:
                    # Append the latest triage result as a new user turn so AI adapts
                    chat_history.append({
                        "role": "user",
                        "content": f"New system logs after your last action: {obs.hardware_logs}. What is your next action JSON?"
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
    except Exception as e:
        print(f"[END] Error: {e}")

if __name__ == "__main__":
    run_agent()
