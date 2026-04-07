import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from client import NetweaverSreEnv, NetweaverSreAction

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
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
TASK_NAME = "netweaver_sre"
BENCHMARK = "netweaver_sre"

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Remove any newlines or spaces inside action json to keep it on a single line
    action_str = action.replace('\n', '').replace('\r', '')
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def run_agent():
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
            chat_history = []
            
            log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
            
            rewards_list = []
            steps_taken = 0
            
            while not obs.done:
                steps_taken += 1
                
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
                    msg = f"New system logs after your last action: {obs.hardware_logs}."
                    if current_mode == "HARD":
                        msg += f" Gradient Variances: {obs.gradient_variances}."
                    msg += " What is your next action JSON?"
                    chat_history.append({"role": "user", "content": msg})
                
                try:
                    ans = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=chat_history,
                        response_format={"type": "json_object"}
                    )
                    ai_reply = ans.choices[0].message.content
                except Exception as e:
                    ai_reply = '{"command": "UNKNOWN", "target": ""}'
                
                chat_history.append({"role": "assistant", "content": ai_reply})
                
                try:
                    payload = json.loads(ai_reply)
                    error_msg = None
                except json.JSONDecodeError:
                    payload = {"command": "UNKNOWN", "target": ""}
                    error_msg = "JSONDecodeError"
                
                action = NetweaverSreAction(
                    command=payload.get("command", "UNKNOWN"),
                    target=str(payload.get("target", "")),
                    value=int(payload["value"]) if payload.get("value") is not None else None
                )
                
                obs_res = env.step(action)
                obs = obs_res.observation
                
                reward = obs.reward if hasattr(obs, 'reward') else (obs_res.reward if hasattr(obs_res, 'reward') else 0.0)
                if reward is None:
                    reward = 0.0
                    
                rewards_list.append(reward)
                log_step(step=steps_taken, action=json.dumps(payload), reward=reward, done=obs.done, error=error_msg)

            # Score is out of 1.0; since reward could be > 0.0 only at the end.
            final_reward = rewards_list[-1] if rewards_list else 0.0
            score = min(max(final_reward, 0.0), 1.0)
            success = score > 0.1
            
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards_list)
            
            # Save Transcript quietly
            os.makedirs("logs", exist_ok=True)
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"logs/transcript_{current_mode}_{timestamp}.json"
            transcript_data = {
                "mode": current_mode,
                "final_reward": final_reward,
                "chat_history": chat_history
            }
            with open(log_filename, "w") as f:
                json.dump(transcript_data, f, indent=4)

    except Exception as e:
        # Need to log END even on exception
        log_end(success=False, steps=0, score=0.0, rewards=[])

if __name__ == "__main__":
    run_agent()
