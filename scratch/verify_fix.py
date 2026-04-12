import requests
import json

BASE_URL = "http://localhost:8000"

def verify():
    print("--- 1. Testing /set_level with 't01' ---")
    resp = requests.post(f"{BASE_URL}/set_level", json={"task_level": "t01"})
    print(f"Set Level Response: {resp.json()}")

    print("\n--- 2. Testing /reset ---")
    resp = requests.post(f"{BASE_URL}/reset")
    data = resp.json()
    obs = data.get("observation", {})
    print(f"Reset Obs Keys: {list(obs.keys())}")
    print(f"Step Count: {obs.get('step_count')}")
    print(f"Logs: {obs.get('hardware_logs')}")

    # Extract faulty node from logs
    logs = obs.get('hardware_logs', [])
    node = "node_45" # Default fallback
    for log in logs:
        if "node_" in log:
            node = log.split("node_")[1].split(" ")[0].strip(".").strip(",")
            node = f"node_{node}"
            break
    
    print(f"\n--- 3. Testing /step with {node} and DRAIN_TRAFFIC ---")
    action = {
        "command": "DRAIN_TRAFFIC",
        "target": node,
        "value": None
    }
    resp = requests.post(f"{BASE_URL}/step", json={"action": action})
    step_data = resp.json()
    print(f"Top-level Response Keys: {list(step_data.keys())}")
    
    step_obs = step_data.get("observation", {})
    print(f"Observation Keys: {list(step_obs.keys())}")
    
    # Try to get reward from both places
    reward = step_data.get('reward') if step_data.get('reward') is not None else step_obs.get('reward')
    done = step_data.get('done') if step_data.get('done') is not None else step_obs.get('done')
    step_count = step_obs.get('step_count')
    
    print(f"Found Reward: {reward}")
    print(f"Found Done: {done}")
    print(f"Found Step Count: {step_count}")

    if reward is not None and reward > 0.9:
        print("\n[SUCCESS] Scoring confirmed!")
    else:
        print("\n[FAILURE] Scoring or Task Selection issue remains.")

if __name__ == "__main__":
    try:
        verify()
    except Exception as e:
        print(f"Error: {e}")
