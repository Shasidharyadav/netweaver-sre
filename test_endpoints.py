"""Test the local server endpoints to verify score clamping."""
import requests
import json

BASE = "http://localhost:7860"

def test_all_levels():
    for level in ["easy", "medium", "hard"]:
        print(f"\n{'='*60}")
        print(f"Testing level: {level}")
        print(f"{'='*60}")
        
        # Set level
        r = requests.post(f"{BASE}/set_level", json={"task_level": level})
        print(f"set_level: {r.json()}")
        
        # Reset
        r = requests.post(f"{BASE}/reset", json={})
        data = r.json()
        obs = data["observation"]
        print(f"reset done={data['done']} reward={data.get('reward')}")
        print(f"logs: {obs['hardware_logs']}")
        
        # Determine action based on level
        if level == "easy":
            for log in obs["hardware_logs"]:
                if "Isolate" in log:
                    node = log.split("Isolate ")[1].split(" ")[0]
            action = {"command": "DRAIN_TRAFFIC", "target": node}
        elif level == "medium":
            for log in obs["hardware_logs"]:
                if "Target:" in log:
                    target_val = int(float(log.split("Target: ")[1].split(")")[0]))
            action = {"command": "TUNE_PFC_THRESHOLD", "target": "", "value": target_val}
        elif level == "hard":
            variances = obs["gradient_variances"]
            spike_i = max(range(len(variances)), key=lambda x: variances[x])
            action = {"command": "RUN_MINI_ITERATION", "target": f"{spike_i}-{spike_i}"}
        
        print(f"action: {action}")
        
        # Step
        r = requests.post(f"{BASE}/step", json={"action": action})
        step_data = r.json()
        reward = step_data.get("reward")
        done = step_data.get("done")
        print(f"step: reward={reward} done={done}")
        print(f"reward type: {type(reward).__name__}")
        
        # If hard mode and not done, need second step
        if level == "hard" and not done:
            obs2 = step_data.get("observation", {})
            logs2 = obs2.get("hardware_logs", [])
            print(f"hard mode logs after triage: {logs2}")
            
            # Find node from TRIAGE CONFIRMED line
            node_name = None
            for log in logs2:
                if "TRIAGE CONFIRMED" in log and "NaN source is" in log:
                    node_name = log.split("NaN source is ")[1].split(" ")[0]
            
            if node_name:
                action2 = {"command": "DRAIN_TRAFFIC", "target": node_name}
                print(f"drain action: {action2}")
                r = requests.post(f"{BASE}/step", json={"action": action2})
                step_data = r.json()
                reward = step_data.get("reward")
                done = step_data.get("done")
                print(f"drain step: reward={reward} done={done}")
        
        # Check grader
        r = requests.post(f"{BASE}/grader", json={})
        g = r.json()
        score = g["score"]
        print(f"\nGRADER: score={score} type={type(score).__name__}")
        print(f"  0 < score < 1: {0 < score < 1}")
        print(f"  score != 0.0: {score != 0.0}")
        print(f"  score != 1.0: {score != 1.0}")
        
        if score <= 0.0 or score >= 1.0:
            print(f"  *** FAIL: Score {score} is out of range! ***")
        else:
            print(f"  PASS: Score {score} is valid")

if __name__ == "__main__":
    test_all_levels()
    print("\n" + "="*60)
    print("All endpoint tests complete!")
