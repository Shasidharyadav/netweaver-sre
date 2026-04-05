import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from server.netweaver_sre_environment import NetweaverSreEnvironment
from models import NetweaverSreAction

def test_all_levels():
    env = NetweaverSreEnvironment()
    
    # --- EASY MODE ---
    print("\n" + "="*40)
    print(">>> [EASY] Testing DRAIN_TRAFFIC")
    obs = env.reset(task_level="easy")
    print(f"Log: {obs.hardware_logs[-1]}")
    faulty = env._faulty_node_id
    print(f"Action: DRAIN_TRAFFIC on {faulty}")
    obs = env.step(NetweaverSreAction(command="DRAIN_TRAFFIC", target=faulty))
    print(f"Result Reward: {obs.reward} | Done: {obs.done}")
    
    # --- MEDIUM MODE ---
    print("\n" + "="*40)
    print(">>> [MEDIUM] Testing TUNE_PFC_THRESHOLD")
    obs = env.reset(task_level="medium")
    print(f"Log: {obs.hardware_logs[-1]}")
    target_pfc = env._target_pfc
    
    # 1st Action: Incorrect PFC
    wrong_pfc = target_pfc + 10
    print(f"Action 1 (Bad Tune): TUNE_PFC_THRESHOLD to {wrong_pfc}")
    obs = env.step(NetweaverSreAction(command="TUNE_PFC_THRESHOLD", target="", value=wrong_pfc))
    print(f"Log: {obs.hardware_logs[-1]}")
    
    # 2nd Action: Correct PFC
    print(f"Action 2 (Correct Tune): TUNE_PFC_THRESHOLD to {target_pfc}")
    obs = env.step(NetweaverSreAction(command="TUNE_PFC_THRESHOLD", target="", value=int(target_pfc)))
    print(f"Result Reward: {obs.reward} | Done: {obs.done}")
    
    # --- HARD MODE ---
    print("\n" + "="*40)
    print(">>> [HARD] Testing RUN_MINI_ITERATION & DRAIN_TRAFFIC")
    obs = env.reset(task_level="hard")
    print(f"Log: {obs.hardware_logs[-1]}")
    
    # Peek at truth
    faulty_idx = int(env._faulty_node_id.split("_")[1]) // 10
    
    # 1st Action: Query wrong range
    bad_start, bad_end = (faulty_idx + 1) % 10, (faulty_idx + 2) % 10
    print(f"Action 1 (Triage Miss): RUN_MINI_ITERATION for {bad_start}-{bad_end}")
    obs = env.step(NetweaverSreAction(command="RUN_MINI_ITERATION", target=f"{bad_start}-{bad_end}"))
    print(f"Log: {obs.hardware_logs[-1]}")
    
    # 2nd Action: Query correct range
    print(f"Action 2 (Triage Hit): RUN_MINI_ITERATION for {faulty_idx}-{faulty_idx}")
    obs = env.step(NetweaverSreAction(command="RUN_MINI_ITERATION", target=f"{faulty_idx}-{faulty_idx}"))
    print(f"Log: {obs.hardware_logs[-1]}")
    
    # 3rd Action: Drain the correct node directly
    print(f"Action 3 (Isolate): DRAIN_TRAFFIC on {env._faulty_node_id}")
    obs = env.step(NetweaverSreAction(command="DRAIN_TRAFFIC", target=env._faulty_node_id))
    print(f"Result Reward: {obs.reward} | Done: {obs.done}")

if __name__ == "__main__":
    test_all_levels()
