"""Verify composed reward functions in train_grpo.py."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_grpo import (
    reward_action_parses,
    reward_correct_command,
    parse_action,
)


def main():
    # Good action
    prompts = ["[TASK=t01] alert: Node node_07 is offline."]
    completions = ['{"command":"DRAIN_TRAFFIC","target":"node_07","value":null}']
    p = reward_action_parses(prompts, completions)
    c = reward_correct_command(prompts, completions)
    print(f"Good T01 DRAIN_TRAFFIC node_07: parse={p[0]:+.1f} command={c[0]:+.1f}")
    assert p[0] > 0 and c[0] > 0, "good action should be rewarded"

    # Wrong command for fault type
    completions = ['{"command":"INCREASE_MTU","target":"sw_core_01","value":9000}']
    p = reward_action_parses(prompts, completions)
    c = reward_correct_command(prompts, completions)
    print(f"Bad  T01 INCREASE_MTU:           parse={p[0]:+.1f} command={c[0]:+.1f}")
    assert p[0] > 0 and c[0] < 0, "wrong-fault command should be penalised"

    # Unparseable
    completions = ["I am not a JSON object at all."]
    p = reward_action_parses(prompts, completions)
    c = reward_correct_command(prompts, completions)
    print(f"Junk completion:                  parse={p[0]:+.1f} command={c[0]:+.1f}")
    assert p[0] < 0, "junk should be penalised"

    # T22 (multi-step) — RUN_MINI_ITERATION is valid for gradient_poisoning
    prompts = ["[TASK=t22] alert: NaN poisoning on cluster_2"]
    completions = ['{"command":"RUN_MINI_ITERATION","target":"cluster_2","value":null}']
    c = reward_correct_command(prompts, completions)
    print(f"Good T22 RUN_MINI_ITERATION:     parse=+1.0 command={c[0]:+.1f}")
    assert c[0] > 0

    print("\n=== ALL REWARD-FUNC TESTS PASSED ===")


if __name__ == "__main__":
    main()
