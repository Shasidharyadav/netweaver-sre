"""Verify parse_action handles all the value-field junk LLMs emit."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_grpo import parse_action

cases = [
    ('{"command":"DRAIN_TRAFFIC","target":"node_07","value":null}', None),
    ('{"command":"INCREASE_MTU","target":"sw_01","value":"none"}', None),
    ('{"command":"INCREASE_MTU","target":"sw_01","value":"NULL"}', None),
    ('{"command":"INCREASE_MTU","target":"sw_01","value":""}', None),
    ('{"command":"INCREASE_MTU","target":"sw_01","value":"undefined"}', None),
    ('{"command":"INCREASE_MTU","target":"sw_01","value":9000}', 9000),
    ('{"command":"INCREASE_MTU","target":"sw_01","value":9000.0}', 9000),
    ('{"command":"INCREASE_MTU","target":"sw_01","value":"9000"}', 9000),
    ('{"command":"INCREASE_MTU","target":"sw_01","value":"abc"}', None),
    ('not json', None),
    ('{}', None),
    ('   {"command": "drain_traffic", "target": " node_99 ", "value": "none"}  ', None),
]

ok = True
for inp, expected_value in cases:
    p = parse_action(inp)
    got = p["value"]
    flag = "OK" if got == expected_value else "FAIL"
    print(f"  {flag}  value={got!r:>10}  (expected {expected_value!r:>5})  <- {inp[:55]!r}")
    if got != expected_value:
        ok = False

print("\nALL PASSED" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
