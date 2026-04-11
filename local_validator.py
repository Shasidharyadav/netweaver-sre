"""
local_validator.py — Pre-submission validation for OpenEnv.

Checks:
1. openenv.yaml has 3+ tasks with inline grader: blocks
2. inference.py has proper [START]/[END] format
3. Task IDs match between openenv.yaml and inference.py
4. Score clamping is in place
"""

import yaml
import re
import sys


def check_openenv_yaml():
    """Verify openenv.yaml has 3+ tasks with inline graders."""
    print("=" * 60)
    print("CHECK 1: openenv.yaml — Tasks with inline graders")
    print("=" * 60)

    try:
        with open("openenv.yaml", "r") as f:
            d = yaml.safe_load(f)
    except FileNotFoundError:
        print("  FAIL: openenv.yaml not found!")
        return False, []

    tasks = d.get("tasks", [])
    if not tasks:
        print("  FAIL: No 'tasks' block found in openenv.yaml")
        return False, []

    ok = 0
    task_ids = []
    for t in tasks:
        tid = t.get("id", "UNKNOWN")
        task_ids.append(tid)
        has_grader = t.get("grader") is not None
        grader_type = t.get("grader", {}).get("type", "N/A") if has_grader else "N/A"
        grader_endpoint = t.get("grader", {}).get("endpoint", "N/A") if has_grader else "N/A"
        status = "OK" if has_grader else "FAIL"
        print(f"  {status}  {tid}")
        print(f"       grader={has_grader} type={grader_type} endpoint={grader_endpoint}")
        if has_grader:
            ok += 1

    print(f"\n  Tasks with graders: {ok}/{len(tasks)}")
    passed = ok >= 3
    print(f"  {'PASS' if passed else 'FAIL'}: need at least 3")
    return passed, task_ids


def check_inference_py(yaml_task_ids):
    """Verify inference.py has matching task IDs and proper format."""
    print("\n" + "=" * 60)
    print("CHECK 2: inference.py — Task IDs and format")
    print("=" * 60)

    try:
        with open("inference.py", "r") as f:
            content = f.read()
    except FileNotFoundError:
        print("  FAIL: inference.py not found!")
        return False

    all_ok = True

    # Check API_KEY usage (not HF_TOKEN as primary)
    if 'os.environ.get("API_KEY")' in content:
        print("  OK   Uses API_KEY env var")
    else:
        print("  FAIL  Missing os.environ.get('API_KEY')")
        all_ok = False

    # Check API_BASE_URL usage
    if 'os.environ.get("API_BASE_URL"' in content:
        print("  OK   Uses API_BASE_URL env var")
    else:
        print("  FAIL  Missing API_BASE_URL")
        all_ok = False

    # Check for from_docker_image
    if "from_docker_image" in content:
        print("  FAIL  Uses from_docker_image() — must use HTTP requests")
        all_ok = False
    else:
        print("  OK   No from_docker_image() usage")

    # Check [START] format
    if "[START] task=" in content:
        print("  OK   Has [START] task= format")
    else:
        print("  FAIL  Missing [START] task= format")
        all_ok = False

    # Check [END] format with task= and score=
    if "task=" in content and "score=" in content:
        print("  OK   Has [END] with task= and score= fields")
    else:
        print("  FAIL  Missing task= or score= in [END] line")
        all_ok = False

    # Check score clamping
    if "0.001" in content and "0.999" in content:
        print("  OK   Score clamping to (0.001, 0.999)")
    else:
        print("  WARN  Score clamping might be missing")

    # Check that TASKS list matches openenv.yaml task IDs
    print("\n  Task ID matching:")
    for tid in yaml_task_ids:
        if tid in content:
            print(f"    OK   '{tid}' found in inference.py")
        else:
            print(f"    FAIL '{tid}' NOT found in inference.py")
            all_ok = False

    # Check task count in TASKS list
    task_count = content.count('"netweaver_sre_')
    print(f"\n  Task references in code: {task_count}")
    
    # Check finally: block for guaranteed [END]
    if "finally:" in content:
        print("  OK   Has finally: block for guaranteed [END] emission")
    else:
        print("  WARN  No finally: block — [END] may not emit on crash")

    return all_ok


def check_environment():
    """Verify the environment clamps scores."""
    print("\n" + "=" * 60)
    print("CHECK 3: Environment — Score clamping")
    print("=" * 60)

    try:
        with open("server/netweaver_sre_environment.py", "r") as f:
            content = f.read()
    except FileNotFoundError:
        print("  FAIL: server/netweaver_sre_environment.py not found!")
        return False

    if "clamp_score" in content:
        print("  OK   Uses clamp_score() function")
    else:
        print("  WARN  No clamp_score() found — check manual clamping")

    if "0.001" in content and "0.999" in content:
        print("  OK   Clamp boundaries set to (0.001, 0.999)")
    else:
        print("  FAIL  Clamp boundaries might allow 0.0 or 1.0")
        return False

    # Check for exact 0.0 or 1.0 reward returns
    lines = content.split("\n")
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if "reward" in stripped and ("= 0.0" in stripped or "= 1.0" in stripped):
            if "clamp_score" not in stripped and "raw_score" not in stripped:
                print(f"  WARN  Line {i}: Possible unclamped reward: {stripped}")

    return True


def main():
    print("\n>>> OpenEnv Pre-Submission Validator")
    print("=" * 60)

    yaml_ok, task_ids = check_openenv_yaml()
    inference_ok = check_inference_py(task_ids)
    env_ok = check_environment()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    results = [
        ("openenv.yaml — 3+ tasks with graders", yaml_ok),
        ("inference.py — format and task IDs", inference_ok),
        ("Environment — score clamping", env_ok),
    ]

    all_pass = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("*** All checks passed! Ready to submit. ***")
    else:
        print("!!! Fix the above issues before submitting. !!!")
        sys.exit(1)


if __name__ == "__main__":
    main()
