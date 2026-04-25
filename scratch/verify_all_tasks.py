"""Verify each of the 22 tasks resolves with the heuristic playbook.

Runs ONE episode per task with the optimal playbook (epsilon=0).
Expected: ALL 22 tasks reach `done=True` and `resolved=True` with
the rubric grader returning a high `total`.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from server.app import app
from scripts.run_training_demo import _run_episode, PLAYBOOK

CLIENT = TestClient(app)


def main() -> None:
    print("Running each task once with the optimal heuristic playbook...\n")
    results = []
    for level in sorted(PLAYBOOK.keys()):
        score = _run_episode(level, epsilon=0.0, max_steps=8)
        # Read /grader after to see breakdown + resolved
        g = CLIENT.get("/grader").json()
        ok = "OK" if g.get("resolved") else "PARTIAL"
        results.append((level, ok, g.get("total", 0.0), g.get("breakdown")))
        br = g.get("breakdown", {})
        print(f"  {level:4s}  {ok:7s}  total={g.get('total', 0.0):.3f}  "
              f"diag={br.get('diagnosis', 0):.2f}  res={br.get('resolution', 0):.2f}  "
              f"bp={br.get('best_practice', 0):.2f}")

    n_resolved = sum(1 for _, ok, *_ in results if ok == "OK")
    avg = sum(t for *_, t, _ in results) / len(results)
    print(f"\nSummary: {n_resolved}/{len(results)} resolved, avg total={avg:.3f}")
    if n_resolved < len(results):
        print("FAIL: not all tasks resolved")
        sys.exit(1)
    print("PASS: all 22 tasks resolved")


if __name__ == "__main__":
    main()
