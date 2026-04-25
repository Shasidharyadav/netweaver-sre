"""Re-render NetWeaver SRE training plots from a JSON log.

Usage:
    python scripts/plot_results.py [--input training_results.json]

Outputs:
    server/assets/reward_curve.png
    server/assets/before_after.png
    server/assets/difficulty_breakdown.png

Tries matplotlib first; falls back to a Pillow-only renderer if matplotlib
is unavailable (some locked-down Windows environments block its native
DLLs).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot NetWeaver SRE training results.")
    parser.add_argument(
        "--input",
        default="training_results.json",
        help="Path to JSON log file produced by train_grpo.py / run_training_demo.py",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Reuse the same plotter as the training demo (matplotlib -> Pillow fallback)
    from scripts.run_training_demo import _plot_all
    _plot_all(results)
    print("Saved: server/assets/{reward_curve,before_after,difficulty_breakdown}.png")


if __name__ == "__main__":
    main()
