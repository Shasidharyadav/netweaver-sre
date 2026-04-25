import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open("training_results.json") as f:
    r = json.load(f)
from scripts.run_training_demo import _plot_with_pillow
_plot_with_pillow(r)
print("plots regenerated")
for n in ["reward_curve", "loss_curve", "baseline_vs_trained", "before_after", "difficulty_breakdown"]:
    p = f"server/assets/{n}.png"
    print(f"  {n}.png: {os.path.getsize(p)} bytes")
